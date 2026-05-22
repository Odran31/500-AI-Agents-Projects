import json
import requests
import hashlib
from typing import List, Dict
from statistics import mean
import math
import os
import time
import re

# ==============================
# CONFIG
# ==============================

API_KEY    = os.environ.get("NVIDIA_API_KEY", "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")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL      = "meta/llama-3.1-70b-instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

CACHE_FILE      = "llm_cache.json"
RETRIEVER_K     = 3
API_DELAY       = 1.5
REQUEST_TIMEOUT = 60


# ==============================
# CACHE LLM
# ==============================

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

llm_cache = load_cache()

def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


# ==============================
# LLM HELPERS
# ==============================

def extract_json_from_text(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {text[:300]}")


def call_llm(prompt: str, retries: int = 5) -> str:
    key = hash_prompt(prompt)
    if key in llm_cache:
        print("  [CACHE HIT]")
        return llm_cache[key]

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,   # plus déterministe pour les scores
        "top_p": 0.9,
        "stream": False,
    }

    for attempt in range(retries):
        wait = 2 ** attempt if attempt > 0 else 0
        if wait:
            print(f"  [RETRY {attempt}/{retries}] waiting {wait}s...")
            time.sleep(wait)
        try:
            r = requests.post(INVOKE_URL, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code in (500, 504):
                print(f"  [HTTP {r.status_code}] attempt {attempt+1}/{retries}...")
                continue
            if r.status_code != 200:
                print(f"  [HTTP {r.status_code}] {r.text[:200]}")
                return json.dumps({"error": f"http_{r.status_code}"})
            data = r.json()
            if "choices" in data:
                result = data["choices"][0]["message"]["content"]
                llm_cache[key] = result
                save_cache(llm_cache)
                time.sleep(API_DELAY)
                return result
            if "error" in data:
                return json.dumps({"error": str(data["error"])})
        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] attempt {attempt+1}/{retries}")
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            return json.dumps({"error": str(e)})

    return json.dumps({"error": "max_retries_exceeded"})


# ==============================
# UTILS
# ==============================

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(score, scale: float = 5.0) -> float:
    try:
        return round(float(score) / scale, 4)
    except (TypeError, ValueError):
        return 0.0

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


# ==============================
# METRICS RETRIEVER (lexical)
# ==============================

def precision_at_k(retrieved, expected, k):
    if not k: return 0
    r = [normalize_text(t) for t in retrieved[:k]]
    e = [normalize_text(t) for t in expected]
    return len(set(r) & set(e)) / k

def recall_at_k(retrieved, expected, k):
    if not expected: return 0
    r = [normalize_text(t) for t in retrieved[:k]]
    e = [normalize_text(t) for t in expected]
    return len(set(r) & set(e)) / len(e)

def mrr(retrieved, expected):
    e = [normalize_text(t) for t in expected]
    for i, doc in enumerate(retrieved):
        if normalize_text(doc) in e:
            return 1 / (i + 1)
    return 0

def ndcg_at_k(retrieved, expected, k):
    e = [normalize_text(t) for t in expected]
    dcg  = sum(1/math.log2(i+2) for i, d in enumerate(retrieved[:k]) if normalize_text(d) in e)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(e), k)))
    return dcg / idcg if idcg > 0 else 0


# ==============================
# RETRIEVER SEMANTIQUE (LLM)
# ==============================

def evaluate_retriever_semantic(question: str, retrieved_docs: list, expected_docs: list) -> dict:
    """
    Le LLM juge si les documents récupérés contiennent les informations
    nécessaires pour répondre à la question, comparé aux documents de référence.
    Score 1-5 sur la pertinence globale.
    """
    if not expected_docs:
        return {"semantic_relevance": 0, "coverage": 0, "warning": "No expected docs"}

    prompt = f"""You are a strict RAG retrieval evaluator. Your job is to assess whether the retrieved documents contain the information needed to answer the question.

QUESTION: {question}

REFERENCE DOCUMENTS (ground truth — what should ideally be retrieved):
{chr(10).join(f"[REF {i+1}] {d}" for i, d in enumerate(expected_docs))}

RETRIEVED DOCUMENTS (what the system actually retrieved):
{chr(10).join(f"[RET {i+1}] {d}" for i, d in enumerate(retrieved_docs))}

Evaluate strictly on TWO dimensions:

1. semantic_relevance (1-5): Do the RETRIEVED documents contain information relevant to the question?
   - 5: All retrieved docs are highly relevant to the question
   - 4: Most retrieved docs are relevant
   - 3: Some retrieved docs are relevant
   - 2: Few retrieved docs are relevant
   - 1: Retrieved docs are mostly irrelevant to the question

2. coverage (1-5): How well do the RETRIEVED documents cover the same information as the REFERENCE documents?
   - 5: Retrieved docs cover all key facts from the reference docs
   - 4: Retrieved docs cover most key facts
   - 3: Retrieved docs cover some key facts
   - 2: Retrieved docs miss most key facts
   - 1: Retrieved docs share almost no information with reference docs

Be STRICT. A score of 5 should be rare. Do not round up.

Return ONLY this JSON (no explanation):
{{
  "semantic_relevance": <1-5>,
  "coverage": <1-5>
}}"""

    raw = call_llm(prompt)
    try:
        data = extract_json_from_text(raw)
        return {
            "semantic_relevance": normalize(data.get("semantic_relevance", 1)),
            "coverage":           normalize(data.get("coverage", 1))
        }
    except Exception as e:
        print(f"  [PARSE ERROR retriever_semantic] {e} | raw={raw[:200]}")
        return {"semantic_relevance": 0, "coverage": 0, "error": str(e)}


# ==============================
# EVALUATE RETRIEVER (lexical + sémantique)
# ==============================

def evaluate_retriever(gt, gen) -> dict:
    expected_docs  = [doc["text"] for doc in gt.get("retrieved_docs", [])]
    retrieved_docs = [doc["text"] for doc in gen.get("retrieved_docs", [])]
    question       = gt.get("question", "")

    if not expected_docs:
        return {"precision": 0, "recall": 0, "mrr": 0, "ndcg": 0,
                "semantic_relevance": 0, "coverage": 0, "warning": "No expected docs"}

    k = RETRIEVER_K

    # Métriques lexicales
    lexical = {
        "precision": precision_at_k(retrieved_docs, expected_docs, k),
        "recall":    recall_at_k(retrieved_docs, expected_docs, k),
        "mrr":       mrr(retrieved_docs, expected_docs),
        "ndcg":      ndcg_at_k(retrieved_docs, expected_docs, k)
    }

    # Métriques sémantiques (LLM)
    semantic = evaluate_retriever_semantic(question, retrieved_docs, expected_docs)

    return {**lexical, **semantic}


# ==============================
# EVALUATE GENERATION
# ==============================

def evaluate_generation(q: str, gen_item: dict, gt_item: dict) -> dict:
    gen_ans      = gen_item.get("answer", "")
    gt_ans       = gt_item.get("answer", "")
    context_docs = [d["text"] for d in gen_item.get("retrieved_docs", [])]

    if not gen_ans:
        return {"error": "empty answer"}

    prompt = f"""You are a strict RAG answer quality evaluator. Score the generated answer on 4 dimensions.

QUESTION: {q}

REFERENCE ANSWER (ground truth):
{gt_ans}

GENERATED ANSWER:
{gen_ans}

CONTEXT DOCUMENTS used to generate the answer:
{chr(10).join(f"[DOC {i+1}] {d}" for i, d in enumerate(context_docs))}

Evaluate STRICTLY on these 4 dimensions (score 1 to 5 each):

1. exactitude: Is every factual claim in the generated answer correct and supported by the context?
   - 5: All facts are correct and grounded in context
   - 4: Minor inaccuracy or one unsupported claim
   - 3: Some correct facts but notable errors or hallucinations
   - 2: Several factual errors
   - 1: Mostly wrong or hallucinated

2. completude: Does the answer cover all important points from the reference answer?
   - 5: All key points covered
   - 4: Most key points covered, minor omission
   - 3: Some key points missing
   - 2: Many important points missing
   - 1: Answer is very incomplete

3. clarte: Is the answer clear, well-structured, and easy to understand?
   - 5: Perfectly clear and well-structured
   - 4: Clear with minor issues
   - 3: Somewhat clear but could be better organized
   - 2: Confusing or poorly structured
   - 1: Very unclear

4. coherence: Is the answer internally consistent and logically coherent?
   - 5: Fully coherent, no contradictions
   - 4: Mostly coherent, minor inconsistency
   - 3: Some inconsistencies
   - 2: Notable contradictions
   - 1: Incoherent

Be STRICT and CRITICAL. Most answers should score between 2 and 4. A score of 5 means the answer is perfect. A score of 1 means it is clearly wrong.

Compare carefully against the reference answer — if the generated answer misses key details that are in the reference, penalize completude.

Return ONLY this JSON (no explanation, no extra text):
{{
  "exactitude": <1-5>,
  "completude": <1-5>,
  "clarte": <1-5>,
  "coherence": <1-5>
}}"""

    raw = call_llm(prompt)
    try:
        data = extract_json_from_text(raw)
        keys = {"exactitude", "completude", "clarte", "coherence"}
        return {k: normalize(v) for k, v in data.items() if k in keys}
    except Exception as e:
        print(f"  [PARSE ERROR generation] {e} | raw={raw[:200]}")
        return {"error": str(e)}


# ==============================
# EVALUATE E2E
# ==============================

def evaluate_e2e(q: str, gen_item: dict, gt_item: dict) -> dict:
    gen_ans = gen_item.get("answer", "")
    gt_ans  = gt_item.get("answer", "")

    prompt = f"""You are a strict end-to-end RAG evaluator. Compare the generated answer to the reference answer for this question.

QUESTION: {q}

REFERENCE ANSWER (ground truth — this is what a perfect answer looks like):
{gt_ans}

GENERATED ANSWER:
{gen_ans}

Give a single overall score from 1 to 5 based on these criteria:
- 5: The generated answer is equivalent to the reference — same facts, same completeness, no errors
- 4: The generated answer is mostly correct with minor omissions or slight differences in wording
- 3: The generated answer is partially correct — it captures some key information but misses important details or adds inaccuracies
- 2: The generated answer is mostly incorrect, incomplete, or significantly different from the reference
- 1: The generated answer is wrong, hallucinated, or completely misses the point

Be STRICT. Compare every claim in the generated answer against the reference. Penalize:
- Any fact present in the reference but missing in the generated answer
- Any fact in the generated answer that contradicts the reference
- Any hallucinated information not supported by the reference

Return ONLY this JSON (no explanation, no extra text):
{{
  "score_global": <1-5>,
  "reason": "<one sentence explaining the score>"
}}"""

    raw = call_llm(prompt)
    try:
        data = extract_json_from_text(raw)
        return {
            "score":  normalize(data["score_global"]),
            "reason": data.get("reason", "")
        }
    except Exception as e:
        print(f"  [PARSE ERROR e2e] {e} | raw={raw[:200]}")
        return {"error": str(e)}


# ==============================
# PIPELINE
# ==============================

def process_item(gt: dict, gen: dict) -> dict:
    q = gt["question"]
    return {
        "question":   q,
        "retriever":  evaluate_retriever(gt, gen),
        "generation": evaluate_generation(q, gen, gt),
        "e2e":        evaluate_e2e(q, gen, gt)
    }


def evaluate_rag(gt_path: str, gen_path: str) -> dict:
    gt_data  = load_json(gt_path)
    gen_data = load_json(gen_path)

    if isinstance(gt_data,  dict) and "results" in gt_data:
        gt_data  = gt_data["results"]
    if isinstance(gen_data, dict) and "results" in gen_data:
        gen_data = gen_data["results"]

    total   = min(len(gt_data), len(gen_data))
    results = []

    for index, (gt, gen) in enumerate(zip(gt_data, gen_data), start=1):
        print(f"\n[{index}/{total}] {gt['question'][:70]}...")
        results.append(process_item(gt, gen))

    # ── Scores globaux ──────────────────────────────────────────
    global_scores = {}

    # Lexical retriever
    for metric in ["precision", "recall", "mrr", "ndcg"]:
        vals = [r["retriever"][metric] for r in results if metric in r["retriever"]]
        global_scores[f"retriever_{metric}"] = round(mean(vals), 4) if vals else 0

    # Semantic retriever
    for metric in ["semantic_relevance", "coverage"]:
        vals = [r["retriever"][metric] for r in results
                if metric in r["retriever"] and isinstance(r["retriever"][metric], float)]
        global_scores[f"retriever_{metric}"] = round(mean(vals), 4) if vals else 0

    # Generation
    gen_scores = []
    for r in results:
        vals = [v for v in r["generation"].values() if isinstance(v, float)]
        if vals:
            gen_scores.append(mean(vals))
    global_scores["generation_avg"] = round(mean(gen_scores), 4) if gen_scores else 0

    for metric in ["exactitude", "completude", "clarte", "coherence"]:
        vals = [r["generation"][metric] for r in results
                if metric in r["generation"] and isinstance(r["generation"][metric], float)]
        global_scores[f"generation_{metric}"] = round(mean(vals), 4) if vals else 0

    # E2E
    e2e_scores = [r["e2e"]["score"] for r in results if "score" in r["e2e"]]
    global_scores["e2e"] = round(mean(e2e_scores), 4) if e2e_scores else 0

    return {"results": results, "global": global_scores}


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    report = evaluate_rag("ground_truth.json", "generated.json")

    with open("rapport_ameliore.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== SCORES GLOBAUX ===")
    print(json.dumps(report["global"], indent=2))