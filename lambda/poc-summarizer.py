# lambda_function.py (FINAL: includes build_prompt_for_product + Bedrock insight + fallback insight)
"""
Aggregated summarizer + two-email SNS notifications (run summary + full summaries).
Added optional Amazon Bedrock "AI-Generated Insights Email" + deterministic fallback insight.
"""

import os
import sys
import json
import time
import uuid
import logging
import traceback
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

import boto3

# ---------- logging ----------
LOG = logging.getLogger("PoCFastSummarizerAggregated")
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s'))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ---------- config / env ----------
REVIEW_TABLE           = os.environ.get("DYNAMO_TABLE", "PoC-Reviews")
SUMMARY_TABLE          = os.environ.get("SUMMARY_TABLE_NAME", "PoC-ReviewSummaries")
OUTPUT_BUCKET          = os.environ.get("OUTPUT_BUCKET", "poc-sentiment-processed")
SNS_TOPIC_ARN          = os.environ.get("SNS_TOPIC_ARN")
USE_SAGEMAKER          = os.environ.get("USE_SAGEMAKER", "0") in {"1","true","True"}
SAGEMAKER_ENDPOINT     = os.environ.get("ENDPOINT_NAME", "poc-summarizer-endpoint")
MAX_CONCURRENT_TASKS   = int(os.environ.get("MAX_CONCURRENT_TASKS", "8"))
MAX_EXEC_SECONDS       = int(os.environ.get("MAX_EXEC_SECONDS", "270"))
MAX_SCAN_ITEMS         = int(os.environ.get("MAX_SCAN_ITEMS", "5000"))
MAX_REVIEWS_PER_PRODUCT= int(os.environ.get("MAX_REVIEWS_PER_PRODUCT", "10"))
MAX_SUMMARY_CHARS      = int(os.environ.get("MAX_SUMMARY_CHARS", "900"))
MAX_TOTAL_PRODUCTS     = int(os.environ.get("MAX_TOTAL_PRODUCTS", "1000"))
SKIP_EXISTING_SUMMARIES= int(os.environ.get("SKIP_EXISTING_SUMMARIES", "1"))
DEBUG_MODE             = os.environ.get("DEBUG", "0") in {"1","true","True"}

# Bedrock / Insights settings (optional)
USE_BEDROCK            = os.environ.get("USE_BEDROCK", "0") in {"1","true","True"}
BEDROCK_MODEL_ID       = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-lite")
BEDROCK_MAX_OUTPUT_TOKENS = int(os.environ.get("BEDROCK_MAX_OUTPUT_TOKENS", "160"))
BEDROCK_TEMPERATURE    = float(os.environ.get("BEDROCK_TEMPERATURE", "0.0"))

INLINE_SUMMARIES_MAX_BYTES = int(os.environ.get("INLINE_SUMMARIES_MAX_BYTES", str(200 * 1024)))  # 200 KB default
AGGREGATE_S3_PREFIX = os.environ.get("AGGREGATE_S3_PREFIX", "summaries/aggregates/")
PRESIGN_EXPIRY_SECONDS = int(os.environ.get("PRESIGN_EXPIRY_SECONDS", str(60*60)))

BOTOCONFIG = boto3.session.Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2})

# ---------- clients ----------
s3 = boto3.client("s3", config=BOTOCONFIG)
ddb = boto3.resource("dynamodb", config=BOTOCONFIG)
sagemaker_runtime = boto3.client("sagemaker-runtime", config=BOTOCONFIG) if USE_SAGEMAKER else None
sns = boto3.client("sns", config=BOTOCONFIG) if SNS_TOPIC_ARN else None

# Bedrock runtime client (optional)
bedrock_runtime = None
if USE_BEDROCK:
    try:
        bedrock_runtime = boto3.client("bedrock-runtime", config=BOTOCONFIG)
        LOG.info("Bedrock runtime client initialized (model=%s)", BEDROCK_MODEL_ID)
    except Exception as e:
        bedrock_runtime = None
        LOG.warning("Failed to create bedrock-runtime client: %s", str(e))

review_table = ddb.Table(REVIEW_TABLE)
summary_table = ddb.Table(SUMMARY_TABLE)

# ---------- try sklearn (optional) ----------
SKLEARN_AVAILABLE = False
try:
    import numpy as _np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    LOG.info("sklearn & numpy available -> sklearn extractor enabled")
except Exception as e:
    SKLEARN_AVAILABLE = False
    LOG.info("sklearn not available, using fallback extractor (%s)", str(e))

# ---------- helpers ----------
def publish_sns(subject: str, message_obj: Any):
    """
    Publish to SNS. Accepts dict or string. Converts dict to readable JSON string.
    """
    if not sns or not SNS_TOPIC_ARN:
        LOG.debug("SNS not configured; skipping publish: %s", subject)
        return
    try:
        if isinstance(message_obj, (dict, list)):
            message = json.dumps(message_obj, indent=2)
        else:
            message = str(message_obj)
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
    except Exception as e:
        LOG.warning("SNS publish failed [%s]: %s", subject, str(e))

def _now_ts(): return int(time.time())
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

# ---------- scan and grouping ----------
def scan_reviews(limit=MAX_SCAN_ITEMS) -> List[Dict[str, Any]]:
    LOG.info(f"Scanning {REVIEW_TABLE} (limit={limit})")
    items = []
    last_key = None
    page_size = 250
    projection = "ReviewID, ProductID, ReviewText, SentimentScore, CreatedAt"
    while True and len(items) < limit:
        params = {"Limit": min(page_size, max(50, limit - len(items))), "ProjectionExpression": projection}
        if last_key:
            params["ExclusiveStartKey"] = last_key
        try:
            resp = review_table.scan(**params)
        except Exception as e:
            LOG.warning(f"Projection scan failed: {e}; retrying without projection")
            try:
                resp = review_table.scan(Limit=min(page_size, limit - len(items)), ExclusiveStartKey=last_key) if last_key else review_table.scan(Limit=min(page_size, limit - len(items)))
            except Exception as e2:
                LOG.error(f"Full scan failed: {e2}")
                break
        batch = resp.get("Items", []) or []
        items.extend(batch)
        if "LastEvaluatedKey" in resp and len(items) < limit:
            last_key = resp["LastEvaluatedKey"]
        else:
            break
    LOG.info(f"Scan complete — fetched {len(items)} reviews")
    return items

def canonical_product_id(item: Dict[str, Any]) -> str:
    for key in ("ProductID","product_id","productid","product","pid"):
        if key in item:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, (int,float)):
                return str(val)
    rid = str(item.get("ReviewID",""))
    m = re.search(r'(p\d{1,6}|prod[_-]?\d+|p\d+)', rid, re.IGNORECASE)
    return m.group(1) if m else ""

def group_by_product(items: List[Dict[str, Any]]) -> dict:
    groups = defaultdict(list)
    for it in items:
        pid = canonical_product_id(it)
        if pid:
            groups[pid].append(it)
    return groups

# ---------- text utilities ----------
STOPWORDS = {"the","and","for","with","that","this","these","those","from","are","was","were","have","has","had",
    "but","not","too","very","its","it's","in","on","at","to","a","an","of","is","be","by","as","or"}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r'[^a-z0-9\s\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ---------- sklearn-based extractor (optional) ----------
def extract_unique_themes_sklearn(reviews: List[Dict[str, Any]], top_k=3):
    texts = [(r.get("ReviewText") or "").strip() for r in reviews if r.get("ReviewText")]
    if len(texts) == 0:
        return [], []

    # Normalize & dedupe, keep order
    seen = set()
    norm_texts = []
    for t in texts:
        nt = normalize_text(t)
        if len(nt) < 10:
            continue
        if nt not in seen:
            seen.add(nt)
            norm_texts.append(nt)
    texts = norm_texts
    if len(texts) == 0:
        return [], []

    try:
        vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2), stop_words="english")
        X = vectorizer.fit_transform(texts)
    except Exception as e:
        LOG.warning("TF-IDF vectorization failed; falling back. %s", e)
        return [], []

    k = min(3, len(texts))
    if k == 0:
        return [], []

    try:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
    except Exception as e:
        LOG.warning("KMeans failed; fallback. %s", e)
        labels = [0]*len(texts)

    feature_names = []
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        try:
            feature_names = vectorizer.get_feature_names()
        except Exception:
            feature_names = []

    strengths = []
    weaknesses = []

    for cluster_id in range(k):
        idx = [i for i, lab in enumerate(labels) if lab == cluster_id]
        if not idx:
            continue
        try:
            cluster_matrix = X[idx].mean(axis=0)
            scores = cluster_matrix.A1 if hasattr(cluster_matrix, "A1") else cluster_matrix.toarray()[0]
        except Exception:
            try:
                agg = _np.sum(X[idx].toarray(), axis=0)
                scores = agg
            except Exception:
                scores = None

        if scores is None:
            continue

        top_indices = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[0:5]
        phrases = [feature_names[i] for i in top_indices if i < len(feature_names)]

        avg_sent = 0.0
        count_sent = 0
        for i in idx:
            try:
                orig_text = texts[i]
                for r in reviews:
                    if normalize_text(r.get("ReviewText","")) == orig_text:
                        avg_sent += safe_float(r.get("SentimentScore", 0.0))
                        count_sent += 1
                        break
            except Exception:
                continue
        avg_sent = (avg_sent / count_sent) if count_sent else 0.0

        if avg_sent >= 0.2:
            strengths.extend(phrases)
        elif avg_sent <= -0.1:
            weaknesses.extend(phrases)
        else:
            strengths.extend(phrases[:2])
            weaknesses.extend(phrases[2:4])

    def clean_list(lst):
        out = []
        for x in lst:
            if not x or len(x.strip()) < 2:
                continue
            xx = x.replace("  ", " ").strip()
            if xx not in out:
                out.append(xx)
            if len(out) >= top_k:
                break
        return out[:top_k]

    return clean_list(strengths), clean_list(weaknesses)

# ---------- fallback extractor (pure Python) ----------
def extract_unique_themes_fallback(reviews: List[Dict[str, Any]], top_k=3):
    texts = [normalize_text(r.get("ReviewText","") or "") for r in reviews if r.get("ReviewText")]
    texts = [t for t in texts if len(t) > 5]
    if not texts:
        return [], []

    docs = []
    for t in texts:
        tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
        docs.append(tokens)

    df = Counter()
    tf = []
    for tokens in docs:
        tf_doc = Counter(tokens)
        tf.append(tf_doc)
        for w in set(tokens):
            df[w] += 1

    total_docs = max(1, len(docs))
    tfidf = {}
    for doc in tf:
        for w, f in doc.items():
            idf = math.log((total_docs + 1) / (df[w] + 1)) + 1
            tfidf[w] = tfidf.get(w, 0.0) + f * idf

    sorted_words = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in sorted_words[:20]]

    strengths = []
    weaknesses = []

    avg_sent = 0.0
    sent_count = 0
    for r in reviews:
        if "SentimentScore" in r:
            avg_sent += safe_float(r.get("SentimentScore", 0.0))
            sent_count += 1
    avg_sent = (avg_sent / sent_count) if sent_count else 0.0

    for w in top_words:
        if len(strengths) >= top_k and len(weaknesses) >= top_k:
            break
        if avg_sent >= 0.15:
            if len(strengths) < top_k:
                strengths.append(w)
        elif avg_sent <= -0.1:
            if len(weaknesses) < top_k:
                weaknesses.append(w)
        else:
            if len(strengths) < top_k:
                strengths.append(w)
            if len(weaknesses) < top_k and len(top_words) > 5:
                weaknesses.append(top_words[-(len(weaknesses)+1)])

    def clean_list(lst):
        out=[]
        for x in lst:
            if x and x.strip() not in out:
                out.append(x.strip())
            if len(out)>=top_k:
                break
        return out[:top_k]

    return clean_list(strengths), clean_list(weaknesses)

# ---------------------------
# dispatcher for extractor
# ---------------------------
def extract_unique_themes(reviews: List[Dict[str, Any]], top_k=3):
    if SKLEARN_AVAILABLE:
        try:
            return extract_unique_themes_sklearn(reviews, top_k=top_k)
        except Exception as e:
            LOG.warning("sklearn extractor runtime failure, falling back: %s", str(e))
            return extract_unique_themes_fallback(reviews, top_k=top_k)
    else:
        return extract_unique_themes_fallback(reviews, top_k=top_k)

# ---------- build_structured_summary (uses extractor) ----------
def build_structured_summary(product_id: str, reviews: List[Dict[str, Any]], agg_sent: float) -> str:
    strengths, weaknesses = extract_unique_themes(reviews, top_k=3)

    # fallback if clustering yields nothing
    if not strengths and not weaknesses:
        pos = [ (r.get("ReviewText","") or "").strip() for r in reviews if safe_float(r.get("SentimentScore")) > 0.4 ]
        neg = [ (r.get("ReviewText","") or "").strip() for r in reviews if safe_float(r.get("SentimentScore")) < -0.2 ]
        if pos:
            s = " ".join(pos[0].split()[:8])
            strengths = [s.rstrip('.,')]
        if neg:
            s = " ".join(neg[0].split()[:8])
            weaknesses = [s.rstrip('.,')]

    def tidy(lst):
        out = []
        for it in lst:
            s = re.sub(r'[_\-\s]{2,}', ' ', it).strip(' .,-')
            s = re.sub(r'\s+', ' ', s)
            if s and s.lower() not in (o.lower() for o in out):
                out.append(s)
        return out

    strengths = tidy(strengths)[:3]
    weaknesses = tidy(weaknesses)[:3]

    strengths_txt = ", ".join(strengths) if strengths else "None notable"
    weaknesses_txt = ", ".join(weaknesses) if weaknesses else "None notable"
    if agg_sent > 0.25:
        overall = f"Overall positive user sentiment with strong points such as {strengths[0] if strengths else 'good performance'}."
    elif agg_sent < -0.15:
        overall = f"Overall negative sentiment mainly due to issues like {weaknesses[0] if weaknesses else 'user complaints'}."
    else:
        overall = f"Mixed user opinions with highlights around {strengths[0] if strengths else 'performance'} and concerns around {weaknesses[0] if weaknesses else 'some issues'}."

    final = f"Strengths: {strengths_txt}; Weaknesses: {weaknesses_txt}; Overall: {overall}"
    final = re.sub(r'\s+', ' ', final)
    if len(final) > MAX_SUMMARY_CHARS:
        final = final[:MAX_SUMMARY_CHARS].rsplit('.',1)[0] + '.'
    return final.strip()

# ---------- SageMaker prompt builder (now defined BEFORE process_product) ----------
def build_prompt_for_product(pid: str, reviews: List[Dict[str, Any]]) -> str:
    # Compact prompt: include only a few representative reviews to keep payload small
    samples = []
    for r in reviews[:6]:
        txt = (r.get("ReviewText","") or "").replace("\n", " ").strip()
        if len(txt) > 400:
            txt = txt[:400] + "..."
        samples.append(txt)
    samples_text = "\n\n".join(samples)
    prompt = (
        f"Summarize product {pid} from the following user reviews. "
        "Output a short structured summary with strengths, weaknesses, and overall sentiment. "
        "Keep it concise. Use this format: Strengths: ...; Weaknesses: ...; Overall: ...\n\n"
        f"Reviews:\n{samples_text}"
    )
    return prompt

# ---------- SageMaker helpers ----------
def parse_sagemaker_response(raw):
    if raw is None:
        return ""
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
    except Exception:
        raw = str(raw)
    try:
        obj = json.loads(raw)
        for k in ("summary_text","generated_text","summary","output_text","text","result"):
            if k in obj and isinstance(obj[k], str):
                return obj[k].strip()
        for k in ("predictions","outputs","results","data"):
            arr = obj.get(k)
            if isinstance(arr, list) and arr:
                v = arr[0]
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, dict):
                    for inner in v.values():
                        if isinstance(inner, str):
                            return inner.strip()
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, list) and obj and isinstance(obj[0], str):
            return obj[0].strip()
        return str(obj)
    except Exception:
        return raw if isinstance(raw, str) else str(raw)

def sage_clean_and_normalize(model_output: str, agg_sent: float) -> str:
    t = model_output or ""
    t = re.sub(r'input[_\s]?reviews[:\s]*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\"[^\"]{1,300}\"', '', t)
    t = re.sub(r'(?i)(back to the page you came from|use this article to|click here|learn more|visit our|please see).*', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if re.search(r'\bstrengths\b', t, flags=re.IGNORECASE) or re.search(r'\bweaknesses\b', t, flags=re.IGNORECASE) or len(t) < 500:
        t = re.sub(r'\s*;\s*', '; ', t)
        t = t.strip(' .')
        if not t.endswith('.'):
            t += '.'
        if len(t) > MAX_SUMMARY_CHARS:
            t = t[:MAX_SUMMARY_CHARS].rsplit('.',1)[0] + '.'
        return t
    return ""

def invoke_sagemaker_quick(prompt: str, agg_sent: float) -> str:
    if not USE_SAGEMAKER or sagemaker_runtime is None:
        return ""
    content_variants = [
        ("application/json", lambda p: json.dumps({"inputs": p}).encode("utf-8")),
        ("text/plain; charset=utf-8", lambda p: p.encode("utf-8")),
    ]
    for ctype, builder in content_variants:
        for attempt in range(1):
            try:
                resp = sagemaker_runtime.invoke_endpoint(
                    EndpointName=SAGEMAKER_ENDPOINT,
                    ContentType=ctype,
                    Accept="application/json",
                    Body=builder(prompt)
                )
                raw = resp["Body"].read()
                parsed = parse_sagemaker_response(raw)
                if parsed:
                    cleaned = sage_clean_and_normalize(parsed, agg_sent)
                    if cleaned:
                        return cleaned
            except Exception as e:
                LOG.debug(f"SageMaker attempt failed: {e}")
            time.sleep(0.2)
    return ""

# ---------------------------
# Bedrock helpers (INSIGHTS)
# ---------------------------
def _shorten_text(s: str, n_chars=800):
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    if len(s) <= n_chars:
        return s
    return s[:n_chars].rsplit(" ", 1)[0] + "..."

def build_bedrock_insight_prompt(run_meta: dict, summary_items: List[dict], top_n_products: int = 5) -> str:
    """
    Build a compact prompt asking Bedrock for a 1-paragraph insight.
    """
    attempted = run_meta.get("attempted", 0)
    written = run_meta.get("written", 0)

    # Aggregate top strengths / weaknesses from summaries
    strengths_counter = Counter()
    weaknesses_counter = Counter()

    for r in summary_items:
        txt = (r.get("Summary") or "")
        m_s = re.search(r"Strengths:\s*([^;]+)", txt, flags=re.IGNORECASE)
        m_w = re.search(r"Weaknesses:\s*([^;]+)", txt, flags=re.IGNORECASE)
        if m_s:
            for p in re.split(r',\s*', m_s.group(1).strip()):
                p = normalize_text(p)[:60]
                if p:
                    strengths_counter[p] += 1
        if m_w:
            for p in re.split(r',\s*', m_w.group(1).strip()):
                p = normalize_text(p)[:60]
                if p:
                    weaknesses_counter[p] += 1

    top_strengths = [k for k,_ in strengths_counter.most_common(4)]
    top_weaknesses = [k for k,_ in weaknesses_counter.most_common(4)]

    # top products by sentiment
    ranked = sorted(summary_items, key=lambda x: safe_float(x.get("AggregateSentiment", 0.0)), reverse=True)
    top = ranked[:top_n_products]
    top_list = []
    for p in top:
        pid = p.get("ProductID", "unknown")
        s_preview = (p.get("Summary") or "")[:120].replace("\n", " ")
        top_list.append(f"{pid}: {s_preview}")

    prompt = (
        "You are a concise product insights assistant. "
        "Given a summarization run, produce a single short paragraph (1-3 sentences) "
        "that highlights (a) the top praised themes across products and approximate share, "
        "(b) the most common complaints, and (c) one product with the strongest improvement or the biggest issue. "
        "Be specific but compact. Use plain AS-IS phrases from the summaries when possible.\n\n"
        f"Run stats: attempted={attempted}, written={written}.\n\n"
        f"Top strengths: {', '.join(top_strengths) if top_strengths else 'none'}.\n"
        f"Top complaints: {', '.join(top_weaknesses) if top_weaknesses else 'none'}.\n"
        f"Top products (brief): { '; '.join(top_list) if top_list else 'none' }.\n\n"
        "Return just the paragraph (no bullets)."
    )
    return _shorten_text(prompt, n_chars=1500)

def call_bedrock_model(prompt: str, max_output_tokens: int = BEDROCK_MAX_OUTPUT_TOKENS, temperature: float = BEDROCK_TEMPERATURE) -> str:
    """
    Invoke Bedrock runtime. Returns text or empty string on failure.
    """
    if not USE_BEDROCK or bedrock_runtime is None:
        LOG.debug("Bedrock not enabled or client missing.")
        return ""

    try:
        payload = {
            "input": prompt,
            "max_tokens": max_output_tokens,
            "temperature": float(temperature)
        }
        resp = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload).encode("utf-8")
        )
        # resp.body is a streaming object in some SDKs
        body_bytes = resp.get("body").read() if isinstance(resp.get("body"), object) else resp.get("body")
        if not body_bytes:
            return ""
        body_text = body_bytes.decode("utf-8", errors="ignore") if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)
        try:
            parsed = json.loads(body_text)
            # common keys
            for key in ("output", "outputs", "results", "content", "body", "text"):
                if isinstance(parsed, dict) and key in parsed:
                    v = parsed[key]
                    if isinstance(v, str):
                        return v.strip()
                    if isinstance(v, list) and v and isinstance(v[0], str):
                        return v[0].strip()
                    if isinstance(v, dict):
                        for inner in v.values():
                            if isinstance(inner, str):
                                return inner.strip()
            if isinstance(parsed, str):
                return parsed.strip()
            return str(parsed)[:1000]
        except Exception:
            return body_text.strip()[:1000]
    except Exception as e:
        LOG.warning("Bedrock invoke failed: %s", str(e))
        return ""

def build_local_insight(run_meta: dict, summary_items: List[dict]) -> str:
    """
    Deterministic, cheap insight generator (fallback when Bedrock is disabled or returns nothing).
    Produces 1-2 sentences summarizing top themes and top complaints with rough percentages.
    """
    total_products = max(1, len(summary_items))
    strengths_counter = Counter()
    weaknesses_counter = Counter()
    improvements = []  # placeholder for product improvement detection (if we had previous data)

    for r in summary_items:
        txt = (r.get("Summary") or "")
        m_s = re.search(r"Strengths:\s*([^;]+)", txt, flags=re.IGNORECASE)
        m_w = re.search(r"Weaknesses:\s*([^;]+)", txt, flags=re.IGNORECASE)
        if m_s:
            for p in re.split(r',\s*', m_s.group(1).strip()):
                p = normalize_text(p)[:60]
                if p:
                    strengths_counter[p] += 1
        if m_w:
            for p in re.split(r',\s*', m_w.group(1).strip()):
                p = normalize_text(p)[:60]
                if p:
                    weaknesses_counter[p] += 1

    def top_with_pct(counter: Counter, top_n=3):
        out = []
        for k, v in counter.most_common(top_n):
            pct = round(100.0 * v / total_products)
            out.append(f"'{k}' ({pct}%)")
        return out

    top_strengths = top_with_pct(strengths_counter, 3)
    top_weaknesses = top_with_pct(weaknesses_counter, 3)

    top_prod = sorted(summary_items, key=lambda x: safe_float(x.get("AggregateSentiment", 0.0)), reverse=True)
    best = top_prod[0].get("ProductID") if top_prod else "unknown"
    worst = top_prod[-1].get("ProductID") if top_prod else "unknown"

    parts = []
    if top_strengths:
        parts.append(f"Across products users praised {', '.join(top_strengths)}.")
    if top_weaknesses:
        parts.append(f"Common complaints include {', '.join(top_weaknesses)}.")
    parts.append(f"Top product by sentiment: {best}. Biggest issues seen on: {worst}.")

    insight = " ".join(parts)
    insight = re.sub(r'\s+', ' ', insight).strip()
    return insight

# ---------- persistence ----------
def save_summary_to_s3(product_id: str, summary: str, agg_sent: float, attempts: int, method: str):
    ts = _now_ts()
    key = f"summaries/{product_id}-{ts}.json"
    obj = {
        "ProductID": product_id,
        "Summary": summary,
        "AggregateSentiment": agg_sent,
        "Timestamp": ts,
        "SummaryAttempts": attempts,
        "Method": method
    }
    try:
        s3.put_object(Bucket=OUTPUT_BUCKET, Key=key, Body=json.dumps(obj).encode("utf-8"))
    except Exception as e:
        LOG.warning(f"S3 put failed for {key}: {e}")
    return key

def write_summary_to_table(product_id: str, summary: str, agg_sent: float, attempts: int, method: str) -> bool:
    ts = _now_ts()
    item = {
        "ProductID": product_id,
        "GeneratedAt": Decimal(str(ts)),
        "Summary": summary,
        "AggregateSentiment": Decimal(str(agg_sent)),
        "SummaryTimestamp": Decimal(str(ts)),
        "UUID": str(uuid.uuid4()),
        "SummaryAttempts": attempts,
        "Method": method
    }
    try:
        summary_table.put_item(Item=item)
        return True
    except Exception as e:
        LOG.error(f"DynamoDB put_item failed for product {product_id}: {e}")
        try:
            fail_key = f"failures/{product_id}-{ts}.json"
            s3.put_object(Bucket=OUTPUT_BUCKET, Key=fail_key, Body=json.dumps({"error": str(e), "item": item}).encode("utf-8"))
        except Exception:
            pass
        return False

# ---------- product worker ----------
def process_product(pid: str, reviews: List[Dict[str, Any]], start_time: float, exec_budget: float):
    try:
        if time.time() - start_time > exec_budget:
            return (pid, "", "timeout", 0.0)
        selected = sorted(reviews, key=lambda r: r.get("CreatedAt") or 0, reverse=True)[:MAX_REVIEWS_PER_PRODUCT]
        vals = [safe_float(r.get("SentimentScore")) for r in selected if r.get("SentimentScore") is not None]
        agg_sent = (sum(vals) / len(vals)) if vals else 0.0

        if SKIP_EXISTING_SUMMARIES:
            try:
                from boto3.dynamodb.conditions import Key
                resp = summary_table.query(KeyConditionExpression=Key('ProductID').eq(pid), Limit=1)
                if resp.get('Count', 0) > 0:
                    LOG.debug(f"Skipping {pid} (exists)")
                    return (pid, "", "skipped-exists", agg_sent)
            except Exception:
                LOG.debug("Summary table query failed or schema mismatch; continuing")

        # optional SageMaker path
        if USE_SAGEMAKER:
            prompt = build_prompt_for_product(pid, selected)
            try:
                model_summary = invoke_sagemaker_quick(prompt, agg_sent)
            except Exception as e:
                LOG.warning("SageMaker quick failed for %s: %s", pid, str(e))
                model_summary = ""
            if model_summary:
                method = "model"
                save_summary_to_s3(pid, model_summary, agg_sent, attempts=1, method=method)
                ok = write_summary_to_table(pid, model_summary, agg_sent, attempts=1, method=method)
                return (pid, model_summary if ok else "", method, agg_sent)

        # fallback deterministic summary using our extractor
        constructed = build_structured_summary(pid, selected, agg_sent)
        method = "constructed"
        save_summary_to_s3(pid, constructed, agg_sent, attempts=0, method=method)
        ok = write_summary_to_table(pid, constructed, agg_sent, attempts=0, method=method)
        return (pid, constructed if ok else "", method, agg_sent)

    except Exception as e:
        LOG.error(f"Error processing product {pid}: {e}\n{traceback.format_exc()}")
        return (pid, "", "error", 0.0)

# ---------- aggregate / upload helpers ----------
def upload_aggregate_and_get_link(aggregate_obj: dict, filename_prefix: str = "summaries-") -> Tuple[str,str]:
    ts = _now_ts()
    key = f"{AGGREGATE_S3_PREFIX}{filename_prefix}{ts}.json"
    try:
        body = json.dumps(aggregate_obj, indent=2).encode("utf-8")
    except Exception:
        body = str(aggregate_obj).encode("utf-8")
    try:
        s3.put_object(Bucket=OUTPUT_BUCKET, Key=key, Body=body)
        url = s3.generate_presigned_url('get_object', Params={'Bucket': OUTPUT_BUCKET, 'Key': key}, ExpiresIn=PRESIGN_EXPIRY_SECONDS)
        return key, url
    except Exception as e:
        LOG.error("Failed to upload aggregate to S3: %s", e)
        return "", ""

def build_run_report(run_meta: dict, summary_items: List[dict]) -> dict:
    ranked = sorted(summary_items, key=lambda x: safe_float(x.get("AggregateSentiment",0.0)), reverse=True)
    top = ranked[:5]
    bottom = ranked[-5:][::-1] if ranked else []
    counts = {"attempted": run_meta.get("attempted",0), "written": run_meta.get("written",0), "skipped": run_meta.get("skipped",0)}
    report = {
        "run_summary": counts,
        "duration_seconds": run_meta.get("duration_s", 0.0),
        "top_products": [{"ProductID": p["ProductID"], "agg_sent": safe_float(p.get("AggregateSentiment",0.0)), "summary_preview": (p.get("Summary") or "")[:200]} for p in top],
        "bottom_products": [{"ProductID": p["ProductID"], "agg_sent": safe_float(p.get("AggregateSentiment",0.0)), "summary_preview": (p.get("Summary") or "")[:200]} for p in bottom],
        "note": "Full summaries are provided in the second email (or via S3 link if large)."
    }
    return report

# ---------- main handler ----------
def lambda_handler(event, context):
    LOG.info("Summarizer invoked (aggregated notifications).")
    start_time = time.time()
    try:
        remaining = None
        try:
            remaining = context.get_remaining_time_in_millis() / 1000.0
        except Exception:
            remaining = None
        exec_budget = min(MAX_EXEC_SECONDS, (remaining - 5) if remaining and remaining > 10 else MAX_EXEC_SECONDS)

        all_reviews = scan_reviews(limit=MAX_SCAN_ITEMS)
        if not all_reviews:
            LOG.info("No reviews found; publishing empty run report.")
            publish_sns("Summarizer run report", {"event":"summarizer_run_report","payload":{"attempted":0,"written":0,"skipped":0,"duration_s":0}})
            return {"statusCode":200,"message":"No reviews"}

        product_groups = group_by_product(all_reviews)
        if not product_groups:
            LOG.info("No products grouped; publishing empty run report.")
            publish_sns("Summarizer run report", {"event":"summarizer_run_report","payload":{"attempted":0,"written":0,"skipped":0,"duration_s":0}})
            return {"statusCode":200,"message":"No products"}

        product_items = list(product_groups.items())[:MAX_TOTAL_PRODUCTS]
        LOG.info(f"Processing {len(product_items)} products")

        results = []
        skipped = 0
        written = 0

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as ex:
            futures = {ex.submit(process_product, pid, revs, start_time, exec_budget): pid for pid, revs in product_items}
            try:
                for fut in as_completed(futures, timeout=max(1.0, exec_budget)):
                    pid = futures.get(fut)
                    try:
                        pid, summary_text, method, agg_sent = fut.result()
                        if method and method.startswith("skipped"):
                            skipped += 1
                        if summary_text:
                            written += 1
                            results.append({"ProductID": pid, "Summary": summary_text, "Method": method, "AggregateSentiment": agg_sent})
                    except Exception as e:
                        LOG.error(f"Task exception for {pid}: {e}")
            except Exception as e:
                LOG.warning(f"Collection loop ended/timeout: {e}")

        duration = time.time() - start_time
        attempted = len(product_items)
        run_meta = {"attempted": attempted, "written": written, "skipped": skipped, "duration_s": duration}
        LOG.info("Run finished: %s", json.dumps(run_meta))

        # Email 1: compact run report
        run_report_obj = build_run_report(run_meta, results)
        publish_sns("Summarizer run report", {"event":"summarizer_run_report", "payload": run_report_obj})

        # Prepare aggregate payload
        aggregate_payload = {"generated_at": _now_ts(), "run_meta": run_meta, "summaries": results}
        try:
            agg_bytes = len(json.dumps(aggregate_payload).encode("utf-8"))
        except Exception:
            agg_bytes = INLINE_SUMMARIES_MAX_BYTES + 1

        presigned_url = ""

        # --- NEW: Generate Bedrock insight and publish as separate SNS email (optional),
        # with deterministic fallback when Bedrock disabled or empty ---
        try:
            insight = ""
            if USE_BEDROCK:
                prompt = build_bedrock_insight_prompt(run_meta, results, top_n_products=5)
                LOG.debug("Calling Bedrock with prompt (len=%d)", len(prompt))
                insight = call_bedrock_model(prompt)
                LOG.debug("Bedrock returned insight len=%d", len(insight) if insight else 0)
            # If Bedrock not used or returned empty - build local deterministic insight
            if not insight:
                insight = build_local_insight(run_meta, results)
                LOG.debug("Using local insight fallback (len=%d)", len(insight) if insight else 0)

            if insight:
                # Optionally save insight to S3 for audit
                try:
                    insight_key = f"insights/insight-{_now_ts()}.txt"
                    s3.put_object(Bucket=OUTPUT_BUCKET, Key=insight_key, Body=insight.encode("utf-8"))
                except Exception as e:
                    LOG.debug("Failed saving insight to S3: %s", str(e))

                # Publish insight as readable message (string) to SNS
                publish_sns("Summarizer Insights", insight)
                LOG.info("Published insight via SNS (len=%d)", len(insight))
            else:
                LOG.info("No insight generated (empty).")
        except Exception as e:
            LOG.warning("Failed to generate/publish insight: %s", str(e))

        # Email 2: full summaries inline or S3 link
        if agg_bytes <= INLINE_SUMMARIES_MAX_BYTES:
            publish_sns("Summaries - full payload", {"event":"summaries_full_inline", "payload": aggregate_payload})
            LOG.info("Published full summaries inline (size %d bytes).", agg_bytes)
        else:
            s3_key, presigned_url = upload_aggregate_and_get_link(aggregate_payload, filename_prefix="summaries-")
            if s3_key:
                publish_sns("Summaries - full payload (S3 link)", {"event":"summaries_full_s3", "payload": {"s3_key": s3_key, "presigned_url": presigned_url, "generated_at": aggregate_payload["generated_at"], "run_meta": run_meta}})
                LOG.info("Uploaded aggregate to s3://%s/%s (size %d bytes) and published link.", OUTPUT_BUCKET, s3_key, agg_bytes)
            else:
                truncated = {"generated_at": aggregate_payload["generated_at"], "run_meta": run_meta, "summaries_preview": [{"ProductID": r["ProductID"], "AggregateSentiment": r.get("AggregateSentiment",0.0), "Summary": (r["Summary"] or "")[:400]} for r in results]}
                publish_sns("Summaries - partial payload", {"event":"summaries_partial", "payload": truncated})
                LOG.warning("Aggregate upload failed; published truncated payload.")

        return {"statusCode":200, "message": f"Processed {attempted} products, wrote {written} summaries.", "written": written, "presigned_url": presigned_url}
    except Exception as e:
        tb = traceback.format_exc()
        LOG.exception("Fatal error in summarizer: %s", e)
        publish_sns("Summarizer fatal error", {"event":"pipeline_error", "payload": {"error": str(e), "trace": tb}})
        raise
