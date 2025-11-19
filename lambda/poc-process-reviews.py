# lambda_function.py (INGEST) - with SNS integration
# Ingest Lambda -> Comprehend -> DynamoDB (safe Decimal conversion) -> SNS notifications
import os
import json
import csv
import io
import logging
import time
import hashlib
from decimal import Decimal
from typing import List, Dict, Any, Optional
import traceback

import boto3
from botocore.exceptions import ClientError, BotoCoreError

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
LOG = logging.getLogger("poc_process_reviews_no_describe")

AWS_REGION = os.environ.get("AWS_REGION")
boto3_kwargs = {"region_name": AWS_REGION} if AWS_REGION else {}

DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")
MAX_RECORDS_PER_FILE = int(os.environ.get("MAX_RECORDS_PER_FILE", "0"))
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")  # NEW

if not DYNAMO_TABLE or not OUTPUT_BUCKET:
    LOG.error("Set DYNAMO_TABLE and OUTPUT_BUCKET environment variables")
    # allow startup, but handler will fail quickly

# boto clients
s3 = boto3.client("s3", **boto3_kwargs)
comprehend = boto3.client("comprehend", **boto3_kwargs)
dynamodb = boto3.resource("dynamodb", **boto3_kwargs)
table = dynamodb.Table(DYNAMO_TABLE) if DYNAMO_TABLE else None
sns = boto3.client("sns", **boto3_kwargs) if SNS_TOPIC_ARN else None

def publish_event(event_type: str, subject: str, payload: Dict[str, Any]):
    if not sns:
        LOG.debug("SNS not configured; skipping publish for %s", event_type)
        return
    try:
        msg = {"event": event_type, "payload": payload}
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=json.dumps(msg))
    except Exception as e:
        LOG.warning("SNS publish failed for %s: %s", event_type, str(e))

def deterministic_id(product: str, text: str) -> str:
    base = f"{product}||{text}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

def safe_get_s3_object(bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()

def detect_all(text: str) -> Dict[str, Any]:
    # resilient: fallback to defaults on errors
    try:
        lang_resp = comprehend.detect_dominant_language(Text=text)
        langs = lang_resp.get("Languages", [])
        lang = langs[0].get("LanguageCode", "en") if langs else "en"
    except Exception:
        lang = "en"
    try:
        sentiment_resp = comprehend.detect_sentiment(Text=text, LanguageCode=lang)
    except Exception:
        sentiment_resp = {"Sentiment": None, "SentimentScore": {}}
    try:
        phrases_resp = comprehend.detect_key_phrases(Text=text, LanguageCode=lang)
    except Exception:
        phrases_resp = {"KeyPhrases": []}
    try:
        entities_resp = comprehend.detect_entities(Text=text, LanguageCode=lang)
    except Exception:
        entities_resp = {"Entities": []}

    scores = sentiment_resp.get("SentimentScore", {})
    pos = float(scores.get("Positive", 0.0))
    neg = float(scores.get("Negative", 0.0))
    neutral = float(scores.get("Neutral", 0.0))
    mixed = float(scores.get("Mixed", 0.0))

    keyphrases = []
    for kp in phrases_resp.get("KeyPhrases", []):
        keyphrases.append({"Text": kp.get("Text"), "Score": float(kp.get("Score", 0.0))})

    entities = []
    for en in entities_resp.get("Entities", []):
        entities.append({"Text": en.get("Text"), "Type": en.get("Type"), "Score": float(en.get("Score", 0.0))})

    return {
        "Sentiment": sentiment_resp.get("Sentiment"),
        "PositiveScore": pos,
        "NegativeScore": neg,
        "NeutralScore": neutral,
        "MixedScore": mixed,
        "KeyPhrases": keyphrases,
        "Entities": entities,
        "Language": lang,
        "RawSentiment": sentiment_resp
    }

def to_decimal_safe(v):
    try:
        if v is None:
            return Decimal("0")
        if isinstance(v, Decimal):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return Decimal(str(v))
    except Exception:
        return Decimal("0")

def convert_structure_numbers(obj):
    if isinstance(obj, list):
        return [convert_structure_numbers(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_structure_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (int, float)):
        return Decimal(str(obj))
    return obj

def parse_input_bytes(key: str, body_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        txt = body_bytes.decode("utf-8")
    except Exception:
        txt = body_bytes.decode("latin-1", errors="ignore")
    items: List[Dict[str, Any]] = []
    lower = key.lower()
    if lower.endswith(".csv"):
        f = io.StringIO(txt)
        reader = csv.DictReader(f)
        for r in reader:
            items.append({
                "ReviewText": r.get("ReviewText") or r.get("text") or r.get("review") or "",
                "ProductID": r.get("ProductID") or r.get("product") or r.get("product_id") or "unknown",
                "ReviewDate": r.get("ReviewDate") or r.get("date") or ""
            })
    else:
        stripped = txt.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                for el in parsed:
                    if isinstance(el, dict):
                        items.append({
                            "ReviewText": el.get("ReviewText") or el.get("text") or el.get("review") or "",
                            "ProductID": el.get("ProductID") or el.get("product") or "unknown",
                            "ReviewDate": el.get("ReviewDate") or el.get("date") or "",
                            "RawOriginal": el
                        })
                    else:
                        items.append({"ReviewText": str(el), "ProductID": "unknown", "ReviewDate": ""})
            elif isinstance(parsed, dict):
                items.append({
                    "ReviewText": parsed.get("ReviewText") or parsed.get("text") or parsed.get("review") or "",
                    "ProductID": parsed.get("ProductID") or parsed.get("product") or "unknown",
                    "ReviewDate": parsed.get("ReviewDate") or parsed.get("date") or "",
                    "RawOriginal": parsed
                })
            else:
                items.append({"ReviewText": str(parsed), "ProductID": "unknown", "ReviewDate": ""})
        except Exception:
            for ln in txt.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    parsed = json.loads(ln)
                    if isinstance(parsed, dict):
                        items.append({
                            "ReviewText": parsed.get("ReviewText") or parsed.get("text") or parsed.get("review") or "",
                            "ProductID": parsed.get("ProductID") or parsed.get("product") or "unknown",
                            "ReviewDate": parsed.get("ReviewDate") or parsed.get("date") or "",
                            "RawOriginal": parsed
                        })
                    else:
                        items.append({"ReviewText": str(parsed), "ProductID": "unknown", "ReviewDate": ""})
                except Exception:
                    items.append({"ReviewText": ln, "ProductID": "unknown", "ReviewDate": ""})
    return items

def put_item_to_dynamo(item: Dict[str, Any]):
    if table is None:
        raise Exception("DYNAMO_TABLE not configured")
    pk = item.get("ReviewID")
    if not pk:
        raise Exception("Missing ReviewID; cannot write to Dynamo")
    pos = to_decimal_safe(item.get("PositiveScore", 0.0))
    neg = to_decimal_safe(item.get("NegativeScore", 0.0))
    neutral = to_decimal_safe(item.get("NeutralScore", 0.0))
    mixed = to_decimal_safe(item.get("MixedScore", 0.0))
    dominant = pos - neg
    keyphrases = convert_structure_numbers(item.get("KeyPhrases", []))
    entities = convert_structure_numbers(item.get("Entities", []))
    raw_json = ""
    try:
        raw_json = json.dumps(item.get("Raw", {}))
    except Exception:
        raw_json = json.dumps({"raw": str(item.get("Raw", ""))})
    put_item = {
        "ReviewID": str(pk),
        "ProductID": str(item.get("ProductID", "unknown")),
        "ReviewText": item.get("ReviewText", "") or "",
        "ReviewDate": item.get("ReviewDate", "") or "",
        "SentimentScore": dominant,
        "PositiveScore": pos,
        "NegativeScore": neg,
        "NeutralScore": neutral,
        "MixedScore": mixed,
        "KeyPhrases": keyphrases,
        "Entities": entities,
        "Raw": raw_json,
        "Summary": item.get("Summary", "") if "Summary" in item else ""
    }
    return table.put_item(Item=put_item)

def process_s3_object(bucket: str, key: str, body_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    ts = int(time.time())
    stats = {"processed": 0, "failed": 0, "skipped": 0, "trace_key": None}
    if body_bytes is None:
        body = safe_get_s3_object(bucket, key)
    else:
        body = body_bytes
    items = parse_input_bytes(key, body)
    LOG.info("Parsed %d items from %s/%s", len(items), bucket, key)
    processed_trace = []
    for idx, raw in enumerate(items):
        if MAX_RECORDS_PER_FILE and idx >= MAX_RECORDS_PER_FILE:
            LOG.info("Reached MAX_RECORDS_PER_FILE=%d", MAX_RECORDS_PER_FILE)
            break
        text = (raw.get("ReviewText") or "").strip()
        if not text:
            stats["skipped"] += 1
            continue
        product = raw.get("ProductID") or "unknown"
        review_date = raw.get("ReviewDate", "")
        review_id = raw.get("ReviewID") or deterministic_id(product, text)
        try:
            analysis = detect_all(text)
        except Exception as e:
            LOG.exception("Comprehend failed for review %s: %s", review_id, e)
            stats["failed"] += 1
            continue
        item = {
            "ReviewID": review_id,
            "ProductID": product,
            "ReviewText": text,
            "ReviewDate": review_date,
            "PositiveScore": analysis.get("PositiveScore", 0.0),
            "NegativeScore": analysis.get("NegativeScore", 0.0),
            "NeutralScore": analysis.get("NeutralScore", 0.0),
            "MixedScore": analysis.get("MixedScore", 0.0),
            "KeyPhrases": analysis.get("KeyPhrases", []),
            "Entities": analysis.get("Entities", []),
            "Raw": {"Analysis": analysis, "Original": raw.get("RawOriginal") or raw}
        }
        try:
            put_item_to_dynamo(item)
            stats["processed"] += 1
            status = "OK"
        except Exception as e:
            LOG.exception("Dynamo put failed for %s: %s", review_id, e)
            stats["failed"] += 1
            status = f"FAILED: {str(e)[:300]}"
        processed_trace.append({**item, "DynamoPutStatus": status})
    # write NDJSON trace
    base = key.split("/")[-1]
    trace_key = f"processed/{base}-{ts}.ndjson"
    try:
        ndjson = "\n".join([json.dumps(x) for x in processed_trace]) + ("\n" if processed_trace else "")
        s3.put_object(Bucket=OUTPUT_BUCKET, Key=trace_key, Body=ndjson.encode("utf-8"))
        stats["trace_key"] = trace_key
        LOG.info("Wrote trace to s3://%s/%s (records=%d)", OUTPUT_BUCKET, trace_key, len(processed_trace))
    except Exception as e:
        LOG.exception("Failed to write trace: %s", e)
    return stats

def lambda_handler(event, context):
    LOG.info("Handler invoked (ingest). Event: %s", json.dumps(event)[:1000])
    records = event.get("Records", [])
    # handle SQS-wrapped S3-event bodies if present
    if records and isinstance(records[0].get("body"), str):
        # SQS -> body contains S3 event JSON
        parsed_records = []
        for rec in records:
            try:
                body = json.loads(rec.get("body"))
                if "Records" in body:
                    parsed_records.extend(body["Records"])
            except Exception:
                LOG.debug("SQS record body is not JSON S3 event; using original record")
                parsed_records.append(rec)
        if parsed_records:
            records = parsed_records

    if not records:
        LOG.warning("No Records in event")
        return {"statusCode": 400, "message": "No Records in event"}
    results = []
    total_p = total_f = total_s = 0
    try:
        for rec in records:
            try:
                s3info = rec.get("s3", {})
                bucket = s3info.get("bucket", {}).get("name")
                key = s3info.get("object", {}).get("key")
                if not bucket or not key:
                    LOG.warning("Record missing bucket/key")
                    continue
                LOG.info("Processing s3://%s/%s", bucket, key)
                stats = process_s3_object(bucket, key)
                results.append({"bucket": bucket, "key": key, "stats": stats})
                total_p += stats.get("processed", 0)
                total_f += stats.get("failed", 0)
                total_s += stats.get("skipped", 0)
                # Publish SNS about the file processed
                publish_event(
                    "file_processed",
                    f"Processed {key}",
                    {"bucket": bucket, "key": key, "stats": stats, "trace_key": stats.get("trace_key")}
                )
            except Exception as e:
                LOG.exception("Top-level error processing record: %s", e)
                results.append({"error": str(e)})
        summary = {"processed": total_p, "failed": total_f, "skipped": total_s, "details": results}
        LOG.info("INGEST SUMMARY: %s", json.dumps(summary))
        return {"statusCode": 200, "summary": summary}
    except Exception as e:
        tb = traceback.format_exc()
        LOG.exception("Fatal error in ingest lambda: %s", e)
        # publish an SNS failure event with stacktrace
        publish_event("pipeline_error", "Ingest Lambda fatal error", {"error": str(e), "trace": tb, "event_sample": (json.dumps(event)[:4000])})
        # re-raise so Lambda records the failure in CloudWatch
        raise
