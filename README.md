# 📘 Sentiment Insights – AWS Serverless Review Processing & AI Summarization System

A Full End-to-End Serverless Architecture using S3, Lambda, DynamoDB, Comprehend, SageMaker, Bedrock, Glue, Athena, Quicksight & EventBridge

## 🧭 Table of Contents

- 📌 Project Overview
- ⚙️ Architecture Diagram
- 🧱 Core AWS Services Used
- 🚀 End-to-End Workflow
- 🛠️ Step-by-Step Implementation Guide
  - Create S3 Buckets
  - Build Process-Reviews Lambda
  - Configure DynamoDB Tables
  - Build Summarizer Lambda
  - Create SageMaker Endpoint (DistilBART)
  - Enable Bedrock Titan for Insights
  - Integrate SNS Email Notifications
  - Set Up EventBridge Scheduler
  - Set Up Glue + Athena
  - Connect Quicksight Dashboard
- 📧 Email Reports Generated
- 🧪 Testing the Full Pipeline
- 🛡 IAM Roles & Permissions
- 📈 Future Enhancements
- 📜 License

---

## 📌 Project Overview

This project is a fully automated serverless AI-driven sentiment analysis and summarization pipeline built on AWS.

It ingests raw customer reviews, analyzes them using NLP, generates structured summaries using LLMs, produces global insights using Bedrock, and sends scheduled emails with results.

The system is:
- Lightweight
- Free-Tier friendly
- Minimal maintenance
- Highly scalable
- AI-enhanced

---

## ⚙️ Architecture Diagram

Paste your draw.io / Canva diagram here once generated.

---

## 🧱 Core AWS Services Used

Compute
- AWS Lambda → Review ingestion + product summarization
- Amazon SageMaker Endpoint → DistilBART LLM summarization model
- Amazon Bedrock (Titan Text Lite) → Global insights generation

Storage
- Amazon S3
  - input-bucket (raw review files)
  - processed-bucket
    - /processed traces
    - /summaries structured results
    - /summaries/aggregates global summary json
    - /insights bedrock insights

Datastores
- DynamoDB – Processed Reviews table
- DynamoDB – Review Summaries table

Analytics
- AWS Glue Database
- AWS Glue Crawlers (optional)
- Amazon Athena
- Amazon Quicksight

Orchestration
- EventBridge (scheduled triggers for both Lambdas)

Notifications
- SNS Topic
  - For ingest Lambda file processed
  - For summarizer run report
  - For full summaries
  - For AI insights (Bedrock)

Monitoring
- CloudWatch Logs
- CloudWatch Metrics

---

## 🚀 End-to-End Workflow

1. User uploads a CSV/JSON file → S3 Input Bucket
2. S3 Event → Process-Reviews Lambda
3. Process-Reviews Lambda:
   - Reads each review
   - Detects sentiment (Comprehend)
   - Extracts entities & key phrases
   - Stores results in DynamoDB - Processed Reviews
   - Saves trace logs in S3/processed
   - Publishes SNS email report
4. EventBridge Scheduled Trigger runs every hour/day
5. Summarizer Lambda:
   - Scans processed reviews
   - Groups by ProductID
   - Sends data to SageMaker endpoint (DistilBART)
   - Generates structured summary
   - Stores in DynamoDB - Summary Table
   - Saves output in S3/summaries
   - Generates aggregate JSON
   - Generates Bedrock insight
   - Sends 3 SNS emails: Run report, Full summaries, AI insights
6. Bedrock (Titan Text Lite) used to generate 1-paragraph global insights
7. Glue crawlers or Athena DDL make processed data queryable
8. Athena queries allow querying historical data
9. Quicksight Dashboard visualizes trends
10. All operations logged in CloudWatch

---

## 🛠️ Step-by-Step Implementation Guide

### 1️⃣ Create S3 Buckets
- poc-review-input-bucket
- poc-sentiment-processed

Inside processed bucket:
- /processed
- /summaries
- /summaries/aggregates
- /insights

Set lifecycle policies appropriate for traces and summaries (e.g., transition to S3 Glacier for old artifacts).

### 2️⃣ Build the Process-Reviews Lambda
- Trigger: S3 Object Created
- Reads file (CSV or JSON)
- Runs Comprehend for:
  - Sentiment
  - Key phrases
  - Entities
- Writes results to DynamoDB
- Stores trace NDJSON in S3
- Publishes SNS event
- Recommended runtime: Python 3.10
- Timeout: 2–3 mins
- Memory: 512 MB

Notes:
- Batch large files and/or use SQS for decoupling if single object processing may time out.
- Use pagination for Comprehend if processing long texts or batches.

### 3️⃣ Configure DynamoDB Tables

Processed Reviews Table:
- PK → ReviewID
- SK → (not required)
- Attributes: ProductID, UserID, ReviewText, Sentiment, SentimentScore, KeyPhrases, Entities, ProcessedAt

Summary Table:
- PK → ProductID
- SK → GeneratedAt (timestamp)
- Attributes: SummaryJSON, SummaryText, Metrics (avg sentiment, counts), ModelVersion

Indexes (optional):
- GSI for querying by sentiment
- GSI for querying by ProductID + date range

### 4️⃣ Build the Summarizer Lambda

Workflow:
- Scan all processed reviews (or use DynamoDB streams / time-windowed queries to limit scope)
- Group by product
- Build prompt/inputs for summarizer
- Invoke SageMaker endpoint (DistilBART)
- If SageMaker fails → fallback extractor (e.g., rule-based/heuristic summarizer)
- Store structured summary in Summary Table
- Upload summary to S3 (/summaries)
- Update /summaries/aggregates with global summary JSON
- Call Bedrock for higher-level insight (one-paragraph)
- Send SNS emails:
  - Run report
  - Full summaries (inline if small, otherwise S3 link)
  - AI insights (Bedrock output)

Implementation notes:
- Consider paginated DynamoDB scans to avoid timeouts.
- Use step functions for complex orchestration if retries/parallelism needed.

### 5️⃣ Create SageMaker Endpoint (DistilBART)
- Model: DistilBART (e.g., summarization variant)
- Deployment type: Serverless endpoint OR small instance like ml.t2.medium (cost/latency trade-offs)
- Invoke from Lambda using boto3.sagemaker-runtime.invoke_endpoint
- Consider model quantization or smaller model for cost reduction.

### 6️⃣ Enable Bedrock (Titan Text Lite)
- Region: us-east-1 (currently required)
- Model ID: amazon.titan-text-lite-v1
- Use for low-cost global insights generation (e.g., one-paragraph insights)
- Ensure IAM permission: "bedrock:InvokeModel"

### 7️⃣ Integrate SNS Notifications
- Create one SNS topic: poc-notifications-topic
- Lambdas publish:
  - File Processed report
  - Summarizer run report
  - Full summaries
  - Bedrock AI-generated insight
- Subscribe email endpoints or SES for formatted emails

### 8️⃣ Set Up EventBridge Scheduler
- Use EventBridge rules to trigger Summarizer Lambda
- Example schedules:
  - rate(1 hour) — hourly runs
  - cron(0 20 * * ? *) — daily at 8 PM UTC
- Optionally trigger Process-Reviews Lambda for periodic batch processing

### 9️⃣ Configure Glue + Athena
Option A: Use Glue Crawlers
- Point to processed bucket
- Auto-create tables for NDJSON/JSON/CSV

Option B: Use Athena CTAS / CREATE EXTERNAL TABLE
- Example:
```sql
CREATE EXTERNAL TABLE processed_reviews (
  ReviewID string,
  ProductID string,
  UserID string,
  ReviewText string,
  Sentiment string,
  SentimentScore double,
  KeyPhrases array<string>,
  Entities array<string>,
  ProcessedAt timestamp
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://poc-sentiment-processed/processed/';
```
- Build Athena views for analytics and Quicksight consumption.

### 🔟 Connect Quicksight Dashboard
Datasources:
- Athena
- DynamoDB (via direct connector if needed)

Visualizations:
- Sentiment trend over time
- Product comparison (average sentiment, counts)
- Key phrase cloud
- Summary table
- Overall sentiment distribution

---

## 📧 Email Reports Generated

You receive three emails after summarizer runs:

1. Run Summary
   - Attempted
   - Written
   - Skipped
   - Duration
   - Top products

2. Full Summaries
   - Inline if <200 KB
   - Otherwise S3 link to full JSON or NDJSON

3. AI Insights Email (Bedrock)
   - Top praised themes
   - Top complaints
   - Best & worst products
   - Trend overview

---

## 🧪 Testing the Full Pipeline

- Upload CSV → S3 input bucket
- Watch CloudWatch logs for both Lambdas
- Verify DynamoDB processed records
- Trigger summarizer manually (or wait for schedule)
- Verify SNS emails (or check SNS messages in CloudWatch / SNS console)
- Query Athena to confirm tables and data
- Refresh Quicksight dashboard to confirm visualizations

---

## 🛡 IAM Roles & Permissions

Lambdas require permissions for:
- S3 read/write
- DynamoDB read/write
- Comprehend (DetectSentiment, DetectKeyPhrases, DetectEntities)
- SageMaker runtime invoke
- Bedrock invoke
- SNS publish
- CloudWatch Logs

Minimal policy examples (adjust as least-privilege):
- "s3:GetObject", "s3:PutObject"
- "dynamodb:PutItem", "dynamodb:Query", "dynamodb:Scan"
- "comprehend:DetectSentiment", "comprehend:DetectKeyPhrases", "comprehend:DetectEntities"
- "sagemaker:InvokeEndpoint"
- "bedrock:InvokeModel"
- "sns:Publish"
- "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"

---

## 📈 Future Enhancements

- Add review deduplication
- Add multi-language support using Amazon Translate
- Expose results via API Gateway and Lambda for real-time queries
- Use RDS for complex relational analytics and joins
- Add product categorization using Comprehend Custom or custom classification models
- Use Step Functions for orchestrating complex multi-step summarization with retries


---
