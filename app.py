from dotenv import load_dotenv
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")

if not MONGODB_URI or not MONGODB_DB:
    raise RuntimeError("Missing MONGODB_URI or MONGODB_DB")

mongo = MongoClient(MONGODB_URI)
db = mongo[MONGODB_DB]
embeddings_col = db["embeddings"]

embeddings_col.delete_many({})

data = [
    """Provident Fund (PF) Enrollment
Topic: PF Enrollment Process 
• When activated: Your PF status is activated in HRIS.
• Next steps (automated email): You will receive an email from HRIS (check your inbox and spam folder).
• Action required by employee: Complete the form, upload the affidavit, and fill the survey by the specified deadline in the email.
• Benefit: Ensures your PF contributions are processed smoothly.
""",

    """Health Insurance Claims (Non-Panel Hospitals)
Topic: Health Insurance Claims for Non-Panel Hospitals 
Provider: EFU Insurance Company.
• Policy: PITB's corporate health insurance (Master Policy ID: EM/001055-00).
• Scenario: For visits to hospitals NOT on EFU's approved panel list.
• How to submit: Claims must be submitted along with all original supporting receipts and the duly attested doctor's form.
• Required documents:
  o Completed EFU Health Insurance Claim Form.
  o Original Medical Bills/Invoices.
  o Original Prescriptions.
  o Diagnostic Reports (if applicable).
  o Discharge Summary (for hospitalization).
  o Copy of Employee's CNIC.
  o Copy of Patient's CNIC/B-Form (if dependent).
  o Doctor's Recommendation/Prescription.
""",

    """HR Service Delivery Overview
Topic: HR Services Overview / How HR Can Help You 
Self-Service Portal: Access personal information, leave requests, payslips, benefits, and company policies.
"""
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(data, normalize_embeddings=True)

docs = [{"text": text, "embedding": embedding.tolist()} for text, embedding in zip(data, embeddings)]
result = embeddings_col.insert_many(docs)
