import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from ai_model.predictor import predict_risk

app = FastAPI(title="KidShield AI Search Monitoring Service")

class QueryRequest(BaseModel):
    query: str
    child_name: Optional[str] = "Child Account"
    parent_email: Optional[str] = None
    send_email: bool = True

class EmailRequest(BaseModel):
    parent_email: str
    child_name: str
    query: str
    category: str
    risk_level: str
    confidence: float
    explanation: str

def send_parent_email_alert(
    parent_email: str,
    child_name: str,
    query: str,
    category: str,
    risk_level: str,
    confidence: float,
    explanation: str
) -> bool:
    """Send an email alert to the parent email address when a search query is analyzed."""
    smtp_server = os.getenv("SMTP_SERVER", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    sender_email = os.getenv("SENDER_EMAIL", "alerts@kidshield.app")

    subject = f"[KidShield Alert] Search Detected: {query} ({category})"
    
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0;">
          <h2 style="color: #6C5CE7; margin-top: 0;">🛡️ KidShield Search Alert</h2>
          <p>Hello,</p>
          <p>A new web search was conducted on your linked child account <strong>{child_name}</strong>.</p>
          
          <div style="background-color: #f9f9fb; border-left: 4px solid {'#FF7675' if risk_level == 'High' else '#FDCB6E' if risk_level == 'Medium' else '#00B894'}; padding: 16px; margin: 16px 0; border-radius: 4px;">
            <p style="margin: 0 0 8px 0;"><strong>Search Query:</strong> "{query}"</p>
            <p style="margin: 0 0 8px 0;"><strong>AI Dataset Category:</strong> <span style="background: #eef; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{category}</span></p>
            <p style="margin: 0 0 8px 0;"><strong>Risk Level:</strong> {risk_level} ({confidence}%)</p>
            <p style="margin: 0;"><strong>Explanation:</strong> {explanation}</p>
          </div>
          
          <p style="color: #666; font-size: 13px;">This alert is recorded in your KidShield Parent Account ({parent_email}).</p>
          <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;" />
          <p style="color: #999; font-size: 11px;">KidShield Parental Controls & Child Safety AI</p>
        </div>
      </body>
    </html>
    """

    if smtp_server and smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = parent_email
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(sender_email, parent_email, msg.as_string())
            print(f"[SUCCESS] Email alert sent to {parent_email} for query '{query}'")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send email alert to {parent_email}: {e}")
            return False
    else:
        print(f"[SIMULATED EMAIL ALERT]")
        print(f"  To: {parent_email}")
        print(f"  Subject: {subject}")
        print(f"  Child: {child_name} | Query: '{query}' | Category: {category} | Risk: {risk_level} ({confidence}%)")
        print(f"  Status: Email queued and registered for KidShield parent account ({parent_email}).")
        return True

@app.post("/analyze-query")
async def analyze_query(data: QueryRequest, background_tasks: BackgroundTasks):
    query = data.query
    result = predict_risk(query)
    label = result["label"]
    confidence = round(result["score"] * 100, 2)

    # Risk level categorization according to ai_training dataset rules
    if label == "Safe":
        risk_level = "Low"
    elif label == "Toxic":
        risk_level = "Medium"
    elif label == "Cyberbullying":
        risk_level = "High"
    elif label == "Hate Speech":
        risk_level = "High"
    else:
        risk_level = "Medium"

    explanation = generate_explanation(label)

    email_sent = False
    if data.parent_email and data.send_email:
        background_tasks.add_task(
            send_parent_email_alert,
            parent_email=data.parent_email,
            child_name=data.child_name or "Child Account",
            query=query,
            category=label,
            risk_level=risk_level,
            confidence=confidence,
            explanation=explanation
        )
        email_sent = True

    return {
        "query": query,
        "category": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "email_sent": email_sent,
        "parent_email": data.parent_email
    }

@app.post("/send-email")
async def send_email_endpoint(data: EmailRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        send_parent_email_alert,
        parent_email=data.parent_email,
        child_name=data.child_name,
        query=data.query,
        category=data.category,
        risk_level=data.risk_level,
        confidence=data.confidence,
        explanation=data.explanation
    )
    return {
        "status": "queued",
        "parent_email": data.parent_email,
        "child_name": data.child_name,
        "category": data.category
    }

def generate_explanation(label):
    explanations = {
        "Safe": "The search appears safe and appropriate according to the AI training dataset.",
        "Toxic": "The search contains offensive language or toxic phrasing according to the AI training dataset.",
        "Cyberbullying": "The search may contain bullying, harassment, or targeted attacks according to the AI training dataset.",
        "Hate Speech": "The search contains hateful or discriminatory content according to the AI training dataset."
    }
    return explanations.get(
        label,
        "Potential harmful content detected according to the AI training dataset."
    )