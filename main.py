import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from fastapi import Depends, FastAPI, BackgroundTasks, Header, HTTPException, status
from pydantic import BaseModel

from ai_model.predictor import predict_risk
from ai_model.explainability import analyze_with_xai

app = FastAPI(title="KidShield AI Search Monitoring Service")

API_KEY = os.getenv("KIDSHIELD_API_KEY", "")


def verify_api_key(x_api_key: str = Header(default="")):
    """Gate access with a shared secret so only the KidShield app (which
    knows the key) can reach these endpoints — this is not per-user auth,
    just a bar against anyone on the internet finding the URL and calling it."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfigured: KIDSHIELD_API_KEY is not set.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

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
    key_words: Optional[str] = ""
    suggested_action: Optional[str] = ""

def send_parent_email_alert(
    parent_email: str,
    child_name: str,
    query: str,
    category: str,
    risk_level: str,
    confidence: float,
    explanation: str,
    key_words: str = "",
    suggested_action: str = "",
) -> bool:
    """Send an immediate email alert for HIGH risk searches."""
    smtp_server = os.getenv("SMTP_SERVER", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    sender_email = os.getenv("SENDER_EMAIL", "alerts@kidshield.app")

    border_color = "#FF7675" if risk_level == "High" else "#FDCB6E" if risk_level == "Medium" else "#00B894"
    subject = f"[KidShield HIGH RISK] Search Detected: {query}"

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e0e0e0;">
          <h2 style="color: #FF7675; margin-top: 0;">HIGH RISK SEARCH DETECTED</h2>
          <p>Hello,</p>
          <p>A concerning web search was detected on your linked child account <strong>{child_name}</strong>.</p>

          <div style="background-color: #fff5f5; border-left: 4px solid {border_color}; padding: 16px; margin: 16px 0; border-radius: 4px;">
            <p style="margin: 0 0 8px 0;"><strong>Search Query:</strong> "{query}"</p>
            <p style="margin: 0 0 8px 0;"><strong>Category:</strong> {category}</p>
            <p style="margin: 0 0 8px 0;"><strong>Confidence:</strong> {confidence}%</p>
            <p style="margin: 0 0 8px 0;"><strong>Risk Level:</strong> {risk_level}</p>
            <p style="margin: 0 0 8px 0;"><strong>Key Words:</strong> {key_words or "—"}</p>
            <p style="margin: 0 0 8px 0;"><strong>Explanation:</strong> {explanation}</p>
            <p style="margin: 0;"><strong>Suggested Action:</strong> {suggested_action or "Review this search with your child."}</p>
          </div>

          <p style="color: #666; font-size: 13px;">This alert is recorded in your KidShield Parent Account ({parent_email}). Open the app to review full search history.</p>
          <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;" />
          <p style="color: #999; font-size: 11px;">KidShield Parental Controls &amp; Child Safety AI</p>
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
            print(f"[SUCCESS] High-risk email alert sent to {parent_email} for query '{query}'")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send email alert to {parent_email}: {e}")
            return False
    else:
        print(f"[SIMULATED HIGH-RISK EMAIL ALERT]")
        print(f"  To: {parent_email}")
        print(f"  Query: '{query}' | Category: {category} | Risk: {risk_level} ({confidence}%)")
        print(f"  Key Words: {key_words}")
        print(f"  Explanation: {explanation}")
        print(f"  Suggested Action: {suggested_action}")
        return True

@app.post("/analyze-query", dependencies=[Depends(verify_api_key)])
async def analyze_query(data: QueryRequest, background_tasks: BackgroundTasks):
    query = data.query
    model_result = predict_risk(query)
    xai = analyze_with_xai(model_result["label"], query, model_result["label_id"])

    label = xai["category"]
    risk_level = xai["risk_level"]
    key_words = xai["key_words"]
    explanation = xai["explanation"]
    suggested_action = xai["suggested_action"]
    confidence = round(model_result["score"] * 100, 2)

    # Alert actions by risk level:
    # Low (Safe) → query logged only
    # Medium (Toxic, Inappropriate) → dashboard flag + in-app notification
    # High (Cyberbullying, Hate, Violent) → dashboard flag + immediate email
    is_flagged = risk_level in ("Medium", "High")
    should_email = risk_level == "High" and data.send_email

    email_sent = False
    if data.parent_email and should_email:
        background_tasks.add_task(
            send_parent_email_alert,
            parent_email=data.parent_email,
            child_name=data.child_name or "Child Account",
            query=query,
            category=label,
            risk_level=risk_level,
            confidence=confidence,
            explanation=explanation,
            key_words=", ".join(key_words),
            suggested_action=suggested_action,
        )
        email_sent = True

    return {
        "query": query,
        "category": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "is_flagged": is_flagged,
        "key_words": key_words,
        "explanation": explanation,
        "suggested_action": suggested_action,
        "email_sent": email_sent,
        "parent_email": data.parent_email,
    }

@app.post("/send-email", dependencies=[Depends(verify_api_key)])
async def send_email_endpoint(data: EmailRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        send_parent_email_alert,
        parent_email=data.parent_email,
        child_name=data.child_name,
        query=data.query,
        category=data.category,
        risk_level=data.risk_level,
        confidence=data.confidence,
        explanation=data.explanation,
        key_words=data.key_words or "",
        suggested_action=data.suggested_action or "",
    )
    return {
        "status": "queued",
        "parent_email": data.parent_email,
        "child_name": data.child_name,
        "category": data.category,
    }
