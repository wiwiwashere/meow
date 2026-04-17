"""
Twilio SMS alert module.
 
Setup:
  1. Sign up at https://twilio.com (free trial works)
  2. Get your Account SID, Auth Token, and Twilio phone number from the dashboard
  3. Create a .env file in your project root with:
 
       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=your_auth_token_here
       TWILIO_FROM_NUMBER=+1xxxxxxxxxx
       TWILIO_TO_NUMBER=+1xxxxxxxxxx
 
Install:
  pip install twilio python-dotenv
"""
import os
import logging
from dotenv import load_dotenv
from twilio.rest import Client
 
load_dotenv()
logger = logging.getLogger(__name__)


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. whatsapp:+14155238886

# client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_whatsapp_alert(phone: str, label: str, confidence: float | None = None):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    # to_number = os.getenv("TWILIO_TO_NUMBER")

    missing = [k for k, v in {
        "TWILIO_ACCOUNT_SID": account_sid,
        "TWILIO_AUTH_TOKEN": auth_token,
        "TWILIO_WHATSAPP_FROM": from_number,
    }.items() if not v]

    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    body = f"There's {label} around here!"
    if confidence is not None:
        body += f"\nConfidence: {confidence:.2%}"

    client = Client(account_sid, auth_token)

    return client.messages.create(
        from_=from_number,
        to=f"whatsapp:{phone}",
        # to=to_number,
        body=body
    )
 
 
# def send_cat_alert(source: str = "unknown", image_path: str = None):
#     """
#     Send an SMS alert when a cat is detected.
 
#     Args:
#         confidence:  Model confidence score (e.g. 0.94 → 94%)
#         source:      Where the detection came from ('predict_single' or 'webcam')
#         image_path:  Optional path to the image that triggered the alert
#     """
#     account_sid = os.getenv("TWILIO_ACCOUNT_SID")
#     auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
#     from_number = os.getenv("TWILIO_FROM_NUMBER")
#     to_number   = os.getenv("TWILIO_TO_NUMBER")
 
#     # Validate env vars
#     missing = [k for k, v in {
#         "TWILIO_ACCOUNT_SID" : account_sid,
#         "TWILIO_AUTH_TOKEN"  : auth_token,
#         "TWILIO_FROM_NUMBER" : from_number,
#         "TWILIO_TO_NUMBER"   : to_number,
#     }.items() if not v]
 
#     if missing:
#         logger.error(f"Missing environment variables: {missing}")
#         logger.error("Create a .env file — see src/notifications/twilio_alert.py for instructions")
#         return False
 
#     # Build message
#     message_body = ("There's a cat around here!")
 
#     try:
#         client = Client(account_sid, auth_token)
#         message = client.messages.create(
#             body=message_body,
#             from_=from_number,
#             to=to_number
#         )
#         logger.info(f"SMS sent — SID: {message.sid}")
#         return True
 
#     except Exception as e:
#         logger.error(f"Failed to send SMS: {e}")
#         return False