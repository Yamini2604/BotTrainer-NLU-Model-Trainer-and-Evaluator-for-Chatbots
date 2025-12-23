from fastapi import APIRouter, HTTPException
from utils.mongo import db
import datetime

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/submit")
def submit_feedback(payload: dict):
   

    db.feedback.insert_one({
        "workspace_id": payload["workspace_id"],
        "user_email": payload["user_email"],
        "rating": int(payload["rating"]),
        "comment": payload.get("comment", ""),
        "created_at": datetime.datetime.utcnow()
    })

    return {"message": "Feedback saved"}
