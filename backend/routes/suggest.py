# backend/routes/suggest.py
from fastapi import APIRouter, Query
from services.intent_suggester import suggest_intent
from services.entity_suggester import suggest_entities

router = APIRouter()

@router.get("/nlp/suggest-intent")
def nlp_suggest_intent(workspace_id: str, q: str = Query(...)):
    """
    Returns: { "intent": {"intent": "...", "confidence": 0.92 } }
    or { "intent": None, "confidence": 0.0 }
    """
    res = suggest_intent(q, workspace_id)
    return {"intent": res}

@router.get("/nlp/suggest-entities")
def nlp_suggest_entities(workspace_id: str, q: str = Query(...)):
    res = suggest_entities(q, workspace_id)
    return {"entities": res}
