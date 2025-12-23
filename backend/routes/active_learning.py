# backend/routes/active_learning.py
from fastapi import APIRouter, HTTPException, Query
from utils.mongo import db
from services.intent_suggester import suggest_intent
from services.entity_suggester import suggest_entities

router = APIRouter()

def pick_uncertain(predictions, threshold=0.70):
    return [p for p in predictions if p.get("confidence", 0.0) < threshold]

@router.post("/active/generate")
def generate_pool(workspace_id: str = Query(...), limit: int = 500):
    # read uploaded dataset
    ds = db.datasets.find_one({"workspace_id": workspace_id}) or {}
    items = ds.get("items", [])[:limit]
    if not items:
        raise HTTPException(404, "Dataset not found or empty. Upload dataset first.")

    predictions = []
    for row in items:
        text = row.get("text", "")
        if not text:
            continue
        intent = suggest_intent(text, workspace_id)
        entities = suggest_entities(text, workspace_id)
        predictions.append({
            "workspace_id": workspace_id,
            "text": text,
            "intent": intent,
            "entities": entities,
            "done": False
        })

    uncertain = pick_uncertain(predictions)
    # replace pool docs
    db.active_learning_pool.delete_many({"workspace_id": workspace_id})
    if uncertain:
        # store with minimal fields
        to_insert = [{"workspace_id": p["workspace_id"], "text": p["text"], "intent": p["intent"], "entities": p["entities"], "done": False} for p in uncertain]
        db.active_learning_pool.insert_many(to_insert)

    return {"status": "ok", "generated": len(predictions), "uncertain": len(uncertain)}

@router.get("/active/next/{workspace_id}")
def get_next_item(workspace_id: str):
    doc = db.active_learning_pool.find_one({"workspace_id": workspace_id, "done": False})
    if not doc:
        return {"item": None}
    item = {
        "id": str(doc["_id"]),
        "workspace_id": doc["workspace_id"],
        "text": doc["text"],
        "intent": doc.get("intent", {}),
        "entities": doc.get("entities", [])
    }
    return {"item": item}

@router.post("/active/mark_done/{workspace_id}")
def mark_done(workspace_id: str, item_id: str, intent: str, entities: list):
    pool = db.active_learning_pool
    try:
        from bson import ObjectId
        doc = pool.find_one({"_id": ObjectId(item_id), "workspace_id": workspace_id})
    except Exception:
        raise HTTPException(400, "Invalid item id")

    if not doc:
        raise HTTPException(404, "Item not found in active pool.")

    # Save annotation into main collection
    db.annotations.insert_one({
        "workspace_id": workspace_id,
        "text": doc["text"],
        "intent": intent,
        "entities": entities
    })
    # mark done
    pool.update_one({"_id": doc["_id"]}, {"$set": {"done": True}})
    return {"status": "ok", "message": "Saved and marked done"}
