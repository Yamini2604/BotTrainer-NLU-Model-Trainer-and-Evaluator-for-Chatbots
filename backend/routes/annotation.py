from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from utils.mongo import db
from bson import ObjectId

router = APIRouter()

class EntityModel(BaseModel):
    start: int
    end: int
    entity: str

class AnnotationModel(BaseModel):
    text: str
    intent: str
    entities: List[EntityModel]
    workspace_id: str = None
    index: int = None

@router.post("/save")
def save_annotation(payload: AnnotationModel):
    if not payload.workspace_id:
        raise HTTPException(400, "workspace_id required")

    # store in annotations collection (existing behavior)
    doc = {
        "workspace_id": payload.workspace_id,
        "text": payload.text,
        "intent": payload.intent,
        "entities": [{"start": e.start, "end": e.end, "label": e.entity} for e in payload.entities],
    }
    db.annotations.insert_one(doc)

    # --- NEW: sync annotation into dataset.items for this workspace if possible ---
    try:
        ds = db.datasets.find_one({"workspace_id": payload.workspace_id})
        if ds and "items" in ds:
            items = ds["items"]
            updated = False
            for it in items:
                # match by text; prefer items that don't already have an intent (unannotated)
                if isinstance(it.get("text"), str) and it.get("text").strip() == payload.text.strip():
                    # update the item with intent/entities (do not overwrite if already annotated)
                    if not it.get("intent"):
                        it["intent"] = payload.intent
                    if not it.get("entities"):
                        it["entities"] = [{"start": e.start, "end": e.end, "label": e.entity} for e in payload.entities]
                    updated = True
                    break
            if updated:
                db.datasets.replace_one({"workspace_id": payload.workspace_id}, {"workspace_id": payload.workspace_id, "items": items})
    except Exception:
        # non-fatal — don't block the save if sync fails
        pass

    return {"status": "ok"}


@router.get("/list/{workspace_id}")
def list_annotations(workspace_id: str):
    annots = list(db.annotations.find({"workspace_id": workspace_id}))
    for a in annots:
        a["id"] = str(a["_id"])
        a.pop("_id", None)
    return {"annotations": annots}


# helper exported for other modules
def get_annotations_by_workspace(workspace_id: str) -> List[Dict[str, Any]]:
    query = {"workspace_id": workspace_id}
    projection = {"_id": 0, "text": 1, "intent": 1, "entities": 1}
    cursor = db.annotations.find(query, projection)
    results = []
    for doc in cursor:
        if "text" in doc and "intent" in doc and "entities" in doc:
            results.append({
                "text": doc["text"],
                "intent": doc["intent"],
                "entities": doc["entities"],
            })
    return results
