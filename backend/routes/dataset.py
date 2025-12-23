from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.mongo import db
import pandas as pd
import json
import io

router = APIRouter()

@router.post("/upload/{workspace_id}")
async def upload_dataset(workspace_id: str, file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            # pick first column as text if needed
            if df.shape[1] == 1:
                col = df.columns[0]
                items = [{"text": str(x)} for x in df[col].dropna().astype(str).tolist()]
            else:
                # use column named query/text or first column
                candidate = None
                for c in df.columns:
                    if any(k in c.lower() for k in ("query","text","utterance","message")):
                        candidate = c
                        break
                if candidate is None:
                    candidate = df.columns[0]
                items = [{"text": str(x)} for x in df[candidate].dropna().astype(str).tolist()]

        elif file.filename.lower().endswith(".json"):
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # list of strings? convert
                if all(isinstance(i, str) for i in parsed):
                    items = [{"text": s} for s in parsed]
                elif all(isinstance(i, dict) for i in parsed):
                    # expect dicts with "query" or "text"
                    items = []
                    for obj in parsed:
                        if "query" in obj:
                            items.append({"text": str(obj["query"])})
                        elif "text" in obj:
                            items.append({"text": str(obj["text"])})
                        else:
                            # take first string-like field
                            found = None
                            for v in obj.values():
                                if isinstance(v, str):
                                    found = v
                                    break
                            if found:
                                items.append({"text": found})
            else:
                raise HTTPException(400, "JSON must be a list")
        else:
            raise HTTPException(400, "Unsupported file type")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse upload: {e}")

    # store dataset doc in Mongo
    db.datasets.replace_one({"workspace_id": workspace_id}, {"workspace_id": workspace_id, "items": items}, upsert=True)
    return {"status": "ok", "rows": len(items)}

@router.get("/fetch/{workspace_id}")
def fetch_dataset(workspace_id: str):
    doc = db.datasets.find_one({"workspace_id": workspace_id})
    if not doc:
        return {"dataset": []}
    return {"dataset": doc.get("items", [])}
