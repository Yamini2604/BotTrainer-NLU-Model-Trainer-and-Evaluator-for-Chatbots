from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.mongo import db
from bson import ObjectId

router = APIRouter()

class CreateWS(BaseModel):
    name: str
    owner: str

@router.post("/create")
def create_workspace(data: CreateWS):
    ws_col = db.workspaces
    res = ws_col.insert_one({"name": data.name, "owner": data.owner})
    return {"status": "ok", "id": str(res.inserted_id)}

@router.get("/list/{email}")
def list_workspaces(email: str):
    ws_col = db.workspaces
    docs = list(ws_col.find({"owner": email}))
    for d in docs:
        d["id"] = str(d["_id"])
        d.pop("_id", None)
    return docs
