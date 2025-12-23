from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from utils.mongo import db
from bson import ObjectId
import datetime
import os
import shutil

router = APIRouter()


# ================================================================
# ADMIN LOGIN
# ================================================================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"     # <-- Change this anytime


@router.post("/login")
def admin_login(payload: dict):
    username = payload.get("username")
    password = payload.get("password")

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return {"message": "Login successful"}

    raise HTTPException(status_code=401, detail="Invalid credentials")


# ================================================================
# DASHBOARD SUMMARY
# ================================================================
@router.get("/summary")
def summary():
    users_count = db.users.count_documents({})
    workspaces_count = db.workspaces.count_documents({})
    datasets_count = db.datasets.count_documents({})
    annotated_count = db.annotations.count_documents({})
    train_runs = db.train_history.count_documents({})
    test_runs = db.test_history.count_documents({})

    return {
        "users_count": users_count,
        "workspaces_count": workspaces_count,
        "datasets_count": datasets_count,
        "annotated_count": annotated_count,
        "train_runs": train_runs,
        "test_runs": test_runs
    }


# ================================================================
# USERS LIST + DELETE
# ================================================================
@router.get("/users")
def list_users():
    curs = db.users.find({}, {"password": 0})
    out = []
    for u in curs:
        u["id"] = str(u["_id"])
        u.pop("_id", None)
        out.append(u)
    return {"users": out}


@router.delete("/user/{user_id}")
def delete_user(user_id: str):
    """
    Deleting a user should also delete:
    - workspaces created by that user
    - datasets inside those workspaces
    - annotations inside those workspaces
    - train/test history for those workspaces
    - rasa model folders
    """
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    email = user.get("email")

    # Workspaces owned by this user
    workspaces = list(db.workspaces.find({"owner": email}))

    for ws in workspaces:
        ws_id = str(ws["_id"])
        _delete_workspace_full(ws_id)

    # Finally delete the user
    db.users.delete_one({"_id": ObjectId(user_id)})

    return {"message": "User and all related data deleted successfully"}


# ================================================================
# WORKSPACES LIST + DELETE
# ================================================================
@router.get("/workspaces")
def list_workspaces():
    curs = db.workspaces.find({})
    out = []
    for w in curs:
        owner = db.users.find_one({"email": w.get("owner")}, {"password": 0})
        out.append({
            "id": str(w["_id"]),
            "name": w.get("name"),
            "owner_email": w.get("owner"),
            "owner": owner.get("email") if owner else None
        })
    return {"workspaces": out}


@router.delete("/workspace/{workspace_id}")
def delete_workspace(workspace_id: str):
    ws = db.workspaces.find_one({"_id": ObjectId(workspace_id)})
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    _delete_workspace_full(workspace_id)

    return {"message": "Workspace and all related content deleted"}


# Utility for deleting an entire workspace
def _delete_workspace_full(workspace_id: str):

    # Delete dataset
    db.datasets.delete_one({"workspace_id": workspace_id})

    # Delete annotations
    db.annotations.delete_many({"workspace_id": workspace_id})

    # Delete train history
    db.train_history.delete_many({"workspace_id": workspace_id})

    # Delete test history
    db.test_history.delete_many({"workspace_id": workspace_id})

    # Remove Rasa model folder if exists
    model_dir = f"models/{workspace_id}"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)

    # Finally delete workspace
    db.workspaces.delete_one({"_id": ObjectId(workspace_id)})


# ================================================================
# DATASETS LIST + DELETE
# ================================================================
@router.get("/datasets")
def list_datasets():
    curs = db.datasets.find({})
    out = []
    for d in curs:
        ws_id = d.get("workspace_id")
        ws = db.workspaces.find_one({"_id": ObjectId(ws_id)}) if ws_id else None
        owner_email = ws.get("owner") if ws else None
        out.append({
            "dataset_id": str(d["_id"]),
            "workspace_id": ws_id,
            "workspace_name": ws.get("name") if ws else None,
            "owner_email": owner_email,
            "rows": len(d.get("items", []))
        })
    return {"datasets": out}


@router.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str):
    doc = db.datasets.find_one({"_id": ObjectId(dataset_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.datasets.delete_one({"_id": ObjectId(dataset_id)})

    return {"message": "Dataset deleted successfully"}


# ================================================================
# DATASET + ANNOTATION PREVIEW
# ================================================================
@router.get("/dataset/preview/{workspace_id}")
def preview_dataset(workspace_id: str, limit: int = Query(20)):
    doc = db.datasets.find_one({"workspace_id": workspace_id})
    if not doc:
        return {"items": []}
    items = doc.get("items", [])[:limit]
    return {"items": items}


@router.get("/annotations")
def list_annotations(limit: int = Query(200)):
    cursor = db.annotations.find({}, {"_id": 1, "workspace_id": 1, "text": 1, "intent": 1, "entities": 1}).limit(limit)
    out = []
    for a in cursor:
        ws = db.workspaces.find_one({"_id": ObjectId(a.get("workspace_id"))}) if a.get("workspace_id") else None
        owner = None
        if ws:
            owner = ws.get("owner")
        out.append({
            "id": str(a["_id"]),
            "workspace_id": a.get("workspace_id"),
            "workspace_name": ws.get("name") if ws else None,
            "owner_email": owner,
            "text": a.get("text"),
            "intent": a.get("intent"),
            "entities": a.get("entities", [])
        })
    return {"annotations": out}


@router.get("/annotations/preview/{workspace_id}")
def preview_annotations(workspace_id: str, limit: int = Query(200)):
    cursor = db.annotations.find({"workspace_id": workspace_id}, {"_id": 1, "text": 1, "intent": 1, "entities": 1}).limit(limit)
    out = []
    for a in cursor:
        out.append({
            "id": str(a["_id"]),
            "text": a.get("text"),
            "intent": a.get("intent"),
            "entities": a.get("entities", [])
        })
    return {"annotations": out}

@router.delete("/annotation/{annotation_id}")
def delete_annotation(annotation_id: str):
    doc = db.annotations.find_one({"_id": ObjectId(annotation_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.annotations.delete_one({"_id": ObjectId(annotation_id)})
    return {"message": "Annotation deleted successfully"}



# ================================================================
# TRAIN/TEST HISTORY
# ================================================================
@router.get("/train_history")
def get_train_history(limit: int = Query(200)):
    cursor = db.train_history.find({}, sort=[("created_at", -1)]).limit(limit)
    out = []
    for r in cursor:
        ws_obj = db.workspaces.find_one({"_id": ObjectId(r.get("workspace_id"))})
        out.append({
            "id": str(r.get("_id")),
            "workspace_id": r.get("workspace_id"),
            "workspace_name": ws_obj.get("name") if ws_obj else None,
            "owner_email": ws_obj.get("owner") if ws_obj else None,
            "model": r.get("model"),
            "train_samples": r.get("train_samples"),
            "test_samples": r.get("test_samples"),
            "metrics": r.get("metrics", {}),
            "created_at": r.get("created_at").isoformat() if isinstance(r.get("created_at"), datetime.datetime) else r.get("created_at")
        })
    return {"train_history": out}


@router.get("/test_history")
def get_test_history(limit: int = Query(200)):
    cursor = db.test_history.find({}, sort=[("created_at", -1)]).limit(limit)
    out = []
    for r in cursor:
        ws_obj = db.workspaces.find_one({"_id": ObjectId(r.get("workspace_id"))})
        out.append({
            "id": str(r.get("_id")),
            "workspace_id": r.get("workspace_id"),
            "workspace_name": ws_obj.get("name") if ws_obj else None,
            "owner_email": ws_obj.get("owner") if ws_obj else None,
            "model": r.get("model"),
            "metrics": r.get("metrics", {}),
            "created_at": r.get("created_at").isoformat() if isinstance(r.get("created_at"), datetime.datetime) else r.get("created_at")
        })
    return {"test_history": out}

@router.get("/feedback")
def list_feedback(limit: int = 200):
    cursor = db.feedback.find({}, sort=[("created_at", -1)]).limit(limit)
    out = []

    for f in cursor:
        ws = db.workspaces.find_one({"_id": ObjectId(f.get("workspace_id"))})
        out.append({
            "id": str(f["_id"]),
            "workspace_name": ws.get("name") if ws else None,
            "workspace_id": f.get("workspace_id"),
            "user_email": f.get("user_email"),
            "rating": f.get("rating"),
            "comment": f.get("comment"),
            "created_at": f.get("created_at").isoformat()
        })

    return {"feedback": out}

