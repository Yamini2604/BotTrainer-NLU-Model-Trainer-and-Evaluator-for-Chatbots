from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from utils.mongo import db
from utils.hashing import hash_password, verify_password

router = APIRouter()

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/signup")
def signup(req: SignupRequest):
    users = db.users
    if users.find_one({"email": req.email.lower()}):
        raise HTTPException(400, "Email already registered")
    hashed = hash_password(req.password)
    users.insert_one({"email": req.email.lower(), "password": hashed})
    return {"status": "ok", "message": "User created"}

@router.post("/login")
def login(req: LoginRequest):
    users = db.users
    user = users.find_one({"email": req.email.lower()})
    if not user:
        raise HTTPException(401, "Invalid credentials")
    if not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    # Minimal auth: return user id and email (token not used here)
    return {"status": "ok", "user_id": str(user["_id"]), "email": user["email"]}
