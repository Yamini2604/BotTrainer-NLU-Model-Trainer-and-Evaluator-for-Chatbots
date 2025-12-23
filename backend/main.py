# backend/app.
#uvicorn main:app --reload --port 8000
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, workspace, dataset, annotation, suggest, train_test, feedback,  active_learning, admin

app = FastAPI(title="BotTrainer")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(workspace.router, prefix="/workspace", tags=["Workspace"])
app.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
app.include_router(annotation.router, prefix="/annotation", tags=["Annotation"])
app.include_router(suggest.router, prefix="/suggest", tags=["Suggest"])
app.include_router(train_test.router, prefix="/train_test", tags=["TrainTest"])
app.include_router(feedback.router)
app.include_router(active_learning.router, prefix="/active", tags=["ActiveLearning"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

@app.get("/")
def root():
    return {"status": "running"}
