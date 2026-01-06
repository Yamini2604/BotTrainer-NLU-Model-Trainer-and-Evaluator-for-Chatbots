**BotTrainer** 🚀
An End-to-End NLP Model Training & Active Learning Platform

📌 **Overview**

BotTrainer is a full-stack NLP training platform that allows users to create, annotate, train, test, and continuously improve conversational AI models using active learning.
It supports multiple NLP frameworks such as Rasa and spaCy, and provides an admin dashboard for monitoring users, datasets, and feedback.
The system is designed to help teams iteratively improve chatbot performance through human-in-the-loop learning.

🏗️ **System Architecture**

Frontend (Streamlit)
        |
        v
Backend (FastAPI)
        |
        v
ML Models (Rasa / spaCy)
        |
        v
MongoDB Database

**How to Run the Project :**

1️⃣ **Backend Setup**

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

Backend runs at:

http://127.0.0.1:8000

2️⃣ **Frontend Setup**

cd frontend
streamlit run app.py

Frontend runs at:

http://localhost:8501
