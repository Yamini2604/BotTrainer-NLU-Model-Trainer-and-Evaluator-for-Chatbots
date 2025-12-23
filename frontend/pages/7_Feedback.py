import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Feedback", layout="centered")
st.title("⭐ Feedback — Help Us Improve BotTrainer")

# Workspace
ws = st.session_state.get("selected_workspace")
user_email = st.session_state.get("email")  

if not ws:
    st.warning("Please open a workspace first")
    st.stop()

if not user_email:
    st.warning("Please login first")
    st.stop()

workspace_id = ws.get("id")
#user_email = user_email.get("email")

st.markdown("### How was your experience?")

# ⭐ STAR RATING
rating = st.radio(
    "Rate your experience",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

comment = st.text_area(
    "Additional comments (optional)",
    placeholder="Tell us what worked well or what can be improved..."
)

if st.button("Submit Feedback"):
    payload = {
        "workspace_id": workspace_id,
        "user_email": user_email,
        "rating": rating,
        "comment": comment
    }

    r = requests.post(f"{API}/feedback/submit", json=payload)

    if r.ok:
        st.success("✅ Thank you for your feedback!")
    else:
        st.error("❌ Failed to submit feedback")
