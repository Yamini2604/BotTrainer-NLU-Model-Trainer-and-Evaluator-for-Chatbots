import streamlit as st
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Active Learning", layout="wide")
st.title("Active Learning — Improve Your Models (spaCy & Rasa)")

# ========= Workspace =========
ws = st.session_state.get("selected_workspace")
if not ws:
    st.warning("Open workspace first")
    st.stop()

workspace_id = ws.get("id")

# ========= Helper: Deduplicate by text =========
def deduplicate_results(rows):
    unique = {}
    for r in rows:
        txt = r["text"]
        # Keep only lowest confidence per text
        if txt not in unique or r["confidence"] < unique[txt]["confidence"]:
            unique[txt] = r
    return list(unique.values())

# ========= Step 1: Choose Model =========
st.subheader("1️⃣ Choose Model for Active Learning")

model_choice = st.radio(
    "Select model to fetch uncertain predictions:",
    ["spaCy", "Rasa"],
    horizontal=True
)

# ========= Step 2: Run Active Learning =========
if st.button("Run Active Learning Test"):
    with st.spinner("Running evaluation and fetching uncertain samples..."):

        if model_choice == "spaCy":
            r = requests.post(f"{API}/train_test/test", params={"workspace_id": workspace_id})
        else:
            r = requests.post(f"{API}/train_test/test_rasa", params={"workspace_id": workspace_id})

        if not r.ok:
            st.error("Failed to fetch uncertain predictions")
            st.stop()

        results = r.json()
        detailed = results.get("detailed_results", [])

        # ⬇️ Deduplicate
        detailed = deduplicate_results(detailed)

        # Filter uncertain (< 0.70)
        uncertain = [row for row in detailed if row["confidence"] < 0.70]

        st.session_state["uncertain_samples"] = uncertain

# ========= Step 3: Display Table =========
st.subheader("2️⃣ Uncertain Samples — Review & Correct")

if "uncertain_samples" not in st.session_state or len(st.session_state["uncertain_samples"]) == 0:
    st.info("Run Active Learning Test to view uncertain samples.")
    st.stop()

samples = st.session_state["uncertain_samples"]

df = pd.DataFrame(samples)
df.insert(0, "S.No", range(1, len(df)+1))

st.markdown("### 🔍 Uncertain Predictions Table")

# Table Layout
for idx, row in df.iterrows():
    with st.container():
        cols = st.columns([1, 3, 3, 2, 3, 2])

        cols[0].write(row["S.No"])
        cols[1].write(row["text"])
        cols[2].write(row["predicted_intent"])
        cols[3].write(f"{row['confidence']:.2f}")

        corrected_intent = cols[4].text_input(
            "Enter Correct Intent",
            value=row["predicted_intent"],
            key=f"intent_{idx}"
        )

        if cols[5].button("Save", key=f"save_{idx}"):

            payload = {
                "workspace_id": workspace_id,
                "text": row["text"],
                "intent": corrected_intent,
                "entities": []
            }

            r = requests.post(f"{API}/annotation/save", json=payload)

            if r.ok:
                st.success(f"Saved corrected annotation for row {row['S.No']}")
            else:
                st.error("Failed to save annotation")

st.markdown("---")

# ========= Step 4: Retrain & Retest =========
st.subheader("3️⃣ Retrain & Retest Models")

train_col1, train_col2 = st.columns(2)

with train_col1:
    if st.button("Retrain spaCy"):
        with st.spinner("Retraining spaCy model..."):
            r = requests.post(f"{API}/train_test/train", params={"workspace_id": workspace_id})
            st.success("spaCy model retrained successfully!")

with train_col2:
    if st.button("Retrain Rasa"):
        with st.spinner("Retraining Rasa model..."):
            r = requests.post(f"{API}/train_test/train_rasa", params={"workspace_id": workspace_id})
            st.success("Rasa model retrained successfully!")

st.markdown("---")

# ========= Step 5: Retest + Show Metrics =========
st.subheader("4️⃣ Retest Updated Models")

retest_model = st.radio("Select model to retest:", ["spaCy", "Rasa"], horizontal=True)

if st.button("Run Retest"):
    with st.spinner("Running retest..."):

        if retest_model == "spaCy":
            r = requests.post(f"{API}/train_test/test", params={"workspace_id": workspace_id})
        else:
            r = requests.post(f"{API}/train_test/test_rasa", params={"workspace_id": workspace_id})

        if not r.ok:
            st.error("Retest failed")
            st.stop()

        results = r.json()
        detailed = results.get("detailed_results", [])

        # Deduplicate again
        detailed = deduplicate_results(detailed)

        st.success("Retest completed — results displayed below!")

       
        # ========= Show Full Result Table =========
        st.markdown("### 📋 Detailed Retest Results")
        result_df = pd.DataFrame(detailed)
        st.dataframe(result_df, use_container_width=True)
