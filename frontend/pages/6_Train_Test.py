# frontend/pages/6_Train_Test.py
import streamlit as st
import requests
import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="BotTrainer - Annotate & Train", layout="wide")
st.title("BotTrainer — Annotate, Train & Compare (spaCy vs Rasa-like)")

# ======== Fetch workspaces =========
user_email =  st.session_state.get("email")  

@st.cache_data(show_spinner=False)
def fetch_workspaces(email: str) -> List[dict]:
    res = requests.get(f"{API}/workspace/list/{email}")
    if res.ok:
        return res.json()
    else:
        return []

workspaces = fetch_workspaces(user_email)
if not workspaces:
    st.error("No workspaces found for your account.")
    st.stop()

workspace_map = {ws['name']: ws['id'] for ws in workspaces}
selected_ws_name = st.sidebar.selectbox("Select Workspace", options=list(workspace_map.keys()))
workspace_id = workspace_map[selected_ws_name]

# ======== Fetch annotations =========
@st.cache_data(show_spinner=False)
def fetch_annotations(ws_id: str) -> List[dict]:
    res = requests.get(f"{API}/annotation/list/{ws_id}")
    if res.ok:
        data = res.json()
        return data.get("annotations", [])
    else:
        return []

annotations = fetch_annotations(workspace_id)
st.sidebar.markdown(f"### Annotations: {len(annotations)} examples loaded")

with st.expander("Show Sample Annotations"):
    for ann in annotations[:10]:
        st.markdown(f"**Text:** {ann['text']}")
        st.markdown(f"**Intent:** {ann.get('intent', 'N/A')}")
        st.markdown(f"**Entities:** {ann.get('entities', [])}")
        st.markdown("---")

# ========= Annotation Form =========
#st.header("Add New Annotation")
#with st.form("annotate_form", clear_on_submit=True):
 #   text = st.text_area("Text")
  #  intent = st.text_input("Intent")
   # entities_str = st.text_area(
    #    "Entities (JSON list, e.g. [{\"start\":0,\"end\":5,\"label\":\"entity\"}])"
   # )
    #submitted = st.form_submit_button("Save Annotation")
    #if submitted:
     #   try:
      #      entities = json.loads(entities_str) if entities_str else []
       # except Exception as e:
        #    st.error(f"Invalid entities JSON: {e}")
         #   entities = []
        #payload = {"workspace_id": workspace_id, "text": text, "intent": intent, "entities": entities}
        #res = requests.post(f"{API}/annotation/save", json=payload)
        #if res.ok:
         #   st.success("Annotation saved successfully!")
          #  fetch_annotations.clear()
        #else:
         #   st.error(f"Failed to save annotation: {res.text}")

# ========= Train & Test =========
st.header("Train & Test Model")

split_ratio = st.slider("Train/Test Split Ratio (Train %)", min_value=0, max_value=100, value=80)

col_train = st.columns([2,1])[0]
if col_train.button("Train (both spaCy & Rasa)"):
    with st.spinner("Training spaCy..."):
        try:
            res_spacy = requests.post(f"{API}/train_test/train", params={"workspace_id": workspace_id, "train_split_ratio": split_ratio/100.0})
            if res_spacy.ok:
                st.success("spaCy training completed")
            else:
                st.error(f"spaCy training failed: {res_spacy.text}")
        except Exception as e:
            st.error(f"spaCy training error: {e}")

    with st.spinner("Training Rasa..."):
        try:
            res_rasa = requests.post(f"{API}/train_test/train_rasa", params={"workspace_id": workspace_id, "train_split_ratio": split_ratio/100.0})
            if res_rasa.ok:
                st.success("Rasa training completed")
            else:
                st.error(f"Rasa training failed: {res_rasa.text}")
        except Exception as e:
            st.error(f"Rasa training error: {e}")

st.markdown("---")

# Buttons to test each model
col1, col2 = st.columns(2)
with col1:
    if st.button("Test (spaCy)"):
        with st.spinner("Testing spaCy..."):
            try:
                r = requests.post(f"{API}/train_test/test", params={"workspace_id": workspace_id})
                if r.ok:
                    st.session_state["spacy_results"] = r.json()
                    st.success("spaCy test completed")
                else:
                    st.error(f"spaCy test failed: {r.text}")
            except Exception as e:
                st.error(f"spaCy test error: {e}")

with col2:
    if st.button("Test (Rasa)"):
        with st.spinner("Testing Rasa..."):
            try:
                r = requests.post(f"{API}/train_test/test_rasa", params={"workspace_id": workspace_id})
                if r.ok:
                    st.session_state["rasa_results"] = r.json()
                    st.success("Rasa test completed")
                else:
                    st.error(f"Rasa test failed: {r.text}")
            except Exception as e:
                st.error(f"Rasa test error: {e}")

# Show individual results if available
st.header("Results")

if "spacy_results" in st.session_state:
    sr = st.session_state["spacy_results"]
    st.subheader("spaCy Results")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{sr.get('accuracy',0):.3f}")
    c2.metric("Precision", f"{sr.get('precision',0):.3f}")
    c3.metric("Recall", f"{sr.get('recall',0):.3f}")
    c4.metric("F1 Score", f"{sr.get('f1_score',0):.3f}")
    st.write(f"Samples tested: {sr.get('samples_tested',0)}")

if "rasa_results" in st.session_state:
    rr = st.session_state["rasa_results"]
    st.subheader("Rasa Results")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Accuracy", f"{rr.get('accuracy',0):.3f}")
    d2.metric("Precision", f"{rr.get('precision',0):.3f}")
    d3.metric("Recall", f"{rr.get('recall',0):.3f}")
    d4.metric("F1 Score", f"{rr.get('f1_score',0):.3f}")
    st.write(f"Samples tested: {rr.get('samples_tested',0)}")

# Comparison
if "spacy_results" in st.session_state and "rasa_results" in st.session_state:
    st.markdown("---")
    st.header("Compare Models")

    sp = st.session_state["spacy_results"]
    ra = st.session_state["rasa_results"]

    # metrics arrays for bar chart
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    sp_vals = [sp.get(m,0) for m in metrics]
    ra_vals = [ra.get(m,0) for m in metrics]

    # Bar chart
    fig, ax = plt.subplots(figsize=(3.5,2))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, sp_vals, width, label="spaCy")
    ax.bar(x + width/2, ra_vals, width, label="Rasa")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()
    st.pyplot(fig)

    # Confusion matrices
    intents = sp.get("all_intents") or ra.get("all_intents") or []
    cm_sp = np.array(sp.get("confusion_matrix", []))
    cm_ra = np.array(ra.get("confusion_matrix", []))

    def plot_cm(cm, labels, title):
        fig, ax = plt.subplots(figsize=(6,5))
        if cm.size == 0:
            ax.text(0.5,0.5,"No confusion matrix data", ha="center")
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, title=title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        st.pyplot(fig)

    st.subheader("spaCy Confusion Matrix")
    plot_cm(cm_sp, intents, "spaCy Confusion Matrix")

    st.subheader("Rasa Confusion Matrix")
    plot_cm(cm_ra, intents, "Rasa Confusion Matrix")

    # Decide best model by f1 (weighted); tie-break by accuracy
    sp_f1 = sp.get("f1_score", 0)
    ra_f1 = ra.get("f1_score", 0)
    if sp_f1 > ra_f1:
        best = "spaCy"
    elif ra_f1 > sp_f1:
        best = "Rasa"
    else:
        # tie-break
        sp_acc = sp.get("accuracy", 0)
        ra_acc = ra.get("accuracy", 0)
        best = "spaCy" if sp_acc >= ra_acc else "Rasa"

    st.markdown("---")
    st.subheader("Recommendation")
    st.write(f"**Best model for this workspace:** **{best}** (decision based on weighted F1 score).")

