import streamlit as st
import requests
import pandas as pd
import json

API = "http://127.0.0.1:8000"

st.title("Upload Dataset")

ws = st.session_state.get("selected_workspace")
if not ws:
    st.warning("Open a workspace first in Workspaces page")
    st.stop()

workspace_id = ws.get("id")

# --- 1. Fetch existing dataset ---
resp = requests.get(f"{API}/dataset/fetch/{workspace_id}")
dataset = []
if resp and resp.status_code == 200:
    dataset = resp.json().get("dataset", [])

if dataset:
    st.subheader("Uploaded Dataset (Preview & Select)")
    # Show preview of first few rows
    preview_df = pd.DataFrame(dataset).head(10)
    st.write(preview_df)

    if st.button("Load this dataset for annotation"):
        st.session_state["current_dataset"] = dataset
        st.session_state["dataset_id"] = workspace_id
        st.success("Dataset loaded in session")
else:
    st.info("No dataset uploaded yet")

# --- 2. Upload new dataset ---
st.subheader("Upload New Dataset")

uploaded = st.file_uploader("CSV or JSON", type=["csv","json"])
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            content = uploaded.getvalue()
            parsed = json.loads(content)
            if isinstance(parsed, list) and all(isinstance(i,str) for i in parsed):
                df = pd.DataFrame({"query": parsed})
            else:
                df = pd.DataFrame(parsed)
        st.write("Preview", df.head())

        # pick text column
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ("query","text","utterance","message"))]
        if text_cols:
            col = text_cols[0]
        else:
            col = df.columns[0]

        if st.button("Upload to backend"):
            # build items
            items = [{"text": str(x)} for x in df[col].dropna().astype(str).tolist()]
            resp = requests.post(
                f"{API}/dataset/upload/{workspace_id}",
                files={"file": ("dataset.json", json.dumps(items), "application/json")}
            )
            if resp and resp.status_code == 200:
                st.success("Uploaded")
                # Save dataset into session by fetching back
                fetch = requests.get(f"{API}/dataset/fetch/{workspace_id}")
                if fetch and fetch.status_code == 200:
                    st.session_state["current_dataset"] = fetch.json().get("dataset", [])
                    st.session_state["dataset_id"] = workspace_id
                    st.success("Dataset loaded in session")
                else:
                    st.error("Uploaded but fetch failed")
            else:
                st.error("Upload failed")
    except Exception as e:
        st.error(f"Parse error: {e}")
