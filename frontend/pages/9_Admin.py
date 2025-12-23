import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Admin Portal", layout="wide")

# ------------------------------------------------------------
# SESSION CHECK — Require admin login
# ------------------------------------------------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if not st.session_state.admin_logged_in:
    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        r = requests.post(f"{API}/admin/login", json={
            "username": username,
            "password": password
        })
        if r.ok:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ------------------------------------------------------------
# ADMIN PORTAL CONTENT
# ------------------------------------------------------------
st.title("Admin Portal — BotTrainer")

def fetch_summary():
    r = requests.get(f"{API}/admin/summary")
    if r.ok:
        return r.json()
    return {}

summary = fetch_summary()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Users", summary.get("users_count", 0))
col2.metric("Workspaces", summary.get("workspaces_count", 0))
col3.metric("Datasets uploaded", summary.get("datasets_count", 0))
col4.metric("Annotated rows", summary.get("annotated_count", 0))
col5.metric("Train runs", summary.get("train_runs", 0))

st.markdown("---")

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "👤 Users",
    "🗂 Workspaces",
    "📂 Datasets",
    "📝 Annotated Data",
    "🛠 Train History",
    "📊 Test History",
    "⭐ Feedback",
    "🚪 Logout"
])

# ------------------------------------------------------------
# TAB 1 — USERS
# ------------------------------------------------------------
with tab1:
    st.subheader("Users List")

    r = requests.get(f"{API}/admin/users")
    if r.ok:
        users = r.json().get("users", [])

        for u in users:
            col1, col2, col3 = st.columns([4, 3, 2])

            with col1:
                st.write(f"{u.get('email','')}")

            with col2:
                st.write(f"User ID: `{u['id']}`")

            with col3:
                if st.button("❌ Delete", key=f"del_user_{u['id']}"):
                    dr = requests.delete(f"{API}/admin/user/{u['id']}")
                    if dr.ok:
                        st.success("User deleted successfully")
                        st.rerun()
                    else:
                        st.error("Delete failed")

# ------------------------------------------------------------
# TAB 2 — WORKSPACES
# ------------------------------------------------------------
with tab2:
    st.subheader("Workspaces List")

    r = requests.get(f"{API}/admin/workspaces")
    if r.ok:
        workspaces = r.json().get("workspaces", [])

        for w in workspaces:
            col1, col2, col3 = st.columns([4, 3, 2])

            with col1:
                st.write(f"**{w['name']}**  \nOwner: {w['owner_email']}")

            with col2:
                st.write(f"Workspace ID: `{w['id']}`")

            with col3:
                if st.button("❌ Delete", key=f"del_ws_{w['id']}"):
                    dr = requests.delete(f"{API}/admin/workspace/{w['id']}")
                    if dr.ok:
                        st.success("Workspace deleted")
                        st.rerun()
                    else:
                        st.error("Delete failed")

# ------------------------------------------------------------
# TAB 3 — DATASETS
# ------------------------------------------------------------
with tab3:
    st.subheader("Datasets List")

    r = requests.get(f"{API}/admin/datasets")
    if r.ok:
        datasets = r.json().get("datasets", [])

        for d in datasets:
            col1, col2, col3 = st.columns([4, 3, 2])

            with col1:
                st.write(
                    f"**Workspace:** {d['workspace_name']}  \n"
                    f"Rows: {d['rows']}  \n"
                    f"Owner: {d['owner_email']}"
                )

            with col2:
                st.write(f"Dataset ID: `{d['dataset_id']}`")

            with col3:
                if st.button("❌ Delete", key=f"del_ds_{d['dataset_id']}"):
                    dr = requests.delete(f"{API}/admin/dataset/{d['dataset_id']}")
                    if dr.ok:
                        st.success("Dataset deleted")
                        st.rerun()
                    else:
                        st.error("Delete failed")

# ------------------------------------------------------------
# TAB 4 — ANNOTATED DATA
# ------------------------------------------------------------
with tab4:
    st.subheader("Annotated Dataset")

    r = requests.get(f"{API}/admin/annotations")
    if r.ok:
        annotations = r.json().get("annotations", [])

        for ann in annotations:
            ann_id = ann["id"]
            ws_name = ann.get("workspace_name")
            ws_id = ann.get("workspace_id")
            owner = ann.get("owner_email")
            text = ann.get("text")
            intent = ann.get("intent")
            entities = ann.get("entities", [])

            col1, col2, col3 = st.columns([5, 4, 2])

            with col1:
                st.write(
                    f"**Text:** {text}  \n"
                    f"**Intent:** {intent}  \n"
                    f"**Entities:** {entities}"
                )

            with col2:
                st.write(
                    f"Workspace: **{ws_name}**  \n"
                    f"WS ID: `{ws_id}`  \n"
                    f"Ann ID: `{ann_id}`  \n"
                    f"Owner: {owner}"
                )

            with col3:
                if st.button("❌ Delete", key=f"del_ann_{ann_id}"):
                    dr = requests.delete(f"{API}/admin/annotation/{ann_id}")
                    if dr.ok:
                        st.success("Annotation deleted")
                        st.rerun()
                    else:
                        st.error("Delete failed")

# ------------------------------------------------------------
# TAB 5 — TRAIN HISTORY
# ------------------------------------------------------------
with tab5:
    st.subheader("Train History")

    r = requests.get(f"{API}/admin/train_history")
    if r.ok:
        df = pd.DataFrame(r.json().get("train_history", []))
        st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# TAB 6 — TEST HISTORY
# ------------------------------------------------------------
with tab6:
    st.subheader("Test History")

    r = requests.get(f"{API}/admin/test_history")
    if r.ok:
        df = pd.DataFrame(r.json().get("test_history", []))
        if not df.empty and "metrics" in df.columns:
            metrics_df = pd.json_normalize(df["metrics"])
            df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
        st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# TAB 7 — FEEDBACK
# ------------------------------------------------------------
with tab7:
    st.subheader("User Feedback")

    r = requests.get(f"{API}/admin/feedback")
    if r.ok:
        feedback = r.json().get("feedback", [])

        if not feedback:
            st.info("No feedback submitted yet")
        else:
            for f in feedback:
                st.markdown(
                    f"""
                    ⭐ **Rating:** {f['rating']}/5  
                    🧑 **User:** {f['user_email']}  
                    🗂 **Workspace:** {f['workspace_name']}  
                    🕒 **Date:** {f['created_at']}  

                    💬 *{f['comment']}*
                    ---
                    """
                )


# ------------------------------------------------------------
# TAB 7 — LOGOUT
# ------------------------------------------------------------
with tab7:
    st.subheader("Logout from Admin")

    if st.button("Logout"):
        st.session_state.admin_logged_in = False
        st.rerun()
