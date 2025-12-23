import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("Workspaces")
email = st.session_state.get("email")
if not email:
    st.warning("Please login first")
    st.stop()

# create new
st.subheader("Create Workspace")
name = st.text_input("Workspace name", key="ws_name")
if st.button("Create Workspace"):
    resp = requests.post(f"{API}/workspace/create", json={"name": name, "owner": email})
    if resp and resp.status_code == 200:
        st.success("Created")
    else:
        st.error("Create failed")

# list
st.subheader("Your Workspaces")
resp = requests.get(f"{API}/workspace/list/{email}")
if resp and resp.status_code == 200:
    wss = resp.json()
    for ws in wss:
        col1, col2 = st.columns([8,2])
        with col1:
            st.write(ws.get("name"))
        with col2:
            if st.button("Open", key=f"open_{ws.get('id')}"):
                st.session_state["selected_workspace"] = {"id": ws.get("id"), "name": ws.get("name")}
                st.success("Opened workspace")
else:
    st.info("No workspaces")
