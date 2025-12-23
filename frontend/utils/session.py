import streamlit as st
def init_session():
    if "token" not in st.session_state:
        st.session_state["token"] = None
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "selected_workspace" not in st.session_state:
        st.session_state["selected_workspace"] = None
    if "dataset_uploaded" not in st.session_state:
        st.session_state["dataset_uploaded"] = False
    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = None
    if "dataset_id" not in st.session_state:
        st.session_state["dataset_id"] = None

init_session()
