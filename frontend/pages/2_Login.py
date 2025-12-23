import streamlit as st
import requests
from utils.session import init_session

API = "http://127.0.0.1:8000"

st.title("Login / Signup")

tab = st.tabs(["Login", "Signup"])

with tab[0]:
    email = st.text_input("Email", key="login_email")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login"):
        resp = requests.post(f"{API}/auth/login", json={"email": email, "password": pwd})
        if resp and resp.status_code == 200:
            data = resp.json()
            st.session_state["email"] = data.get("email")
            st.success("Logged in")
        else:
            st.error("Login failed")

with tab[1]:
    semail = st.text_input("Signup Email", key="su_email")
    spwd = st.text_input("Signup Password", type="password", key="su_pwd")
    if st.button("Signup"):
        resp = requests.post(f"{API}/auth/signup", json={"email": semail, "password": spwd})
        if resp and resp.status_code in (200,201):
            st.success("User created, please login")
        else:
            st.error("Signup failed")
