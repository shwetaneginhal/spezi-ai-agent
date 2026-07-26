import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="Spezi - Your German Local Friend", page_icon="🥤")

st.title("🥤 Spezi is cold, carbonated, and live!")
st.subheader("Chat with your German Local buddy")

if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Please enter your Username here for personalized chats")
    user_id_input = st.text_input("Username (e.g., Fanta):", value=st.session_state.user_id)
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Say something in English or German..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Spezi is thinking..."):
            try:
                payload = {
                    "user_id": st.session_state.user_id,
                    "message": user_input
                }
                res = requests.post(API_URL, json=payload, timeout=60)
                
                if res.status_code == 200:
                    spezi_reply = res.json()["response"]
                else:
                    spezi_reply = f"Error from backend: {res.status_code} - {res.text}"
                    
            except Exception as e:
                spezi_reply = f"Could not connect to API server: {e}"

            st.markdown(spezi_reply)
            st.session_state.messages.append({"role": "assistant", "content": spezi_reply})
