import streamlit as st
# CHANGE: Import query_documents instead of get_chat_response
from app import query_documents 

st.title("RAG Chatbot")

# Init state & display history
for msg in st.session_state.setdefault("messages", []):
    st.chat_message(msg["role"]).write(msg["content"])

# Input & Response
if prompt := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        # CHANGE: Call query_documents to perform the RAG search
        response = query_documents(prompt)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)