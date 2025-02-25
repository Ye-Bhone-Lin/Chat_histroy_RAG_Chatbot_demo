import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid

with open("rag_components.pkl", "rb") as f:
    loaded_data = pickle.load(f)

gemini_api_key = st.secrets["general"]["GEMINI_API_KEY"]

retriever = loaded_data["retriever"]
contextualize_q_prompt = loaded_data['contextualize_q_prompt']
qa_prompt = loaded_data["qa_prompt"]
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",google_api_key=gemini_api_key,temperature=0.2,max_tokens=None)

history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(model,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

store = {}
# Unique session handling
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ChatMessageHistory()

# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state["chat_history"]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_keys = "input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user_input = st.chat_input()
session_id = st.session_state["session_id"]

if user_input:
    if user_input.lower() == "restart":
        st.session_state["chat_history"] = ChatMessageHistory()
        st.write("Chat history cleared!")
    else:
        answer = conversational_rag_chain.invoke(
            {'input': user_input},
            config={"configurable": {"session_id": session_id}}
        )['answer']
        st.write(answer)
        