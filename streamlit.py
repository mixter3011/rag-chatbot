import streamlit as st
from main import ChatBot
import base64

bot = ChatBot()

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.set_page_config(page_title="MRIIRS EduBot", layout="wide")


add_bg_from_local('assets/bg1.jpeg')  


st.markdown("""
    <style>
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    .css-1d391kg .sidebar-content {
        background-color: transparent !important;
    }
    [data-testid=stSidebar] [data-testid=stMarkdownContainer] p {
        color: white !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.5) !important;
        padding: 20px !important;
        border-radius: 10px !important;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* General text color */
    .stApp, .stTextInput, .stMarkdown, .stChatMessage {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Chat message container */
    .stChatMessage {
        background-color: rgba(0, 0, 0, 0.1) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title('MRIIRS EduBot')
    st.markdown("""
        Welcome! This assistant can help you with any questions about your college life at MRIIRS, 
        including internships, exams, clubs, and academic policies.
    """)


st.title("MRIIRS EduBot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about MRIIRS."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = bot.ask(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})