from main import ChatBot
import streamlit as st


bot = ChatBot()


st.set_page_config(page_title="MRIIRS College Assistant Bot")


with st.sidebar:
    st.title('MRIIRS College Assistant Bot')
    st.markdown("""
        Welcome! This assistant can help you with any questions about your college life at MRIIRS, 
        including internships, exams, clubs, and academic policies.
    """)


def generate_response(input):
    result = bot.ask(input)  
    return result


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me anything about your college life at MRIIRS."}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if input := st.chat_input("Type your question here..."):
    
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Retrieving the answer from the college database..."):
                response = generate_response(input)
                st.write(response)
        
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
