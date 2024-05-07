import wget
import streamlit as st
import threading
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def bar_custom(current, total, width=80):
    print(f"Downloading {current / total * 100:.2f}% [{current}/{total}] bytes")

@st.cache(show_spinner=False)
def download_model(model_url):
    wget.download(model_url, bar=bar_custom)

def init_page() -> None:
    st.set_page_config(page_title="Personal Chatbot")
    st.header("Personal Chatbot")
    st.sidebar.title("Options")

def select_llm() -> LlamaCPP:
    return LlamaCPP(
        model_path="llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=500,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant. Reply in markdown format.")
        ]

def get_answer(llm, messages) -> str:
    response = llm.complete(messages)
    return response.text

def main() -> None:
    init_page()

    # Download the model asynchronously
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
    threading.Thread(target=download_model, args=(model_url,), daemon=True).start()

    # Continue with the rest of the app
    llm = select_llm()
    init_messages()

    user_input = st.text_input("Input your question!")

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):
            answer = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            st.markdown(message.content)

if __name__ == "__main__":
    main()
