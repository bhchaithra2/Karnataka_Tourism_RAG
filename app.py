import streamlit as st
import os

from dotenv import load_dotenv

from rag_engine import (
    load_vector_store,
    retrieve_context,
    generate_answer
)

# Load environment variables
load_dotenv()


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Karnataka Tourism Chatbot",
    page_icon="🏛️",
    layout="wide"
)


# --------------------------------
# 2. LOAD RAG DATABASE
# --------------------------------

@st.cache_resource
def initialize_rag():

    index, chunks, model = load_vector_store(
        "karnataka_docs"
    )

    return index, chunks, model


# --------------------------------
# 3. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("🏛️ Karnataka Tourism")

    st.caption("AI Tourism Information Assistant")

    st.divider()

    # PROJECT INFORMATION

    st.subheader("📌 About Project")

    st.write(
        """
        Karnataka Tourism Chatbot is an AI-based
        application that answers questions about
        Karnataka tourism using RAG technology.
        """
    )

    st.divider()

    # TECHNOLOGIES

    st.subheader("💻 Technologies Used")

    st.write(
        """
        🐍 Python

        🔍 Retrieval Augmented Generation (RAG)

        🤖 Large Language Model (LLM)

        🧠 Sentence Transformers

        🗃️ FAISS Vector Database

        🌐 Streamlit
        """
    )

    st.divider()

    # RAG PIPELINE

    st.subheader("🔄 RAG Pipeline")

    st.write(
        """
        📄 1. Load PDF Files

        ✂️ 2. Create Text Chunks

        🧠 3. Generate Embeddings

        🗃️ 4. Store in Vector Database

        🔍 5. Retrieve Relevant Information

        🤖 6. Generate Answer
        """
    )

    st.divider()

    # karnataka_docs

    st.subheader("📚 Knowledge Base")

    st.write(
        """
        📄 Adventure Tourism

        📄 Heritage, History and Culture

        📄 Food and Cuisine

        📄 Wildlife and Safari
        """
    )

    st.divider()

    # TEAM MEMBERS

    st.subheader("👥 Team Members")

    st.write(
        """
        1. Jyothi H G

        2. Aishwarya

        3. Chaithra.B.H

        4. Madhav
        """
    )


# --------------------------------
# 4. MAIN PAGE
# --------------------------------

st.title("🏛️ Karnataka Tourism Chatbot")

st.subheader(
    "Explore Karnataka with Artificial Intelligence!"
)

st.caption(
    "Discover Adventure | Heritage | Culture | Food | Wildlife"
)

st.write(
    "Ask anything about Karnataka and explore "
    "its beautiful destinations, history and traditions."
)

st.divider()


# --------------------------------
# 5. API KEY
# --------------------------------

try:

    api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    api_key = os.getenv("GROQ_API_KEY")


# --------------------------------
# 6. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# DISPLAY OLD MESSAGES

for message in st.session_state.messages:

    if message["role"] == "user":

        avatar = "🧭"

    else:

        avatar = "🏛️"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])


# --------------------------------
# 7. CHAT INPUT
# --------------------------------

question = st.chat_input(
    "Ask anything about Karnataka tourism..."
)


# --------------------------------
# 8. GENERATE RESPONSE
# --------------------------------

if question:

    # Display user question

    with st.chat_message("user", avatar="🧭"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # AI RESPONSE

    with st.chat_message("assistant", avatar="🏛️"):

        if not api_key:

            st.error(
                "Please configure your Groq API key."
            )

        elif not os.path.exists("karnataka_docs") or not any(
            name.lower().endswith(".pdf")
            for name in os.listdir("karnataka_docs")
        ):

            st.warning(
                "Please add your PDF files inside Data_files folder."
            )

        else:

            try:

                with st.spinner(
                    "🔍 Searching Karnataka information..."
                ):

                    # Load database

                    index, chunks, model = initialize_rag()

                    # Retrieve information

                    results = retrieve_context(
                        question,
                        index,
                        chunks,
                        model
                    )

                    # Generate answer

                    answer = generate_answer(
                        question,
                        results,
                        api_key
                    )

                # Display answer

                st.markdown(answer)

                # Display sources

                st.markdown("### 📚 Sources")

                sources = set()

                for item in results:

                    sources.add(
                        item["source"]
                        + " - Page "
                        + str(item["page"])
                    )

                for source in sources:

                    st.write("📄", source)

                # Save assistant message

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(
                    f"Error occurred: {error}"
                )


# --------------------------------
# 9. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Karnataka Tourism AI | Powered by Python, RAG and LLM"
)
