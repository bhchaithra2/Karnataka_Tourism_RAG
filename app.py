
import streamlit as st
from pathlib import Path

from rag_engine import (
    find_pdf_files,
    load_vector_store,
    retrieve_context,
    generate_answer
)


# --------------------------------
# 1. PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Karnataka Tourism Chatbot",
    page_icon="🏛️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------
# 2. LOAD RAG SYSTEM
# --------------------------------

@st.cache_resource
def initialize_rag():

    index, chunks, model = load_vector_store(
        str(BASE_DIR)
    )

    return index, chunks, model


# --------------------------------
# 3. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("🏛️ Karnataka Tourism")

    st.caption("AI Tourism Information Assistant")

    st.divider()

    st.subheader("📌 About Project")

    st.write(
        """
        Karnataka Tourism Chatbot is an AI-based
        application that answers questions about
        Karnataka tourism using RAG technology.
        """
    )

    st.divider()

    st.subheader("💻 Technologies Used")

    st.write("🐍 Python")
    st.write("🔍 Retrieval Augmented Generation")
    st.write("🤖 Large Language Model")
    st.write("🧠 Sentence Transformers")
    st.write("🗃️ FAISS Vector Database")
    st.write("🌐 Streamlit")

    st.divider()

    st.subheader("🔄 RAG Pipeline")

    st.write("1. Load PDF Files")
    st.write("2. Create Text Chunks")
    st.write("3. Generate Embeddings")
    st.write("4. Store in FAISS Database")
    st.write("5. Retrieve Relevant Information")
    st.write("6. Generate Answer using LLM")

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.write("🏕️ Adventure Tourism")
    st.write("🏛️ Heritage, History and Culture")
    st.write("🍛 Food and Cuisine")
    st.write("🐅 Wildlife and Safari")

    st.divider()

     st.subheader("👥 Our Team Members")

     team_members = [
        "Chaithra B H",
        "Jyothi H G",
        "Aishwarya",
        "Madhav U",
     ]

     for number, name in enumerate(team_members, start=1):
         st.write(f"**{number}.** {name}")


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
# 5. CHECK PDF FILES
# --------------------------------

pdf_files = find_pdf_files(str(BASE_DIR))

if pdf_files:

    st.sidebar.success(
        f"{len(pdf_files)} PDF files found"
    )

else:

    st.warning(
        "No PDF files found. Please upload your four "
        "PDF files inside the karnataka_docs folder."
    )

    st.stop()


# --------------------------------
# 6. GET GROQ API KEY
# --------------------------------

try:

    api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    api_key = None


if not api_key:

    st.error(
        "Groq API key is missing. Please add it in "
        "Streamlit Settings → Secrets."
    )

    st.stop()


# --------------------------------
# 7. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    avatar = "🧭" if message["role"] == "user" else "🏛️"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])


# --------------------------------
# 8. CHAT INPUT
# --------------------------------

question = st.chat_input(
    "Ask anything about Karnataka tourism..."
)


# --------------------------------
# 9. GENERATE ANSWER
# --------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user", avatar="🧭"):

        st.markdown(question)

    with st.chat_message("assistant", avatar="🏛️"):

        try:

            with st.spinner(
                "🔍 Searching Karnataka tourism information..."
            ):

                index, chunks, model = initialize_rag()

                results = retrieve_context(
                    question,
                    index,
                    chunks,
                    model
                )

                answer = generate_answer(
                    question,
                    results,
                    api_key
                )

            st.markdown(answer)

            st.markdown("### 📚 Sources")

            sources = set()

            for item in results:

                sources.add(
                    f'{item["source"]} - Page {item["page"]}'
                )

            for source in sources:

                st.write("📄", source)

            full_response = answer + "\n\n**Sources:**\n"

            for source in sources:

                full_response += f"\n- {source}"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )

        except Exception as error:

            st.error(f"Something went wrong: {error}")


# --------------------------------
# 10. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Karnataka Tourism AI | Powered by Python, RAG and LLM"
)
