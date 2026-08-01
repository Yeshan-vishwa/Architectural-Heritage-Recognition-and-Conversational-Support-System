import hashlib
import io
import os
import re
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Heritage Architecture Assistant",
    page_icon="🏛️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 10px;
        padding: 5px 9px;
        min-height: 82px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.76rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div[data-testid="stImage"] {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏛️ Heritage Architecture Assistant")

st.write(
    "Upload an architectural heritage image for classification, "
    "or use the chat to ask general heritage architecture questions."
)


# LOAD COMPUTER VISION MODEL

@st.cache_resource
def load_cv_model():

    return load_model(
        "best_heritage_model.keras",
        compile=False
    )


try:

    model = load_cv_model()

except Exception as error:

    st.error(
        "The trained model could not be loaded. Make sure "
        "'best_heritage_model.keras' is in the same folder as app.py."
    )

    st.exception(error)
    st.stop()


# LOAD CLASS NAMES

train_folder = Path("train")

if not train_folder.exists():

    st.error(
        "The 'train' folder was not found. It must be in the "
        "same project folder as app.py."
    )

    st.stop()


class_names = sorted(
    [
        folder.name
        for folder in train_folder.iterdir()
        if folder.is_dir()
    ]
)


if len(class_names) != model.output_shape[-1]:

    st.error(
        "The number of class folders does not match "
        "the number of outputs in the trained model."
    )

    st.stop()


IMAGE_HEIGHT = model.input_shape[1]
IMAGE_WIDTH = model.input_shape[2]


# CREATE RAG SYSTEM

@st.cache_resource
def create_rag_system():

    knowledge_folder = Path(
        "heritage_documents_detailed"
    )

    if not knowledge_folder.exists():

        raise FileNotFoundError(
            "The 'heritage_documents_detailed' folder was not found."
        )


    knowledge_documents = []
    document_lookup = {}


    for file_path in sorted(
        knowledge_folder.glob("*.txt")
    ):

        document_text = file_path.read_text(
            encoding="utf-8"
        )

        document = Document(
            page_content=document_text,
            metadata={
                "source": file_path.name
            }
        )

        knowledge_documents.append(
            document
        )

        document_lookup[
            file_path.name
        ] = document


    if len(knowledge_documents) < 5:

        raise ValueError(
            "At least five heritage documents are required."
        )


    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_database = InMemoryVectorStore(
        embedding=embeddings
    )

    vector_database.add_documents(
        documents=knowledge_documents
    )

    retriever = vector_database.as_retriever(
        search_kwargs={
            "k": 1
        }
    )

    llm = OllamaLLM(
        model="llama3",
        temperature=0.2
    )

    return (
        retriever,
        llm,
        document_lookup
    )


try:

    retriever, llm, document_lookup = create_rag_system()

except Exception as error:

    st.error(
        "The RAG system could not be created. Make sure Ollama is "
        "running and that 'llama3' and 'nomic-embed-text' are installed."
    )

    st.exception(error)
    st.stop()


# CLASS AND DOCUMENT MAPPINGS

CLASS_TO_SOURCE = {
    "altar": "altar.txt",
    "apse": "apse.txt",
    "bell_tower": "bell_tower.txt",
    "column": "column.txt",
    "dome(inner)": "dome_inner.txt",
    "dome(outer)": "dome_outer.txt",
    "flying_buttress": "flying_buttress.txt",
    "gargoyle": "gargoyle.txt",
    "stained_glass": "stained_glass.txt",
    "vault": "vault.txt"
}


TOPIC_ALIASES = {
    "altar.txt": [
        "altar",
        "altars"
    ],

    "apse.txt": [
        "apse",
        "apses"
    ],

    "bell_tower.txt": [
        "bell tower",
        "bell towers",
        "belfry",
        "belfries"
    ],

    "column.txt": [
        "column",
        "columns",
        "pillar",
        "pillars"
    ],

    "dome_inner.txt": [
        "dome inner",
        "dome(inner)",
        "inner dome",
        "interior dome",
        "inside dome",
        "dome interior",
        "internal dome"
    ],

    "dome_outer.txt": [
        "dome outer",
        "dome(outer)",
        "outer dome",
        "exterior dome",
        "outside dome",
        "dome exterior",
        "external dome"
    ],

    "flying_buttress.txt": [
        "flying buttress",
        "flying buttresses"
    ],

    "gargoyle.txt": [
        "gargoyle",
        "gargoyles",
        "grotesque",
        "grotesques"
    ],

    "stained_glass.txt": [
        "stained glass",
        "stained-glass",
        "coloured glass",
        "colored glass"
    ],

    "vault.txt": [
        "vault",
        "vaults",
        "vaulted ceiling",
        "vaulted ceilings",
        "rib vault",
        "rib vaults",
        "barrel vault",
        "barrel vaults",
        "groin vault",
        "groin vaults"
    ]
}


# TEXT FORMATTING

def format_class_name(class_name):

    display_name = (
        class_name
        .replace("_", " ")
        .replace("(", " ")
        .replace(")", "")
        .title()
    )

    return " ".join(
        display_name.split()
    )


def format_source_name(source):

    if not source:
        return ""

    return (
        source
        .replace(".txt", "")
        .replace("_", " ")
        .title()
    )


def clean_llm_answer(answer):

    answer = answer.strip()

    unwanted_openings = [
        r"^according to the supplied knowledge,\s*",
        r"^according to the provided knowledge,\s*",
        r"^according to the knowledge provided,\s*",
        r"^according to the reference information,\s*",
        r"^based on the supplied knowledge,\s*",
        r"^based on the provided knowledge,\s*",
        r"^based on the reference information,\s*",
        r"^according to the context,\s*",
        r"^based on the context,\s*",
        r"^the supplied knowledge states that\s*",
        r"^the provided knowledge states that\s*",
        r"^the reference information states that\s*",
        r"^here is a friendly and knowledgeable explanation[^:]*:\s*",
        r"^here is a friendly explanation[^:]*:\s*",
        r"^here is a simple explanation[^:]*:\s*",
        r"^here's a friendly and knowledgeable explanation[^:]*:\s*",
        r"^here's a friendly explanation[^:]*:\s*",
        r"^here's a simple explanation[^:]*:\s*",
        r"^here’s a friendly explanation[^:]*:\s*",
        r"^here’s a simple explanation[^:]*:\s*",
        r"^welcome to this fascinating architectural feature!\s*",
        r"^welcome!\s*"
    ]

    for pattern in unwanted_openings:

        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE
        )


    unwanted_phrases = [
        "According to the supplied knowledge, ",
        "According to the provided knowledge, ",
        "According to the knowledge provided, ",
        "According to the reference information, ",
        "Based on the supplied knowledge, ",
        "Based on the provided knowledge, ",
        "Based on the reference information, ",
        "According to the context, ",
        "Based on the context, "
    ]

    for phrase in unwanted_phrases:

        answer = answer.replace(
            phrase,
            ""
        )


    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )


    if answer:

        answer = (
            answer[0].upper()
            + answer[1:]
        )

    return answer.strip()


# TOPIC AND MEMORY HELPERS

def detect_explicit_topic(question):

    question_lower = question.lower()

    for source, aliases in TOPIC_ALIASES.items():

        for alias in aliases:

            if alias in question_lower:
                return source

    return None


def get_previous_source(conversation):

    for message in reversed(
        conversation
    ):

        if (
            message["role"] == "assistant"
            and message.get("source")
        ):

            return message["source"]

    return None


def build_conversation_text(conversation):

    conversation_lines = []

    for message in conversation[-10:]:

        if (
            message.get("message_type")
            == "image_explanation"
        ):

            continue

        conversation_lines.append(
            message["role"].title()
            + ": "
            + message["content"]
        )

    return "\n".join(
        conversation_lines
    )


# IMAGE PREDICTION

def predict_uploaded_image(image_bytes):

    uploaded_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    resized_image = uploaded_image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    prediction_scores = model.predict(
        image_batch,
        verbose=0
    )

    predicted_index = int(
        np.argmax(
            prediction_scores[0]
        )
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        np.max(
            prediction_scores[0]
        )
    )

    prediction_data = {
        "predicted_class": predicted_class,

        "display_name": format_class_name(
            predicted_class
        ),

        "predicted_index": predicted_index,

        "confidence": confidence,

        "confidence_percent": round(
            confidence * 100,
            2
        )
    }

    return (
        uploaded_image,
        prediction_data
    )


# RETRIEVAL HELPERS

def retrieve_knowledge(query):

    retrieved_documents = retriever.invoke(
        query
    )

    if not retrieved_documents:

        raise ValueError(
            "No relevant heritage document was retrieved."
        )

    return retrieved_documents


def get_prediction_document(predicted_class):

    source = CLASS_TO_SOURCE.get(
        predicted_class
    )

    if (
        source is not None
        and source in document_lookup
    ):

        return document_lookup[
            source
        ]


    retrieved_documents = retrieve_knowledge(
        format_class_name(
            predicted_class
        )
    )

    return retrieved_documents[0]


def get_general_question_document(
    question,
    conversation
):

    explicit_source = detect_explicit_topic(
        question
    )

    previous_source = get_previous_source(
        conversation
    )


    if (
        explicit_source is not None
        and explicit_source in document_lookup
    ):

        return document_lookup[
            explicit_source
        ]


    if (
        previous_source is not None
        and previous_source in document_lookup
    ):

        return document_lookup[
            previous_source
        ]


    retrieved_documents = retrieve_knowledge(
        question
    )

    return retrieved_documents[0]


# IMAGE EXPLANATION

def explain_prediction(
    predicted_class,
    confidence
):

    readable_class = format_class_name(
        predicted_class
    )

    document = get_prediction_document(
        predicted_class
    )

    context = document.page_content
    source = document.metadata["source"]


    prompt = f"""
You are a friendly heritage architecture guide.

A computer vision model predicts that the uploaded image contains:

Architectural element: {readable_class}
Prediction confidence: {confidence:.2%}

Write one natural paragraph of approximately 90 to 120 words.

Include:
- what the architectural element is
- its main purpose
- its most important visual features
- brief historical or architectural context

Rules:
- Use only the reference information.
- Do not invent facts.
- Treat the classification as a model prediction, not absolute certainty.
- Do not mention the knowledge base, document, context, prompt,
  reference information or requested word count.
- Do not begin with "According to", "Based on", "Welcome",
  "Here is", "Here's" or "This fascinating feature".
- Begin directly with the name of the architectural element.
- Do not add an introductory sentence before the explanation.

Reference information:
{context}

Explanation:
"""


    explanation = llm.invoke(
        prompt
    )

    explanation = clean_llm_answer(
        explanation
    )

    return (
        explanation,
        source
    )


# GENERAL CHAT

def answer_general_question(
    question,
    conversation
):

    document = get_general_question_document(
        question,
        conversation
    )

    context = document.page_content
    source = document.metadata["source"]

    previous_conversation = build_conversation_text(
        conversation
    )


    prompt = f"""
You are a friendly heritage architecture guide.

Answer the visitor's current question naturally.

Conversation rules:
- Use the conversation history to understand follow-up expressions such as
  "it", "its", "this", "that", "these", "those", "they" and "them".
- Continue discussing the current architectural element unless the visitor
  clearly introduces a different architectural element.
- Use the previous messages only to understand the topic and references.

Answer rules:
- Use only the reference information for factual claims.
- Do not invent facts.
- Answer the question directly.
- Keep the response clear, concise and student-friendly.
- Do not repeat the question.
- Do not mention the knowledge base, source document, context,
  reference information or prompt.
- Do not begin with "According to", "Based on", "Here is",
  "Here's" or "Welcome".
- If the requested information is absent, say:
  "I couldn't find that information in the current heritage knowledge."

Conversation history:
{previous_conversation}

Reference information:
{context}

Current question:
{question}

Answer:
"""


    answer = llm.invoke(
        prompt
    )

    answer = clean_llm_answer(
        answer
    )

    return (
        answer,
        source
    )


# IMAGE-AWARE CHAT

def answer_image_question(
    predicted_class,
    question,
    conversation
):

    readable_class = format_class_name(
        predicted_class
    )

    document = get_prediction_document(
        predicted_class
    )

    context = document.page_content
    source = document.metadata["source"]

    previous_conversation = build_conversation_text(
        conversation
    )


    prompt = f"""
You are a friendly heritage architecture guide.

The uploaded image is currently being discussed.

The computer vision model predicts that the image shows:

{readable_class}

Answer the visitor's current question naturally.

Conversation rules:
- Understand follow-up expressions such as "it", "its", "this",
  "that", "these", "those", "they" and "them" as references to
  {readable_class}, unless the conversation clearly indicates otherwise.
- Use previous messages to understand the visitor's follow-up questions.
- Keep the conversation focused on {readable_class}.

Answer rules:
- Use only the reference information for factual claims.
- Do not invent facts.
- Answer the question directly.
- Keep the response clear, concise and student-friendly.
- Do not repeat the question.
- Do not mention the knowledge base, source document, context,
  reference information or prompt.
- Do not begin with "According to", "Based on", "Here is",
  "Here's" or "Welcome".
- Do not claim that the image classification is certainly correct.
- If the requested information is absent, say:
  "I couldn't find that information in the current heritage knowledge."

Conversation history:
{previous_conversation}

Reference information:
{context}

Current question:
{question}

Answer:
"""


    answer = llm.invoke(
        prompt
    )

    answer = clean_llm_answer(
        answer
    )

    return (
        answer,
        source
    )


# SESSION STATE

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "explanation" not in st.session_state:
    st.session_state.explanation = None

if "explanation_source" not in st.session_state:
    st.session_state.explanation_source = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "pending_conversation" not in st.session_state:
    st.session_state.pending_conversation = []


# MAIN LAYOUT

upload_column, chat_column = st.columns(
    [0.9, 1.1],
    gap="large"
)


# LEFT COLUMN: IMAGE ANALYSIS

with upload_column:

    st.subheader(
        "Image Analysis"
    )

    st.caption(
        "Upload an image to classify a heritage element. "
        "Image upload is optional."
    )


    uploaded_file = st.file_uploader(
        "Upload a heritage architecture image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        new_image_id = hashlib.md5(
            image_bytes
        ).hexdigest()


        if st.session_state.image_id != new_image_id:

            st.session_state.image_id = new_image_id
            st.session_state.prediction_data = None
            st.session_state.uploaded_image = None
            st.session_state.explanation = None
            st.session_state.explanation_source = None
            st.session_state.chat_messages = []
            st.session_state.pending_question = None
            st.session_state.pending_conversation = []


        if st.session_state.prediction_data is None:

            try:

                uploaded_image, prediction_data = predict_uploaded_image(
                    image_bytes
                )

                st.session_state.uploaded_image = (
                    uploaded_image
                )

                st.session_state.prediction_data = (
                    prediction_data
                )

            except Exception as error:

                st.error(
                    "The uploaded file could not be processed as an image."
                )

                st.exception(error)


        if st.session_state.prediction_data is not None:

            prediction_data = (
                st.session_state.prediction_data
            )


            st.subheader(
                "Prediction"
            )


            result_column, confidence_column = st.columns(
                [1.1, 1]
            )


            with result_column:

                st.metric(
                    label="Architectural element",
                    value=prediction_data[
                        "display_name"
                    ]
                )


            with confidence_column:

                st.metric(
                    label="Confidence",
                    value=(
                        f'{prediction_data["confidence"]:.2%}'
                    )
                )


            if prediction_data["confidence"] < 0.50:

                st.warning(
                    "The model has low confidence in this prediction. "
                    "Interpret the result cautiously."
                )

            elif prediction_data["confidence"] < 0.70:

                st.info(
                    "The model has moderate confidence in this prediction."
                )

            else:

                st.success(
                    "The model has high confidence in this prediction."
                )


            if st.session_state.uploaded_image is not None:

                image_column, image_spacing_column = st.columns(
                    [5, 1]
                )

                with image_column:

                    st.image(
                        st.session_state.uploaded_image,
                        caption="Uploaded image",
                        width=440
                    )


            if st.session_state.explanation is None:

                try:

                    with st.spinner(
                        "Generating image explanation..."
                    ):

                        explanation, source = explain_prediction(
                            prediction_data[
                                "predicted_class"
                            ],

                            prediction_data[
                                "confidence"
                            ]
                        )


                    st.session_state.explanation = (
                        explanation
                    )

                    st.session_state.explanation_source = (
                        source
                    )


                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",

                            "content": explanation,

                            "source": source,

                            "message_type": "image_explanation",

                            "title": prediction_data[
                                "display_name"
                            ]
                        }
                    )


                except Exception as error:

                    st.error(
                        "The explanation could not be generated. "
                        "Check that Ollama is running."
                    )

                    st.exception(error)


    else:

        if st.session_state.image_id is not None:

            st.session_state.image_id = None
            st.session_state.prediction_data = None
            st.session_state.uploaded_image = None
            st.session_state.explanation = None
            st.session_state.explanation_source = None
            st.session_state.chat_messages = []
            st.session_state.pending_question = None
            st.session_state.pending_conversation = []


        st.info(
            "No image is currently uploaded. "
            "You can still ask general questions in the chat."
        )


# RIGHT COLUMN: CHAT ASSISTANT

with chat_column:

    st.subheader(
        "Heritage Chat Assistant"
    )


    if st.session_state.prediction_data is not None:

        st.success(
            "🖼️ Image context: "
            + st.session_state.prediction_data[
                "display_name"
            ]
        )

        st.caption(
            "Ask about its history, purpose, materials, "
            "visual features, location or conservation."
        )

    else:

        st.info(
            "💬 General heritage chat"
        )

        st.caption(
            "Ask general questions about altars, apses, bell towers, "
            "columns, domes, flying buttresses, gargoyles, "
            "stained glass or vaults."
        )


    control_column, spacer_column = st.columns(
        [1, 4]
    )


    with control_column:

        if st.button(
            "Clear chat",
            key="clear_chat_button"
        ):

            st.session_state.chat_messages = []
            st.session_state.pending_question = None
            st.session_state.pending_conversation = []


            if (
                st.session_state.explanation is not None
                and st.session_state.prediction_data is not None
            ):

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",

                        "content": (
                            st.session_state.explanation
                        ),

                        "source": (
                            st.session_state.explanation_source
                        ),

                        "message_type": "image_explanation",

                        "title": (
                            st.session_state.prediction_data[
                                "display_name"
                            ]
                        )
                    }
                )

            st.rerun()


    # SCROLLABLE CHAT HISTORY

    chat_history_container = st.container(
        height=500,
        border=True
    )


    with chat_history_container:

        # CONVERSATION STARTERS

        if len(st.session_state.chat_messages) == 0:

            st.markdown(
                "### 💡 Conversation starters"
            )

            st.caption(
                "Choose a suggested question or type your own below."
            )

            starter_column_1, starter_column_2 = st.columns(2)


            with starter_column_1:

                if st.button(
                    "What is a gargoyle?",
                    key="starter_gargoyle",
                    width="stretch"
                ):

                    starter_question = (
                        "What is a gargoyle?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


                if st.button(
                    "Where are bell towers commonly found?",
                    key="starter_bell_tower",
                    width="stretch"
                ):

                    starter_question = (
                        "Where are bell towers commonly found?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


                if st.button(
                    "What is stained glass?",
                    key="starter_stained_glass",
                    width="stretch"
                ):

                    starter_question = (
                        "What is stained glass?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


            with starter_column_2:

                if st.button(
                    "What is an apse?",
                    key="starter_apse",
                    width="stretch"
                ):

                    starter_question = (
                        "What is an apse?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


                if st.button(
                    "What is an interior dome?",
                    key="starter_inner_dome",
                    width="stretch"
                ):

                    starter_question = (
                        "What is an interior dome?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


                if st.button(
                    "What are flying buttresses?",
                    key="starter_flying_buttress",
                    width="stretch"
                ):

                    starter_question = (
                        "What are flying buttresses?"
                    )

                    st.session_state.pending_conversation = []

                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": starter_question
                        }
                    )

                    st.session_state.pending_question = (
                        starter_question
                    )

                    st.rerun()


        # DISPLAY SAVED MESSAGES

        for message in st.session_state.chat_messages:

            with st.chat_message(
                message["role"]
            ):

                if (
                    message.get("message_type")
                    == "image_explanation"
                ):

                    st.markdown(
                        "#### Image analysis summary"
                    )

                    if message.get("title"):

                        st.markdown(
                            "**Predicted element: "
                            + message["title"]
                            + "**"
                        )

                    st.write(
                        message["content"]
                    )

                else:

                    st.write(
                        message["content"]
                    )


                if message.get("source"):

                    st.caption(
                        "Source: "
                        + format_source_name(
                            message["source"]
                        )
                    )


        # PROCESS PENDING QUESTION

        if st.session_state.pending_question is not None:

            pending_question = (
                st.session_state.pending_question
            )

            previous_messages = list(
                st.session_state.pending_conversation
            )


            with st.chat_message(
                "assistant"
            ):

                try:

                    with st.spinner(
                        "Thinking..."
                    ):

                        if (
                            st.session_state.prediction_data
                            is not None
                        ):

                            answer, source = answer_image_question(
                                st.session_state.prediction_data[
                                    "predicted_class"
                                ],

                                pending_question,

                                previous_messages
                            )

                        else:

                            answer, source = answer_general_question(
                                pending_question,

                                previous_messages
                            )


                    st.write(
                        answer
                    )

                    st.caption(
                        "Source: "
                        + format_source_name(
                            source
                        )
                    )


                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",

                            "content": answer,

                            "source": source
                        }
                    )


                except Exception:

                    error_message = (
                        "I could not generate an answer. "
                        "Please check that Ollama is running."
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",

                            "content": error_message
                        }
                    )


            st.session_state.pending_question = None
            st.session_state.pending_conversation = []

            st.rerun()


    # CHAT INPUT BELOW THE SCROLLABLE BOX

    user_question = st.chat_input(
        "Ask a heritage architecture question...",
        key="heritage_chat_input"
    )


    if user_question:

        cleaned_question = (
            user_question.strip()
        )

        if cleaned_question:

            st.session_state.pending_conversation = list(
                st.session_state.chat_messages
            )

            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": cleaned_question
                }
            )

            st.session_state.pending_question = (
                cleaned_question
            )

            st.rerun()


# FOOTER

st.divider()

st.caption(
    "Computer Vision: EfficientNetB0  |  "
    "LLM: Llama 3 through Ollama  |  "
    "RAG: LangChain with an in-memory vector store"
)