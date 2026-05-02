from sentence_transformers import SentenceTransformer
import numpy as np

# -------- DATASET --------
documents = [
    "Persistent sadness and low energy may indicate depression.",
    "Excessive worry and restlessness are common symptoms of anxiety.",
    "Rapid heartbeat and sweating can occur during anxiety attacks.",
    "Difficulty sleeping can negatively impact mental health.",
    "Social withdrawal may be a sign of emotional distress.",
    "Mindfulness and deep breathing can help reduce stress levels.",
    "Regular physical activity improves mental well-being.",
    "Overthinking and constant fear may indicate anxiety disorder.",
    "Sudden mood swings can reflect emotional instability.",
    "Lack of motivation and fatigue are linked to depression.",
]

# -------- EMBEDDING MODEL --------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode once
doc_vectors = model.encode(documents, convert_to_numpy=True)

# Simulate large dataset
TOTAL_DOCS = 12000
VECTOR_DIM = doc_vectors.shape[1]


def retrieve(query, top_k):
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    scores = np.dot(doc_vectors, query_vec) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec)
    )

    #  Filter weak matches
    threshold = 0.3
    filtered_idx = [i for i, s in enumerate(scores) if s > threshold]

    if not filtered_idx:
        return [], []

    sorted_idx = sorted(filtered_idx, key=lambda i: scores[i], reverse=True)[:top_k]

    return [documents[i] for i in sorted_idx], scores[sorted_idx]

def is_mental_health_query(query):
    keywords = [
        "mental health", "stress", "anxiety", "depression",
        "emotions", "panic", "fear", "overthinking"
    ]

    q = query.lower()

    # quick keyword fallback
    if any(word in q for word in ["shivering", "heartbeat", "panic", "fear", "stress"]):
        return True

    query_vec = model.encode([query], convert_to_numpy=True)[0]
    keyword_vecs = model.encode(keywords, convert_to_numpy=True)

    scores = np.dot(keyword_vecs, query_vec) / (
        np.linalg.norm(keyword_vecs, axis=1) * np.linalg.norm(query_vec)
    )

    return max(scores) > 0.35

def detect_emotion(query):
    q = query.lower()

    if any(word in q for word in ["sad", "depressed", "hopeless", "tired"]):
        return "sadness"
    elif any(word in q for word in ["anxious", "panic", "nervous", "fear", "shivering"]):
        return "anxiety"
    elif any(word in q for word in ["angry", "frustrated", "irritated"]):
        return "anger"
    elif any(word in q for word in ["confused", "lost", "overthinking"]):
        return "confusion"
    elif any(word in q for word in ["shivering", "heartbeat", "sweating", "restless"]):
        return "anxiety"
    else:
        return "neutral"

import random

def extract_query_symptoms(query):
    symptom_map = {
        "shivering": "shivering",
        "sweating": "sweating",
        "heartbeat": "rapid heartbeat",
        "restless": "restlessness",
        "restlessness": "restlessness",
        "panic": "panic",
        "fear": "fear",
        "trembling": "trembling",
        "nervous": "nervousness",
        "overthinking": "overthinking",
    }

    q = query.lower()
    return [label for token, label in symptom_map.items() if token in q]


def build_context_summary(context):
    return " ".join(context).strip()

def generate_answer(query, context):
    q = query.lower()
    symptoms = extract_query_symptoms(query)
    context_summary = build_context_summary(context)

    if any(term in q for term in ["why", "cause", "causes", "reason", "reasons"]):
        intent = "cause"
    elif any(term in q for term in ["what should i do", "how do i", "how can i", "advice", "manage", "cope with", "handle", "deal with", "tips"]):
        intent = "advice"
    elif any(term in q for term in ["feel", "feeling", "experiencing", "signs", "symptoms", "symptom", "am i", "does this mean", "what does"]):
        intent = "symptom"
    else:
        intent = "general"

    empathy = {
        "anxiety": "I can hear how unsettling this feels. Anxiety and panic can be difficult to understand, and it's okay to ask for clarity.",
        "sadness": "It sounds like you are carrying something heavy right now. Thank you for sharing how you feel.",
        "anger": "Feeling frustrated or irritated is valid, and it helps to acknowledge that.",
        "confusion": "It makes sense to feel uncertain when your mind is trying to make sense of strong sensations.",
        "neutral": "Thank you for sharing that. I'm here to help you understand it clearly.",
    }.get(detect_emotion(query), "I’m sorry you’re going through this. I want to help you make sense of it.")

    answer_parts = [empathy]

    if intent == "cause":
        answer_parts.append(
            "Panic attacks and strong anxiety reactions often arise from the nervous system responding to stress, fear, or perceived threat."
        )
        answer_parts.append(
            "That response can be triggered by intense worry, sudden pressure, overthinking, or a situation that feels unsafe even when you are not in immediate danger."
        )
        if symptoms:
            answer_parts.append(
                f"Because you mentioned {', '.join(symptoms)}, it may help to know that those sensations are often part of the body’s alarm response."
            )
        answer_parts.append(
            "The causes are usually more about how the body interprets stress than a single symptom itself."
        )

    elif intent == "advice":
        answer_parts.append(
            "When you ask what to do, the most useful step is often to slow down and create a safer space for your body and mind."
        )
        answer_parts.append(
            "Try these practical steps:"
        )
        answer_parts.append(
            "- Pause and take a few slow breaths to interrupt the stress response."
        )
        answer_parts.append(
            "- Name what you are feeling in a calm way, such as anxiety, panic, or tension."
        )
        answer_parts.append(
            "- If it feels overwhelming, reach out to someone you trust or consider professional support."
        )

    elif intent == "symptom":
        if symptoms:
            answer_parts.append(
                f"Those sensations — {', '.join(symptoms)} — are often signals that the body is reacting to anxiety or panic."
            )
        else:
            answer_parts.append(
                "When people describe symptoms like these, it usually means the body is experiencing a strong stress response."
            )
        answer_parts.append(
            "That does not mean the experience is permanent, but it can be helpful to learn what the sensations are telling you."
        )
        answer_parts.append(
            "Paying attention to the pattern can make it easier to respond in a calmer way next time."
        )

    else:
        answer_parts.append(
            "This sounds like a question about how anxiety and panic show up for you."
        )
        answer_parts.append(
            "Often the same feelings come from a combination of stress, fear, and an overactive fight-or-flight response."
        )
        if symptoms:
            answer_parts.append(
                f"From your query, I see {', '.join(symptoms)} may be part of what you are noticing."
            )
        answer_parts.append(
            "Understanding that context can help make the response feel more manageable."
        )

    if context_summary:
        answer_parts.append("Here is some relevant information from the retrieved content:")
        answer_parts.append(context_summary)

    if intent != "advice":
        answer_parts.append(
            "Would you like me to share a few grounding techniques or explain this in a bit more detail?"
        )

    answer_parts.append("Note: This is general guidance and not a medical diagnosis.")

    return "\n\n".join(answer_parts)

def debug_rag(query, context, scores):
    issues = []

    max_score = max(scores)

    if len(scores) == 0:
        return ["No relevant context retrieved"]
    
    if max_score < 0.25:
        issues.append("Low retrieval relevance (weak semantic match)")
    elif max_score < 0.6:
        issues.append("Moderate retrieval quality (partial match)")
    else:
        issues.append("High retrieval quality (strong semantic alignment)")

    if len(context) < 2:
        issues.append("Insufficient context retrieved")

    if "dog" in query.lower() and not any("dog" in c.lower() for c in context):
        issues.append("Domain mismatch between query and context")

    return issues

def suggest_fixes(issues):
    fixes = []

    for issue in issues:
        if "Low retrieval" in issue or "Moderate" in issue:
            fixes.append("Improve embedding model quality or fine-tune for domain-specific data")
            fixes.append("Increase dataset diversity and coverage")

        if "Insufficient context" in issue:
            fixes.append("Retrieve more documents (increase top-k)")

        if "Domain mismatch" in issue:
            fixes.append("Apply domain-specific filtering or classification before retrieval")

    if not fixes:
        fixes.append("System performing well, minor improvements can be made with better embeddings")

    return fixes

def get_confidence(scores):
    return round(max(scores) * 100, 2)


import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="RAG Debugger",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Global */
body {
    background-color: #0b0f17;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* Title */
h1 {
    font-size: 34px;
    font-weight: 600;
    letter-spacing: -0.5px;
}

/* Section cards */
.card {
    background: #121826;
    border: 1px solid #1f2a44;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 16px;
    transition: 0.2s ease-in-out;
}

.card:hover {
    border-color: #2f81f7;
}

/* Metrics */
.metric-card {
    background: #121826;
    border: 1px solid #1f2a44;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
}

.metric-label {
    font-size: 13px;
    color: #8b949e;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #2f81f7;
    background-color: transparent;
    color: #2f81f7;
}

.stButton > button:hover {
    background-color: #2f81f7;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0e1422;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    padding: 10px 16px;
}

</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR --------
st.sidebar.title("Configuration")

top_k = st.sidebar.slider("Top-K Retrieval", 1, 4, 2)
show_scores = st.sidebar.toggle("Show Similarity Scores", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("RAG Debugger v1.0")

# -------- HEADER --------
st.title("Mental Health Assistant")
st.caption("AI-powered conversational system using Retrieval-Augmented Generation")


# -------- QUERY INPUT --------
query = st.chat_input("Ask about mental health...")

# -------- QUICK EXAMPLES --------
st.markdown("**Example Queries:**")
examples = [
    "Why do I feel anxious suddenly?",
    "How to handle overthinking?",
    "Signs of depression",
    "What causes panic attacks?"
]

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex):
        query=ex

# -------- MAIN OUTPUT --------
if query:

    # Save user message
    st.session_state.chat_history.append(("user", query))

    # -------- DOMAIN FILTER --------
    if not is_mental_health_query(query):
        answer = "This assistant is designed for mental health queries only."

    else:
        context, scores = retrieve(query, top_k)
        confidence = get_confidence(scores)

        if confidence < 20:
            answer = "I couldn't find relevant information. Please try a more specific mental health question."
        else:
            answer = generate_answer(query, context)

        issues = debug_rag(query, context, scores)
        fixes = suggest_fixes(issues)

    # Save bot response
    st.session_state.chat_history.append(("assistant", answer))

# -------- FINAL CHAT DISPLAY --------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# -------- DEBUG PANEL --------
if query and 'issues' in locals():
    with st.expander("System Analysis"):
        st.write("Issues Detected:")
        for i in issues:
            st.write("-", i)

        st.write("Suggested Fixes:")
        for f in fixes:
            st.write("-", f)