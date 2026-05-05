import streamlit as st
import pandas as pd
import re
import random
import numpy as np
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

st.set_page_config(
    page_title="Mental Health Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ========== SESSION STATE INITIALIZATION ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "severity" not in st.session_state:
    st.session_state.severity = 0  # 0=normal, 1=moderate, 2=high, 3=critical
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "neutral"
if "repeated_distress" not in st.session_state:
    st.session_state.repeated_distress = 0
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "techniques_used" not in st.session_state:
    st.session_state.techniques_used = []
if "escalation_count" not in st.session_state:
    st.session_state.escalation_count = 0
if "status" not in st.session_state:
    st.session_state.status = "Listening"

# ========== DATASET LOADING ==========
df = pd.read_csv("Mental_Health_FAQ.csv")
documents = (df["Questions"] + " " + df["Answers"]).tolist()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

documents = [clean_text(doc) for doc in documents]

# ========== TEXT NORMALIZATION & CRITICAL DETECTION ==========
def normalize_query_text(query):
    q = query.lower()
    q = re.sub(r"[''`´]", "'", q)
    q = re.sub(r"[^a-z0-9'\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\b(can't|cannot|cant)\b", "can't", q)
    q = re.sub(r"([a-z])\1{2,}", r"\1", q)
    return q

CRITICAL_PANIC_PHRASES = [
    "can't breathe", "cannot breathe", "cant breathe", "breathing problem", "no air",
    "chest pain", "dying", "losing control", "out of control", "help me",
    "can't stand", "cannot stand", "cant stand", "falling", "dizzy", "faint",
    "collapsing", "pass out", "panic attack", "hyperventilate", "suffocating"
]

ESCALATION_INDICATORS = [
    "it's increasing", "its increasing", "not stopping", "getting worse", "much worse",
    "can't handle", "cant handle", "breaking down", "i can't cope", "cant cope",
    "making it worse", "not helping", "still panicking"
]

# ========== STAGE-BASED DETECTION ==========
def detect_severity(query, history_messages):
    """
    Returns severity level:
    0=normal, 1=moderate, 2=high, 3=critical
    """
    q = normalize_query_text(query)

    # CRITICAL - immediate danger signals
    if any(phrase in q for phrase in CRITICAL_PANIC_PHRASES):
        return 3

    # Check for worsening/escalation
    if any(indicator in q for indicator in ESCALATION_INDICATORS):
        return 3

    # HIGH - significant distress
    if any(word in q for word in ["panicking", "shaking", "sweating heavily", "hyperventilating", "overwhelming", "breaking down"]):
        return 2

    # Pattern: previous panic + new concern = escalation
    if len(history_messages) >= 2:
        recent = " ".join([msg.get("content", "").lower() for msg in history_messages[-2:] if msg.get("role") == "user"])
        if ("panic" in recent or "breathe" in recent) and ("worse" in q or "still" in q or "continue" in q or "more" in q):
            return 3

    # MODERATE - moderate concern
    if any(word in q for word in ["panic", "anxious", "scared", "worried", "overwhelmed", "stressed"]):
        return 1

    # NORMAL
    return 0

def get_severity_label(severity):
    """Map severity to label"""
    labels = {0: "normal", 1: "moderate", 2: "high", 3: "critical"}
    return labels.get(severity, "normal")

def detect_emotion(query):
    q = query.lower()
    
    if any(word in q for word in ["panic", "shivering", "trembling", "rapid heartbeat", "hyperventilate", "scared", "fear", "terrified"]):
        return "panic"
    if any(word in q for word in ["stressed", "stress", "overwhelmed", "pressured", "burned out", "confused", "scattered"]):
        return "stress"
    if any(word in q for word in ["sad", "depressed", "hopeless", "tired", "down", "lonely", "empty"]):
        return "sadness"
    if any(word in q for word in ["anxious", "nervous", "worry", "restless", "uneasy", "tense"]):
        return "anxiety"
    return "neutral"

def get_intent(query, history_context):
    q = query.lower()
    
    # Safety FIRST
    if any(term in q for term in ["suicidal", "self harm", "self-harm", "hurt myself", "can't cope", "unsafe", "end it"]):
        return "safety"
    
    # Action-oriented
    if any(term in q for term in ["what should i do", "help me", "suggest", "what can i do", "how do i", "how should"]):
        return "action"
    
    # Understanding
    if any(term in q for term in ["why", "what is", "what's", "cause", "reason", "explain", "what happens"]):
        return "explanation"
    
    # Emotional support
    if any(term in q for term in ["sad", "upset", "overwhelmed", "panic", "scared", "lonely", "anxious"]):
        return "support"
    
    # Simple engagement
    if history_context and q in {"yeah", "ok", "okay", "sure", "fine", "maybe", "idk"}:
        return "engagement"
    
    return "support"

def is_mental_health_query(query):
    q = normalize_query_text(query)

    # Strong signals (ANY → True)
    strong_signals = [
        "can't", "cant", "breathe", "panic", "scared", "afraid",
        "alone", "lonely", "crying", "help", "stress", "anxiety",
        "depressed", "overwhelmed", "not feeling good",
        "shaking", "dizzy", "heartbeat", "fear"
    ]

    if any(word in q for word in strong_signals):
        return True

    # Short emotional phrases
    if len(q.split()) <= 5:
        if any(word in q for word in ["bad", "low", "sad", "tired", "lost"]):
            return True

    # Follow-up context
    if len(st.session_state.chat_history) > 0:
        return True

    return False

# ========== EMBEDDING & RETRIEVAL ==========
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        st.error("Embedding model failed to load")
        return None

model = load_model()
if model is not None:
    doc_vectors = model.encode(documents, convert_to_numpy=True)
else:
    doc_vectors = None

if api_key:
    groq_client = Groq(api_key=api_key)
else:
    groq_client = None

def retrieve(query, top_k):
    if model is None or doc_vectors is None:
        return [], []
    
    query_vec = model.encode([query], convert_to_numpy=True)[0]
    scores = np.dot(doc_vectors, query_vec) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec)
    )
    
    threshold = 0.4
    filtered_idx = [i for i, s in enumerate(scores) if s > threshold]
    
    if not filtered_idx:
        return [], []
    
    sorted_idx = sorted(filtered_idx, key=lambda i: scores[i], reverse=True)[:top_k]
    return [documents[i] for i in sorted_idx], scores[sorted_idx]

def handle_small_talk(query):
    q = query.lower().strip()

    if q in ["hi", "hello", "hey"]:
        return "Hey, I'm here with you. How are you feeling today?"

    if "how are you" in q:
        return "I'm here and listening. How are you feeling right now?"

    return None

# ========== RESPONSE COMPOSITION ==========
def get_presence_phrase():
    return random.choice([
        "I'm here with you.",
        "Let's take this step by step.",
        "You're not alone in this.",
        "I'm listening."
    ])

def get_followup_question(kind="general"):
    options = {
        "general": [
            "What's on your mind right now?",
            "How are you feeling about that?",
            "What feels most pressing?"
        ],
        "grounding": [
            "What do you notice in your body?",
            "Can you name one thing you see?",
            "What's one sound you hear around you?"
        ],
        "choice": [
            "Would you like to try something small?",
            "What would help most right now?"
        ]
    }
    return random.choice(options.get(kind, options["general"]))

ACK_VARIATIONS = [
    "That sounds really tough.",
    "I can hear how heavy this feels.",
    "That’s a lot to deal with.",
    "I understand why that would feel overwhelming."
]

INTERPRET_VARIATIONS = [
    "It seems like this is really affecting you right now.",
    "Your mind and body are reacting strongly to this.",
    "This situation feels intense for you.",
    "You're going through something difficult."
]
def compose_response(ack, interpret, step=None, ask_question=False):
    """
    Compose response with exact structure:
    1. Acknowledge emotion
    2. Interpret gently
    3. ONE helpful step
    4. Optional ONE soft question
    """
    lines = []
    
    lines.append(ack)
    lines.append(interpret)
    
    if step:
        lines.append(step)
    
    if ask_question and random.random() > 0.6:
        question = get_followup_question()
        lines.append(question)
    
    return "\n\n".join(lines)

# ========== ESCALATION & CRISIS RESPONSE ==========
def get_crisis_response(query):
    """Immediate response for critical panic/safety"""
    q = normalize_query_text(query)
    
    response = "I can hear that you're in a really difficult moment right now. You're not alone, and I'm here with you.\n\n"
    
    # Physical grounding for breathing issues
    if any(word in q for word in ["breathe", "breathing", "suffocating", "air"]):
        response += "Try this right now:\n"
        response += "- Sit or lie down if you can\n"
        response += "- Breathe in slowly through your nose for 4 counts\n"
        response += "- Hold for 2 counts\n"
        response += "- Breathe out slowly through your mouth for 6 counts\n"
        response += "- Do this 3 times with me\n\n"
    else:
        response += "Take one moment to ground yourself:\n"
        response += "- If you can, sit down\n"
        response += "- Feel your feet on the floor\n"
        response += "- Take one slow breath with me\n\n"
    
    response += "This moment will pass. If symptoms don't improve or are getting worse, please contact a doctor or emergency service immediately."
    
    return response

def get_emergency_escalation():
    """Strong alert for repeating critical signals"""
    return (
        "⚠️ This is serious and I want to make sure you get proper help.\n\n"
        "Please contact:\n"
        "- Call emergency services\n"
        "- Go to nearest emergency room\n"
        "- Call National Crisis Hotline: 112\n"
        "- Tell a trusted person nearby what's happening\n\n"
        "You don't have to handle this alone. Real people are trained to help."
    )

# ========== LLM RESPONSE GENERATION ==========
def generate_response(query, context, intent, emotion, severity):
    """
    Main response generator with specific structure:
    1. Acknowledge emotion
    2. Interpret gently
    3. ONE helpful step
    4. Optional ONE soft question
    """
    # PANIC MODE STAGES
    if severity >= 1:  # Any panic detected
        if severity == 1:  # Stage 1: initial panic
            ack = random.choice([
                "I can hear how anxious you're feeling right now.",
                "That sounds really overwhelming.",
                "I sense you're in a lot of distress."
            ])
            interpret = "This is your body's natural response to stress."
            step = "Let's try a simple breathing exercise: breathe in slowly for 4 counts, hold for 4, breathe out for 4."
            return compose_response(ack, interpret, step, ask_question=False)

        elif severity == 2:  # Stage 2: worsening
            ack = random.choice([
                "I hear how intense this is getting.",
                "This feels like it's building strongly.",
                "You're experiencing significant panic right now."
            ])
            interpret = "Your body is in fight-or-flight mode, which can feel terrifying."
            step = "Stop what you're doing if you can. Place both feet flat on the floor and focus only on your breathing."
            return compose_response(ack, interpret, step, ask_question=False)

        elif severity == 3:  # Stage 3: critical
            ack = "This is a medical emergency."
            interpret = "You need immediate professional help."
            step = "Call emergency services (911) right now or go to the nearest emergency room."
            return compose_response(ack, interpret, step, ask_question=False)

    # NORMAL CONVERSATION
    if intent == "safety":
        return (
            "I'm really glad you shared this with me.\n"
            "You deserve support and you're not alone.\n\n"
            "Please reach out:\n"
            "- Call 988 (US Crisis Line)\n"
            "- Text HOME to 741741 (Crisis Text Line)\n"
            "- Contact a trusted person nearby"
        )

    if intent == "action":
        ack = random.choice([
            "I understand you want practical help.",
            "That makes sense to look for something actionable.",
            "Let's focus on what you can do right now."
        ])
        interpret = "Taking small steps can help shift how you're feeling."
        step = random.choice([
            "Try one slow breath right now. Notice how your body responds.",
            "If you can, pause for a moment and take three deep breaths.",
            "Try naming one small thing that feels a bit easier."
        ])
        return compose_response(ack, interpret, step, ask_question=False)

    if intent == "explanation":
        ack = "That's a really good question."
        interpret = "Understanding what's happening can help reduce the fear."
        context_detail = (context[0][:150] if context else "Sometimes these feelings come from the body's stress response. That's completely normal.")
        step = context_detail
        return compose_response(ack, interpret, step, ask_question=True)

    if intent == "support":
        ack = random.choice([
            "That sounds hard to carry right now.",
            "I can tell this is weighing on you.",
            "This feels like a lot."
        ])
        interpret = "It's completely valid to feel this way."
        step = random.choice([
            "Try placing one hand on your chest and take a slow breath.",
            "Focus on one thing you can see around you.",
            "You don’t have to solve everything right now."
        ])
        return compose_response(ack, interpret, step, ask_question=random.choice([True, False]))

    if intent == "engagement":
        ack = "I'm here with you."
        interpret = "Let's keep the conversation going."
        step = random.choice([
            "Try placing your hand on your chest and take a slow breath.",
            "Focus on one thing around you and describe it.",
            "You don’t need to solve everything right now."
        ])
        return compose_response(ack, interpret, step, ask_question=False)

    # Fallback
    ack = "I'm here with you."
    interpret = "Tell me what you're experiencing right now."
    step = "How can I support you?"
    return compose_response(ack, interpret, step, ask_question=False)

def generate_llm_response(query, context, intent, emotion, severity):
    """
    Generate LLM response with fallback logic.
    Uses Groq API if available.
    """
    if severity >= 3:
        return get_crisis_response(query)

    # Track repeated distress
    if "repeated_distress" not in st.session_state:
        st.session_state.repeated_distress = 0
    
    if severity >= 2:
        st.session_state.repeated_distress += 1
    else:
        st.session_state.repeated_distress = 0
    # Force escalation if user keeps struggling
    if st.session_state.repeated_distress >= 2:
        severity = max(severity, 3)

    # Update session state
    st.session_state.severity = severity

    if emotion in {"panic", "anxiety", "stress"}:
        st.session_state.last_emotion = emotion

    # 🚨 CRITICAL ESCALATION FIRST
    if severity == 3:
        st.session_state.escalation_count += 1
        if st.session_state.escalation_count >= 2:
            return get_emergency_escalation()   # STRONG message
        else:
            return get_crisis_response(query)   # First warning
    # ⚠️ PANIC STAGES
    if severity >= 1:
        return generate_response(query, context, intent, emotion, severity)

    # For normal cases, try LLM if available
    if groq_client is None:
        return generate_response(query, context, intent, emotion, severity)
    
    context_summary = context[:300] if context else ""
    system_prompt = f"""
You are a calm, supportive mental health companion.

Your role:
- Act like a caring human, not a bot
- Be emotionally intelligent and natural
- Avoid repeating phrases

Behavior rules:
- Keep responses SHORT (2–3 sentences max)
- Do NOT always ask questions
- Sometimes just support
- Match emotional intensity

Critical handling:
- If user shows panic (breathing issue, fear, shaking):
    → Guide breathing step-by-step
- If user says it's getting worse AGAIN:
    → Strongly tell them to seek medical help

Never say:
- "everything will be fine"

Context:
{context_summary}
"""
    
    try:
        message = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=200
        )
        response = message.choices[0].message.content.strip()
        
        if len(response) > 400:
            response = response[:400].rsplit(".", 1)[0] + "."
        
        return response if response else generate_response(query, context, intent, emotion, severity)
    
    except Exception as e:
        return generate_response(query, context, intent, emotion, severity)

# ========== MODERN UI STYLING ==========
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    padding: 0 !important;
}

.stChatMessage {
    padding: 16px !important;
}

/* Chat messages styling */
.chat-message-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
    animation: fadeIn 0.5s ease-in;
}

.chat-message-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 16px;
    animation: fadeIn 0.5s ease-in;
}

.message-bubble-user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    padding: 14px 18px;
    border-radius: 18px;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    line-height: 1.5;
}

.message-bubble-bot {
    background: #1e293b;
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 18px;
    max-width: 70%;
    word-wrap: break-word;
    border: 1px solid #334155;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    line-height: 1.5;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Breathing guide */
.breathing-guide {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
    margin: 20px 0;
}

.breathing-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, #60a5fa, #3b82f6);
    animation: breathe 4s ease-in-out infinite;
    box-shadow: 0 0 30px rgba(96, 165, 250, 0.5);
}

@keyframes breathe {
    0%, 100% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.2); opacity: 1; }
}

/* Status indicator */
.status-indicator {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #1e293b;
    color: #60a5fa;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid #334155;
    z-index: 1000;
}

.header-section {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    margin-bottom: 20px;
}

.header-title {
    font-size: 32px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 24px;
}

.header-tagline {
    color: #60a5fa;
    font-size: 16px;
    font-weight: 500;
}

.quick-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    padding: 0 20px 24px;
}

.quick-btn {
    padding: 10px 16px !important;
    border-radius: 20px !important;
    border: 2px solid #334155 !important;
    background: transparent !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    cursor: pointer;
    transition: all 0.3s;
}

.quick-btn:hover {
    border-color: #60a5fa !important;
    color: #60a5fa !important;
    background: rgba(96, 165, 250, 0.1) !important;
}

.chat-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
}

.input-container {
    position: relative;
    padding: 10px 0;
    background: transparent;
}

.stChatInputContainer {
    width: 100% !important;
    padding: 0 20px !important;
    box-sizing: border-box;
}

.stChatInput {
    border-radius: 20px !important;
    border: 2px solid #334155 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

.stChatInput input {
    font-size: 15px !important;
}

.alert-banner {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 16px;
    border-left: 4px solid #fca5a5;
}

.info-badge {
    display: inline-block;
    background: #334155;
    color: #94a3b8;
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 8px;
}

.confidence-high {
    color: #4ade80 !important;
}

.confidence-medium {
    color: #facc15 !important;
}

.confidence-low {
    color: #ef4444 !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #1e293b;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
<style>
[data-testid="stChatInput"] textarea {
    border-radius: 20px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* Fix send button */
button[kind="secondary"] {
    z-index: 9999 !important;
}

/* Remove overlay blocking */
.block-container {
    position: relative;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)

# ========== UI LAYOUT ==========

# Header
st.markdown("""
<div class="header-section">
    <div class="header-title">Mental Health Companion</div>
    <div class="header-subtitle">A calm, supportive conversation</div>
    <div class="header-tagline">You're not alone. I'm here with you.</div>
</div>
""", unsafe_allow_html=True)

# Alert if panic escalating
if st.session_state.severity >= 2:
    st.markdown(f"""
    <div class="alert-banner">
        I notice you're in distress. I'm right here with you. 
        {"If this doesn't improve, please reach out to emergency services." if st.session_state.severity == 3 else ""}
    </div>
    """, unsafe_allow_html=True)

# Breathing guide for panic
if st.session_state.severity >= 1:
    st.markdown("""
    <div class="breathing-guide">
        <div class="breathing-circle"></div>
    </div>
    <p style="text-align: center; color: #94a3b8; font-size: 14px;">Breathe with the circle - in as it expands, out as it contracts</p>
    """, unsafe_allow_html=True)

# Status indicator
status_text = "Responding..." if st.session_state.status == "Responding" else "Listening"
st.markdown(f"""
<div class="status-indicator">
    {status_text}
</div>
""", unsafe_allow_html=True)

# Quick action buttons
st.markdown("""
<div class="quick-buttons">
""", unsafe_allow_html=True)

cols = st.columns(4)
quick_options = [
    "I feel anxious",
    "I can't sleep",
    "I feel overwhelmed",
    "Help with panic"
]

query = None
for i, (col, opt) in enumerate(zip(cols, quick_options)):
    with col:
        if st.button(opt, key=f"quick_{i}", use_container_width=True):
            query = opt

st.markdown("</div>", unsafe_allow_html=True)

# Main chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Chat input
st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
query = st.chat_input("Tell me what's on your mind...", key="main_input")

if query:
    st.session_state.status = "Responding"
    query = query.strip()
    small_talk = handle_small_talk(query)
    if small_talk:
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": small_talk})
        st.session_state.status = "Listening"
        st.rerun()
        st.stop()
    # Process query
    history_context = " ".join([msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"][-3:])
    intent = get_intent(query, history_context)
    emotion = detect_emotion(query)
    severity = detect_severity(query, st.session_state.chat_history)

    # Always process as meaningful input
    context, scores = retrieve(query, 3)
    
    confidence = np.mean(scores) * 100 if scores is not None and len(scores) > 0 else 0
    confidence_label = "High" if confidence > 70 else "Medium" if confidence > 40 else "Low"
    
    # Detect severity again (important)
    severity = detect_severity(query, st.session_state.chat_history)
    
    # Generate response using your main logic
    answer = generate_response(query, context, intent, emotion, severity)
    
    small_talk = handle_small_talk(query)
    if small_talk:
        answer = small_talk
        context, scores = [], []
        confidence_label = "Low"
    else:
        context, scores = retrieve(query, 3)
        confidence = np.mean(scores) * 100 if scores is not None and len(scores) > 0 else 0
        confidence_label = "High" if confidence > 70 else "Medium" if confidence > 40 else "Low"
        severity = detect_severity(query, st.session_state.chat_history)
        # PRIORITY: Always handle serious cases with your logic
        if severity >= 1:
            answer = generate_response(query, context, intent, emotion, severity)
        else:
            answer = generate_llm_response(query, context, intent, emotion,severity)
    
    # Add to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    st.session_state.status = "Listening"
    st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-message-user">
            <div class="message-bubble-user">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message-bot">
            <div class="message-bubble-bot">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar debug info
with st.sidebar:
    st.title("System Analysis")
    
    if st.session_state.chat_history:
        last_query = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        if last_query:
            emotion = detect_emotion(last_query)
            intent = get_intent(last_query, "")
            severity = detect_severity(last_query, st.session_state.chat_history)
            
            st.markdown("**Session Status**")
            st.write(f"Messages: {len(st.session_state.chat_history)}")
            st.write(f"Severity: {get_severity_label(st.session_state.severity)}")
            
            st.markdown("**Last Analysis**")
            st.write(f"Emotion: {emotion.title()}")
            st.write(f"Intent: {intent.title()}")
            
            severity_colors = {
                0: "🟢 Normal",
                1: "🟡 Moderate",
                2: "🟠 High",
                3: "🔴 Critical"
            }
            st.write(f"Level: {severity_colors.get(st.session_state.severity, '?')}")
