import streamlit as st
import streamlit.components.v1 as components
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
    st.session_state.severity = 0  # 0=normal, 1=mild, 2=elevated, 3=high, 4=panic, 5=critical
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
if "support_stage" not in st.session_state:
    st.session_state.support_stage = "initial"
if "breathing_attempts" not in st.session_state:
    st.session_state.breathing_attempts = 0
if "last_bot_response" not in st.session_state:
    st.session_state.last_bot_response = ""
if "panic_detected" not in st.session_state:
    st.session_state.panic_detected = False
if "panic_level" not in st.session_state:
    st.session_state.panic_level = 0
if "reassurance_count" not in st.session_state:
    st.session_state.reassurance_count = 0
if "show_breathing_popup" not in st.session_state:
    st.session_state.show_breathing_popup = False
if "breathing_popup_active" not in st.session_state:
    st.session_state.breathing_popup_active = False
if "breathing_popup_started_at" not in st.session_state:
    st.session_state.breathing_popup_started_at = 0.0
if "breathing_popup_expires_at" not in st.session_state:
    st.session_state.breathing_popup_expires_at = 0.0
if "breathing_popup_id" not in st.session_state:
    st.session_state.breathing_popup_id = ""
if "severity_history" not in st.session_state:
    st.session_state.severity_history = []
if "symptom_count" not in st.session_state:
    st.session_state.symptom_count = 0
if "physical_symptoms" not in st.session_state:
    st.session_state.physical_symptoms = []
if "escalation_trend" not in st.session_state:
    st.session_state.escalation_trend = "stable"
if "risk_level" not in st.session_state:
    st.session_state.risk_level = "low"
if "detected_symptoms" not in st.session_state:
    st.session_state.detected_symptoms = []
if "emotional_intensity" not in st.session_state:
    st.session_state.emotional_intensity = 0
if "support_strategy_history" not in st.session_state:
    st.session_state.support_strategy_history = []
if "panic_episode_turns" not in st.session_state:
    st.session_state.panic_episode_turns = 0
if "last_severity_analysis" not in st.session_state:
    st.session_state.last_severity_analysis = {}
if "last_response_strategy" not in st.session_state:
    st.session_state.last_response_strategy = "general"
if "last_breathing_popup_at" not in st.session_state:
    st.session_state.last_breathing_popup_at = 0.0
if "emotional_timeline" not in st.session_state:
    st.session_state.emotional_timeline = []
if "recovery_streak" not in st.session_state:
    st.session_state.recovery_streak = 0
if "support_flow_stage" not in st.session_state:
    st.session_state.support_flow_stage = 1
if "calming_success_count" not in st.session_state:
    st.session_state.calming_success_count = 0
if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.time()

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

REASSURANCE_SEEKING = [
    "will i be okay",
    "am i dying",
    "what if something happens",
    "is this dangerous",
    "will something happen to me",
    "am i safe",
    "is this serious"
]

RECOVERY_PHRASES = [
    "i feel better",
    "i'm calmer",
    "im calmer",
    "feeling better",
    "more relaxed",
    "less anxious",
    "i'm okay now",
    "im okay now",
    "better now",
    "calmer now"
]

SILENT_DISTRESS = [
    "idk",
    "i don't know",
    "dont know",
    "nothing",
    "leave it",
    "forget it",
    "never mind",
    "nvm",
    "it's nothing",
    "its nothing"
]

SELF_HARM_KEYWORDS = [
    "i want to die",
    "kill myself",
    "suicide",
    "self harm",
    "hurt myself",
    "end my life",
    "don't want to live",
    "i can't go on",
    "want to disappear",
    "better off dead"
]

# ========== BREATHING POPUP DETECTION ==========
BREATHING_TRIGGER_PHRASES = [
    "breathe",
    "breathing",
    "inhale",
    "exhale",
    "deep breath",
    "calm your breathing"
]

def detect_breathing_trigger(response_text):
    """
    Detect if bot response contains breathing-related guidance.
    Returns True if breathing popup should be shown.
    """
    response_lower = response_text.lower()
    # Normalize punctuation
    response_normalized = re.sub(r"[^\w\s]", " ", response_lower)
    response_normalized = re.sub(r"\s+", " ", response_normalized).strip()
    
    for phrase in BREATHING_TRIGGER_PHRASES:
        if phrase in response_normalized:
            return True
    return False

def start_breathing_popup():
    """Start one temporary breathing popup instance."""
    now = time.time()
    popup_id = f"breathing-{int(now * 1000)}-{random.randint(1000, 9999)}"
    st.session_state.show_breathing_popup = True
    st.session_state.breathing_popup_active = True
    st.session_state.breathing_popup_started_at = now
    st.session_state.breathing_popup_expires_at = now + 14
    st.session_state.breathing_popup_id = popup_id

def close_expired_breathing_popup():
    """Clear popup state after its server-side lifetime has passed."""
    expires_at = st.session_state.get("breathing_popup_expires_at", 0.0)
    if st.session_state.get("show_breathing_popup") and expires_at and time.time() >= expires_at:
        st.session_state.show_breathing_popup = False
        st.session_state.breathing_popup_active = False
        st.session_state.breathing_popup_started_at = 0.0
        st.session_state.breathing_popup_expires_at = 0.0

# ========== SIMPLIFIED SEVERITY FUNCTION (pure analysis wrapper) ==========
def detect_severity(query, history_messages):
    """
    Pure wrapper that returns severity level only.
    Returns severity level: 0=normal, 1=mild, 2=elevated, 3=high, 4=panic, 5=critical
    """
    return analyze_severity(query, history_messages).get("severity", 0)

def get_severity_label(severity):
    """Map severity to label"""
    labels = {
        0: "normal",
        1: "mild distress",
        2: "elevated anxiety",
        3: "high distress",
        4: "panic state",
        5: "critical support"
    }
    return labels.get(severity, "normal")

def get_support_flow_stage(severity, analysis=None):
    """Map current emotional state to a 7-stage support flow."""
    analysis = analysis or {}
    strategy = st.session_state.get("last_response_strategy", "general")
    if severity >= 5:
        return 7
    if strategy == "trusted_help" or severity == 4:
        return 6
    if strategy == "stabilization":
        return 5
    if strategy == "grounding" or severity == 3:
        return 4
    if strategy == "breathing" or analysis.get("physical_symptoms"):
        return 3
    if severity >= 1:
        return 2
    return 1

def get_support_stage_label(stage):
    labels = {
        1: "Normal conversation",
        2: "Emotional support",
        3: "Breathing assistance",
        4: "Grounding assistance",
        5: "Panic stabilization",
        6: "Trusted-person support",
        7: "Emergency escalation"
    }
    return labels.get(stage, "Normal conversation")

def get_stability_score():
    """Higher score means calmer recent pattern."""
    history = st.session_state.get("severity_history", [])[-6:]
    if not history:
        return 100
    average = sum(history) / len(history)
    volatility = sum(abs(history[i] - history[i - 1]) for i in range(1, len(history)))
    score = 100 - (average * 14) - (volatility * 5)
    return max(0, min(100, int(score)))

def get_trend_indicator():
    trend = st.session_state.get("escalation_trend", "stable")
    return {
        "escalating": ("Rising", "#f97316"),
        "improving": ("Easing", "#22c55e"),
        "fluctuating": ("Variable", "#eab308"),
        "stable": ("Stable", "#60a5fa")
    }.get(trend, ("Stable", "#60a5fa"))

def format_timeline_summary():
    timeline = st.session_state.get("emotional_timeline", [])[-6:]
    if not timeline:
        return "No emotional timeline yet."
    return " -> ".join(f"{item['severity']}:{item['emotion'][:6]}" for item in timeline)

def record_emotional_timeline(emotion, analysis):
    """Store a compact emotional telemetry point for dashboard continuity."""
    item = {
        "time": time.time(),
        "emotion": emotion,
        "severity": analysis.get("severity", 0),
        "intensity": analysis.get("emotional_intensity", 0),
        "stage": st.session_state.get("support_flow_stage", 1),
    }
    timeline = st.session_state.get("emotional_timeline", [])
    timeline.append(item)
    st.session_state.emotional_timeline = timeline[-12:]

# ========== ADVANCED SEVERITY DETECTION WITH WEIGHTED SCORING ==========
def analyze_severity(query, history_messages, severity_history=None):
    """
    Pure severity analysis with:
    - Weighted keyword scoring
    - Multi-message escalation
    - Symptom combination detection
    - Context-aware false positive reduction
    - No session_state mutation
    
    Returns a dict with severity, trend, risk, score, and symptoms.
    """
    q = normalize_query_text(query)
    score = 0
    detected_symptoms = []
    physical_symptoms_found = []
    emotional_signals_found = []
    context_flags = []
    history_messages = history_messages or []
    severity_history = list(severity_history or [])
    recent_user_messages = [
        normalize_query_text(msg.get("content", ""))
        for msg in history_messages[-8:]
        if msg.get("role") == "user"
    ]
    previous_user_messages = recent_user_messages[:-1] if recent_user_messages and recent_user_messages[-1] == q else recent_user_messages
    recent_context = " ".join(previous_user_messages[-4:])
    recent_panic_context = any(
        term in recent_context
        for term in ["panic", "breathe", "breathing", "shiver", "shake", "tremble", "chest", "dizzy", "suffocat", "can't calm", "cant calm"]
    )
    
    # ===== WEIGHT-BASED KEYWORD SCORING =====
    
    # Critical/Emergency Keywords
    critical_keywords = {
        "suicide": 10, "self harm": 10, "kill myself": 11, "end my life": 11,
        "i want to die": 11, "don't want to live": 10, "better off dead": 10,
        "can't breathe": 9, "cannot breathe": 9, "cant breathe": 9,
        "no air": 9, "suffocating": 9, "chest pain": 8, "losing consciousness": 9,
        "fainting": 8, "pass out": 8, "severe pain": 7
    }
    
    for keyword, weight in critical_keywords.items():
        if keyword in q:
            score += weight
            detected_symptoms.append(keyword)
            if keyword not in ["suicide", "self harm", "kill myself", "end my life", "i want to die", "don't want to live", "better off dead"]:
                physical_symptoms_found.append(keyword)
    
    # High Panic Keywords
    panic_keywords = {
        "hyperventilate": 4, "panic attack": 5, "panicking": 4,
        "panic": 4, "shaking": 4, "shivering": 4, "trembling": 4, "rapid heartbeat": 3,
        "dizzy": 3, "dizziness": 3, "can't stand": 4, "cant stand": 4,
        "overwhelming": 2, "breaking down": 3, "losing control": 3,
        "out of control": 3, "can't calm down": 3, "cant calm down": 3,
        "sweating heavily": 3, "heart racing": 3, "tight chest": 4,
        "chest tight": 4, "chest tightness": 4
    }
    
    for keyword, weight in panic_keywords.items():
        if keyword in q:
            score += weight
            detected_symptoms.append(keyword)
            if keyword in ["shaking", "shivering", "trembling", "rapid heartbeat", "dizzy", "dizziness", "can't stand", "cant stand", "sweating heavily", "heart racing", "tight chest", "chest tight", "chest tightness", "hyperventilate"]:
                physical_symptoms_found.append(keyword)
    
    # Moderate Anxiety Keywords
    anxiety_keywords = {
        "anxious": 1, "anxiety": 1, "scared": 1, "afraid": 1,
        "nervous": 1, "worried": 1, "overwhelmed": 1, "stressed": 1,
        "stressed out": 2, "stress": 1, "tense": 1, "uneasy": 1,
        "restless": 1, "hopeless": 3, "helpless": 2
    }
    
    for keyword, weight in anxiety_keywords.items():
        if keyword in q:
            score += weight
            detected_symptoms.append(keyword)
            emotional_signals_found.append(keyword)
    
    # ===== ESCALATION INDICATORS =====
    escalation_phrases = [
        "getting worse", "it's worse", "its worse", "much worse",
        "still panicking", "not stopping", "can't handle it", "cant handle it",
        "making it worse", "not helping", "it's increasing", "its increasing",
        "worse than before", "keeps happening", "won't stop",
        "still can't", "still cant", "i still can't", "i still cant", "i can't", "i cant"
    ]
    
    for phrase in escalation_phrases:
        if phrase in q:
            score += 4 if "still" in phrase else 3
            detected_symptoms.append(f"escalation: {phrase}")
    
    # ===== MULTI-MESSAGE CONTEXT ANALYSIS =====
    if len(previous_user_messages) >= 1:
        breathing_mentions = sum(1 for msg in previous_user_messages if "breathe" in msg or "breathing" in msg)
        chest_mentions = sum(1 for msg in previous_user_messages if "chest" in msg or "heart" in msg)
        dizzy_mentions = sum(1 for msg in previous_user_messages if "dizzy" in msg or "faint" in msg or "dizz" in msg)
        panic_mentions = sum(1 for msg in previous_user_messages if any(term in msg for term in ["panic", "shiver", "shake", "tremble", "suffocat", "can't calm", "cant calm"]))
        
        continuation_phrases = ["i can't", "i cant", "still", "again", "same", "not working", "no use", "worse", "help"]
        is_distress_continuation = any(phrase in q for phrase in continuation_phrases)

        if breathing_mentions >= 1 and (("breathe" in q or "breathing" in q) or is_distress_continuation):
            score += 3
            detected_symptoms.append("repeated: breathing distress")
            physical_symptoms_found.append("repeated breathing distress")
        if chest_mentions >= 1 and (("chest" in q or "heart" in q) or is_distress_continuation):
            score += 2
            detected_symptoms.append("repeated: chest symptoms")
            physical_symptoms_found.append("repeated chest symptoms")
        if dizzy_mentions >= 1 and (("dizzy" in q or "faint" in q or "dizz" in q) or is_distress_continuation):
            score += 2
            detected_symptoms.append("repeated: dizziness")
            physical_symptoms_found.append("repeated dizziness")
        if panic_mentions >= 1 and is_distress_continuation:
            score += 2
            detected_symptoms.append("continued panic distress")
        
        if "worse" in q and any(symptom in " ".join(previous_user_messages[-3:]) for symptom in ["panic", "breathe", "chest", "dizzy", "heart"]):
            score += 3
            detected_symptoms.append("progressive: worsening trend")
    
    # ===== SYMPTOM COMBINATION MULTIPLIER =====
    # Multiple physical symptoms = higher urgency
    physical_symptoms = ["breathe", "chest", "dizzy", "heart", "faint", "shake", "shiver", "tremble", "sweat", "suffocat"]
    physical_count = sum(1 for symptom in physical_symptoms if symptom in q)
    
    if physical_count >= 3:
        score += 2  # Multiple physical symptoms
        detected_symptoms.append("multi-symptom: physical crisis")
    elif physical_count >= 2:
        score += 1

    recent_high_severity_count = sum(1 for sev in severity_history[-4:] if sev >= 2)
    if recent_high_severity_count >= 2 and score >= 2:
        score += 2
        detected_symptoms.append("trend: sustained distress")
    
    # ===== CONTEXTUAL FILTERING =====
    harmless_contexts = [
        "watched a movie", "watching a movie", "scary movie", "dangerous movie",
        "horror film", "horror movie", "movie", "bad dream", "nightmare", "fictional", "character",
        "story", "book", "tv show", "series", "hypothetical", "what if",
        "imagine if", "suppose", "pretend", "for a project", "assignment",
        "essay", "research", "someone in a movie"
    ]
    
    harmless_context = any(context in q for context in harmless_contexts)
    if harmless_context:
        context_flags.append("harmless_context")

    first_person_current_distress = any(phrase in q for phrase in [
        "i am", "i'm", "im", "i feel", "i can't", "i cant", "my chest",
        "my heart", "help me", "right now", "happening now", "still can't", "still cant"
    ])
    severe_self_harm = any(phrase in q for phrase in [
        "kill myself", "end my life", "i want to die", "suicide",
        "hurt myself", "self harm", "don't want to live"
    ])
    severe_physical = any(phrase in q for phrase in [
        "can't breathe", "cannot breathe", "cant breathe", "chest pain",
        "no air", "suffocating", "pass out", "fainting"
    ])
    worsening_signal = any(phrase in q for phrase in [
        "getting worse", "it's worse", "its worse", "much worse",
        "worse than before", "not stopping", "won't stop", "still can't",
        "still cant", "i still can't", "i still cant", "can't calm down",
        "cant calm down"
    ])

    if harmless_context and not severe_self_harm and not first_person_current_distress:
        score = min(score, 1)
    elif harmless_context and not severe_self_harm and not severe_physical:
        score = min(score, 2)
    elif harmless_context:
        score = max(0, score - 2)
    
    # ===== SEVERITY LEVEL MAPPING =====
    if "can't" in q and recent_panic_context:
        score += 3
        detected_symptoms.append("context continuation: unable to calm")

    repeated_physical_distress = (
        recent_panic_context
        and (
            severe_physical
            or physical_count >= 2
            or any(term in q for term in ["shaking", "shivering", "trembling", "dizzy", "chest tight", "tight chest"])
        )
    )
    sustained_escalation = recent_high_severity_count >= 2 and score >= 6
    critical_panic_context = (
        not harmless_context
        and first_person_current_distress
        and (
            (severe_physical and worsening_signal)
            or (severe_physical and repeated_physical_distress)
            or (worsening_signal and physical_count >= 2)
            or (sustained_escalation and worsening_signal)
        )
    )

    if severe_self_harm and not harmless_context:
        severity = 5
        risk_level = "critical"
    elif critical_panic_context:
        severity = 5  # CRITICAL SUPPORT
        risk_level = "critical"
    elif score >= 10 and (severe_physical or repeated_physical_distress or worsening_signal):
        severity = 4  # PANIC STATE
        risk_level = "panic"
    elif score >= 7 or severe_physical:
        severity = 3  # HIGH DISTRESS
        risk_level = "high"
    elif score >= 4:
        severity = 2  # ELEVATED ANXIETY
        risk_level = "elevated"
    elif score >= 1:
        severity = 1  # MILD DISTRESS
        risk_level = "moderate"
    else:
        severity = 0  # NORMAL
        risk_level = "low"
    
    # ===== ESCALATION TREND FROM EXISTING HISTORY ONLY =====
    simulated_history = severity_history + [severity]
    if len(simulated_history) >= 3:
        last_three = simulated_history[-3:]
        if last_three[-1] > last_three[0]:
            escalation_trend = "escalating"
        elif last_three[-1] < last_three[0]:
            escalation_trend = "improving"
        elif len(set(last_three)) == 1:
            escalation_trend = "stable"
        else:
            escalation_trend = "fluctuating"
    else:
        escalation_trend = "stable"
    
    return {
        "severity": severity,
        "escalation_trend": escalation_trend,
        "risk_level": risk_level,
        "detected_symptoms": detected_symptoms,
        "physical_symptoms": list(dict.fromkeys(physical_symptoms_found)),
        "emotional_signals": list(dict.fromkeys(emotional_signals_found)),
        "symptom_count": len(detected_symptoms),
        "score": score,
        "emotional_intensity": min(10, max(0, score)),
        "context_flags": context_flags,
        "first_person_current_distress": first_person_current_distress,
        "query": q,
    }

def detect_advanced_severity(query, history_messages):
    """
    Backward-compatible pure wrapper.
    Returns: severity (0-3), escalation_trend, risk_level, detected_symptoms
    """
    analysis = analyze_severity(query, history_messages)
    return (
        analysis["severity"],
        analysis["escalation_trend"],
        analysis["risk_level"],
        analysis["detected_symptoms"]
    )

def update_severity_state(analysis):
    """Apply one severity analysis result to session state exactly once."""
    severity = analysis.get("severity", 0)
    st.session_state.severity = severity
    st.session_state.severity_history.append(severity)
    if len(st.session_state.severity_history) > 10:
        st.session_state.severity_history = st.session_state.severity_history[-10:]
    st.session_state.detected_symptoms = analysis.get("detected_symptoms", [])
    st.session_state.physical_symptoms = analysis.get("physical_symptoms", [])
    st.session_state.escalation_trend = analysis.get("escalation_trend", "stable")
    st.session_state.risk_level = analysis.get("risk_level", "low")
    st.session_state.symptom_count = analysis.get("symptom_count", 0)
    st.session_state.emotional_intensity = analysis.get("emotional_intensity", 0)
    st.session_state.panic_detected = severity >= 1
    st.session_state.panic_level = severity
    st.session_state.last_severity_analysis = analysis
    st.session_state.support_flow_stage = get_support_flow_stage(severity, analysis)
    if severity >= 2:
        st.session_state.panic_episode_turns += 1
    elif severity == 0:
        st.session_state.panic_episode_turns = 0
        st.session_state.support_strategy_history = []
    if severity == 0 or any(phrase in normalize_query_text(analysis.get("query", "")) for phrase in RECOVERY_PHRASES):
        st.session_state.recovery_streak += 1
        if severity == 0:
            st.session_state.calming_success_count += 1
    else:
        st.session_state.recovery_streak = 0
    if severity >= 5:
        st.session_state.support_stage = "critical"
    elif severity == 4:
        st.session_state.support_stage = "panic_state"
    elif severity == 3:
        st.session_state.support_stage = "panic_high"
    elif severity == 2:
        st.session_state.support_stage = "grounding"

def detect_emotion(query):
    q = normalize_query_text(query)

    # Self-harm / suicide crisis
    if any(word in q for word in SELF_HARM_KEYWORDS):
        return "crisis"

    # Panic and urgent physical symptoms
    if any(word in q for word in [
        "can't breathe", "cant breathe", "cannot breathe", "no air",
        "suffocating", "shivering", "shiver", "trembling", "tremble",
        "shaking", "shake", "rapid heartbeat", "heart racing",
        "hyperventilate", "panic attack", "panicking", "panic",
        "dizzy", "dizziness", "faint", "chest tight", "tight chest",
        "chest pain", "can't stand", "cant stand"
    ]):
        return "panic"

    # Trauma / memory distress
    if any(word in q for word in [
        "past", "memory", "memories", "incident", "incidents",
        "trauma", "flashback", "remembering", "disturbing", "nothing feels real", "feel disconnected", "outside my body",
        "not real", "dissociating", "detached from reality", "unreal", "disconnected from myself"
    ]):
        return "trauma"

    # Anxiety / Future Fear
    if any(word in q for word in [
        "anxious",
        "nervous",
        "worry",
        "restless",
        "uneasy",
        "fear",
        "scared",
        "future",
        "what if i fail",
        "career",
        "placements",
        "job tension",
        "afraid of failing",
        "fear of future"
        "scared to talk",
        "avoid people",
        "feel judged",
        "nervous around people",
        "social anxiety",
        "awkward around people",
        "fear of people",
        "can't talk to people",
        "anxious around others"
    ]):
        return "future_anxiety"

    # Stress / Burnout
    if any(word in q for word in [
        "stressed",
        "stress",
        "overwhelmed",
        "pressured",
        "burned out",
        "burnt out",
        "mentally tired",
        "emotionally tired",
        "drained",
        "exhausted",
        "can't do this anymore",
        "too tired",
        "angry",
        "frustrated",
        "irritated",
        "annoyed",
        "furious",
        "mad",
        "losing my temper",
        "rage"
    ]):
        return "burnout"

    # Sadness
    if any(word in q for word in [
        "sad", "depressed", "hopeless",
        "tired", "down", "lonely", "burden", "everyone hates me", "people are tired of me", "i trouble everyone", "i make things worse", "nobody wants me", "i'm unwanted"
    ]):
        return "sadness"

    return "neutral"

def detect_contextual_emotion(query, history_messages, analysis):
    """Pure emotion label for display/response using current text plus recent context."""
    emotion = detect_emotion(query)
    if emotion != "neutral":
        return emotion
    q = normalize_query_text(query)
    recent_user = " ".join(
        normalize_query_text(msg.get("content", ""))
        for msg in (history_messages or [])[-6:]
        if msg.get("role") == "user"
    )
    if analysis.get("severity", 0) >= 2:
        return "panic"
    if any(term in recent_user for term in ["panic", "breathe", "shiver", "shake", "tremble", "suffocat"]) and any(term in q for term in ["can't", "cant", "still", "same", "again", "help"]):
        return "panic"
    if any(term in q for term in ["stress", "stressed", "overwhelmed", "can't", "cant"]):
        return "stress"
    return emotion

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

    greetings = [
        "hi", "hello", "hey", "hii", "heyy"
    ]

    if q in greetings:
        return random.choice([
            "Hi. How has your day been so far?",
            "Hello. What's been on your mind lately?",
            "Hey there. How are you feeling today?"
        ])

    if "how are you" in q:
        return random.choice([
            "I'm here and listening.",
            "I'm doing okay. More importantly, how are you feeling?",
            "I'm here with you. Tell me what's going on."
        ])

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
    
    if (
        ask_question
        and st.session_state.support_stage != "silent_support"
        and random.random() > 0.85
    ):
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
    
    response += (
        "Your symptoms sound serious right now, especially the breathing difficulty and dizziness.\n\n"
        "Please contact a trusted person, medical professional, or emergency service if this continues or gets worse.\n"
        "You do not have to handle this alone."
    )
    
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

def remember_support_strategy(strategy):
    """Track support strategies so the bot does not repeat the same intervention."""
    st.session_state.last_response_strategy = strategy
    strategy_stage = {
        "general": 1,
        "emotional_support": 2,
        "breathing": 3,
        "grounding": 4,
        "stabilization": 5,
        "trusted_help": 6,
    }
    st.session_state.support_flow_stage = max(
        st.session_state.get("support_flow_stage", 1),
        strategy_stage.get(strategy, 1)
    )
    history = st.session_state.get("support_strategy_history", [])
    history.append({
        "strategy": strategy,
        "time": time.time(),
        "severity": st.session_state.get("severity", 0)
    })
    st.session_state.support_strategy_history = history[-8:]

def has_used_strategy(strategy):
    return any(item.get("strategy") == strategy for item in st.session_state.get("support_strategy_history", []))

def choose_panic_strategy(severity):
    """Progress panic support from breathing to grounding to outside help."""
    turns = st.session_state.get("panic_episode_turns", 0)
    if severity >= 5 or turns >= 5:
        return "trusted_help"
    if not has_used_strategy("emotional_support"):
        return "emotional_support"
    if not has_used_strategy("breathing"):
        return "breathing"
    if not has_used_strategy("grounding"):
        return "grounding"
    if not has_used_strategy("stabilization"):
        return "stabilization"
    return "trusted_help"

def get_adaptive_panic_response(query, severity, analysis=None):
    """Context-aware panic response that avoids repeated breathing scripts."""
    analysis = analysis or st.session_state.get("last_severity_analysis", {})
    strategy = choose_panic_strategy(severity)
    remember_support_strategy(strategy)

    if strategy == "emotional_support":
        return (
            "That sounds frightening, and I am going to slow this down with you.\n\n"
            "For this moment, just sit or lean somewhere steady and notice that you are here, reading this, one second at a time.\n\n"
            "I will not rush you or overload you with steps."
        )

    if strategy == "breathing":
        return (
            "Let's try one small breathing step, only if it feels possible.\n\n"
            "Breathe in gently through your nose for 3 counts, then exhale slowly for 5 counts. Do that twice, without forcing a deep breath.\n\n"
            "If it feels physically blocked or painful, stop the exercise and tell someone nearby."
        )

    if strategy == "grounding":
        return (
            "You already tried the first calming step, so let's shift instead of repeating it.\n\n"
            "Press both feet into the floor. Name 5 things you can see, then touch one solid object near you and describe its temperature or texture.\n\n"
            "Your job for the next minute is not to solve the panic. It is only to anchor your attention outside the fear loop."
        )

    if strategy == "stabilization":
        return (
            "This is continuing, so let's move into stabilization mode.\n\n"
            "Keep your eyes open, place one hand on a surface, and say quietly: this is a panic wave, not a command. I only need to get through the next minute.\n\n"
            "Do not keep testing your breathing. Keep your attention on the room and the surface under your hand."
        )

    return (
        "Because this is continuing, I do not want you handling it alone.\n\n"
        "Please call or message a trusted person now and tell them: I am having intense panic symptoms and I need you to stay with me. If air hunger, chest pain, faintness, or inability to stand continues, contact local emergency services or urgent medical help.\n\n"
        "I will stay calm with you here, but this is the point where real-world support matters."
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

    stage = st.session_state.support_stage
    # ===== STAGE: GROUNDING =====
    if stage == "grounding":
        return (
            "Let's focus on calming your body first.\n\n"
            "Place both feet on the floor.\n"
            "Take one slow breath in through your nose.\n"
            "Now slowly breathe out.\n\n"
            "You are safe in this moment."
        )
    # ===== STAGE: TRAUMA SUPPORT =====
    if stage == "trauma_support":
        return (
            "Painful memories can feel overwhelming when they suddenly come back.\n\n"
            "Right now, try to gently bring your attention back to the present moment.\n"
            "Look around you and notice a few things you can see or hear.\n\n"
            "You don't have to fight these memories all at once."
        )

    # ===== STAGE: BURNOUT SUPPORT ====
    if stage == "burnout_support":
        return (
            "It sounds like you've been carrying too much for too long.\n\n"
            "Mental exhaustion can make even small things feel overwhelming.\n"
            "Right now, don't focus on fixing everything at once.\n\n"
            "Try giving yourself permission to slow down for a moment and focus on one small thing at a time."
        )
    
    # ===== STAGE: FUTURE ANXIETY SUPPORT =====
    if stage == "future_support":
        return (
            "Fear about the future can feel really overwhelming, especially when everything feels uncertain.\n\n"
            "Right now, try not to solve your entire future at once.\n"
            "Focus only on the next small step you can take today.\n\n"
            "You do not need to have everything figured out immediately."
        )
    
    # ===== STAGE: SOCIAL ANXIETY SUPPORT =====
    if stage == "social_anxiety_support":
        return (
            "Social situations can feel mentally exhausting when you're constantly worried about being judged.\n\n"
            "You're not weak or strange for feeling this way.\n"
            "A lot of people experience anxiety around others, even if they don't show it.\n\n"
            "Try focusing on one small interaction at a time instead of pressuring yourself to be perfect."
        )
    
    # ===== STAGE: BURDEN SUPPORT =====
    if stage == "burden_support":
        return (
            "It sounds like you're carrying a lot of emotional weight right now.\n\n"
            "When people feel overwhelmed or emotionally exhausted, the mind can start convincing them that they're a burden to others.\n\n"
            "That feeling can be very painful, but you still deserve support, care, and space to talk about what you're going through."
        )
    
    # ===== STAGE: ANGER / FRUSTRATION SUPPORT =====
    if stage == "anger_support":
        return (
            "It sounds like a lot of frustration has been building up inside you.\n\n"
            "When emotions become intense, it's easy to react quickly or feel out of control.\n\n"
            "Before responding to anything around you, try pausing for a moment and focusing on slowing your breathing and relaxing your shoulders."
        )
    
    # ===== STAGE: DISSOCIATION SUPPORT =====
    if stage == "dissociation_support":
        return (
            "That disconnected or unreal feeling can happen when the mind becomes overwhelmed.\n\n"
            "Right now, try focusing on your surroundings very gently.\n"
            "Look around and name:\n"
            "• 5 things you can see\n"
            "• 4 things you can touch\n"
            "• 3 things you can hear\n\n"
            "You are here in the present moment, even if things feel unreal right now."
        )

    # ===== STAGE: RECOVERY =====
    if stage == "recovery":
        return (
            "I'm really glad you're feeling even slightly calmer now.\n\n"
            "You handled a very difficult moment step by step, and that matters.\n"
            "Try to be gentle with yourself while your mind and body continue settling down.\n\n"
            "You don't need to recover all at once."
        )

    # ===== STAGE: SILENT DISTRESS SUPPORT =====
    if stage == "silent_support":
        return (
            "That's okay. You don't have to explain everything immediately.\n\n"
            "Sometimes feelings are difficult to put into words, especially when things feel overwhelming.\n\n"
            "I'm still here with you."
        )

    # ===== STAGE: CRITICAL =====
    if stage == "critical":
        return (
            "I'm concerned about the symptoms you're describing.\n\n"
            "Please contact a trusted person, doctor, or emergency service right now — especially if breathing difficulty, dizziness, or inability to stand continues.\n\n"
            "You do not have to handle this alone."
        )
    
    # ===== STAGE: SUICIDE / SELF-HARM CRISIS =====
    if stage == "suicide_support":
        return (
            "I'm really glad you told me this.\n\n"
            "You deserve immediate support, and you do not have to carry this alone.\n\n"
            "Please contact a trusted person, mental health professional, or emergency support service right now.\n\n"
            "If you're in immediate danger, call emergency services or a suicide crisis hotline in your area."
        )

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

        elif severity == 3:  # Stage 3: high
            ack = "I'm really concerned about what you're experiencing right now."
            interpret = "Your symptoms are staying intense, so I don't want to keep repeating the same breathing step."
            step = "Ground yourself with your feet on the floor, then contact a trusted person nearby if this does not start easing soon."
            return compose_response(ack, interpret, step, ask_question=False)

        elif severity >= 4:  # Stage 4+: panic/critical
            ack = "I'm really concerned about what you're experiencing right now."
            interpret = "Difficulty breathing, chest symptoms, faintness, or feeling unable to stand can become serious."
            step = "Please contact emergency services or reach out to a doctor or trusted person nearby now."
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
    
    past_context = " ".join(
        [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    ).lower()
    if (
        ("past" in past_context or "disturbing" in past_context or "memories" in past_context)
        and any(word in query.lower() for word in ["help", "overcome", "escape", "stop", "heal"])
    ):
        return (
            "Healing from painful memories takes time, and you're already taking an important step by talking about it.\n\n"
            "Right now, try not to fight the thoughts aggressively. Instead:\n\n"
            "• Remind yourself that the memories are from the past and cannot harm you right now\n"
            "• Focus on your surroundings — notice 5 things you can see\n"
            "• Take slow breaths and relax your shoulders\n"
            "• Try grounding yourself in the present moment\n\n"
            "You do not need to overcome everything at once. Small steps matter."
        )
    
    if (
        emotion == "trauma"
        and any(word in query.lower() for word in [
            "help",
            "overcome",
            "heal",
            "escape",
            "stop"
        ])
    ):
        return (
            "Healing from painful memories takes time, and you don't have to force yourself to overcome everything immediately.\n\n"
            "Right now, focus on feeling safe in the present moment.\n"
            "Try grounding yourself by noticing:\n"
            "• 5 things you can see\n"
            "• 4 things you can touch\n"
            "• 3 things you can hear\n\n"
            "These memories are painful, but they are memories — you are here in the present right now."
        )

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
    analysis = st.session_state.get("last_severity_analysis", {})

    # Track repeated distress
    if "repeated_distress" not in st.session_state:
        st.session_state.repeated_distress = 0
    
    failure_signs = [
        "i tried",
        "tried many times",
        "not working",
        "no use",
        "still happening",
        "getting worse",
        "worse",
        "not stopping",
        "still panicking",
        "can't calm down",
        "cant calm down"
    ]

    breathing_words = [
        "breathe",
        "breathing",
        "panic",
        "calm down"
    ]
    if any(word in query.lower() for word in breathing_words):
        st.session_state.breathing_attempts += 1
    
    if severity >= 2 or any(sign in query.lower() for sign in failure_signs):
        st.session_state.repeated_distress += 1
    else:
        st.session_state.repeated_distress = 0
    # Force escalation if user keeps struggling
    if (
        st.session_state.repeated_distress >= 2
        and st.session_state.breathing_attempts >= 2
    ):
        severity = max(severity, 4)

    # Reassurance-seeking loop detection
    if any(phrase in query.lower() for phrase in REASSURANCE_SEEKING):
        st.session_state.reassurance_count += 1
    else:
        st.session_state.reassurance_count = max(
            0,
            st.session_state.reassurance_count - 1
        )
    # Repeated reassurance loop
    if st.session_state.reassurance_count >= 3:
        return (
            "I notice that your mind keeps seeking reassurance because you're feeling scared right now.\n\n"
            "Anxiety can create a strong urge to repeatedly check whether you're safe.\n"
            "Instead of searching for certainty, try focusing on slowing your breathing and grounding yourself in the present moment."
        )
    
    # Recovery detected → reduce escalation gradually
    if any(phrase in query.lower() for phrase in RECOVERY_PHRASES):
        st.session_state.panic_level = max(
            0,
            st.session_state.panic_level - 1
        )
        st.session_state.repeated_distress = 0
    
    if emotion in {"panic", "anxiety", "stress"}:
        st.session_state.last_emotion = emotion

    # 🚨 CRITICAL ESCALATION FIRST
    if severity >= 5:
        st.session_state.escalation_count += 1
        if st.session_state.escalation_count >= 2:
            return get_emergency_escalation()   # STRONG message
        else:
            return get_adaptive_panic_response(query, severity, analysis)
    # ⚠️ PANIC STAGES
    if severity >= 2 or analysis.get("physical_symptoms"):
        return get_adaptive_panic_response(query, severity, analysis)

    # For normal cases, try LLM if available
    if groq_client is None:
        return generate_response(query, context, intent, emotion, severity)
    
    context_summary = context[:300] if context else ""
    recent_user_context = " | ".join(
        msg["content"] for msg in st.session_state.chat_history[-8:]
        if msg.get("role") == "user"
    )[-500:]
    used_strategies = ", ".join(
        item.get("strategy", "support").replace("_", " ")
        for item in st.session_state.get("support_strategy_history", [])[-4:]
    ) or "none yet"
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
- Use recent context so your reply feels continuous
- Avoid repeating the last support strategy or sentence shape

Critical handling:
- If user shows panic (breathing issue, fear, shaking):
    → Guide breathing step-by-step
- If user says it's getting worse AGAIN:
    → Strongly tell them to seek medical help

Never say:
- "everything will be fine"

Context:
{context_summary}

Recent user context:
{recent_user_context}

Current support stage:
{get_support_stage_label(st.session_state.get("support_flow_stage", 1))}

Recent support strategies:
{used_strategies}
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

/* ===== Calm Overlay ===== */

.calm-overlay {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    z-index: 9999;

    width: 320px;
    height: 320px;

    border-radius: 30px;

    background: rgba(15, 23, 42, 0.72);

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 0 60px rgba(96,165,250,0.18),
        0 0 120px rgba(59,130,246,0.08);

    animation: fadeOverlay 0.5s ease;
}

.breathing-circle {
    width: 120px;
    height: 120px;

    border-radius: 50%;

    background:
        radial-gradient(circle,
        #60a5fa 0%,
        #3b82f6 45%,
        #1d4ed8 100%);

    animation: breathe 9s ease-in-out infinite;

    box-shadow:
        0 0 40px rgba(96,165,250,0.5),
        0 0 80px rgba(59,130,246,0.25);
}

.calm-text {
    margin-top: 24px;

    text-align: center;

    color: #e2e8f0;

    font-size: 16px;

    line-height: 1.7;
}

.calm-subtext {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 8px;
}

@keyframes breathe {
    0%, 100% {
        transform: scale(0.82);
        opacity: 0.72;
    }

    50% {
        transform: scale(1.28);
        opacity: 1;
    }
}

@keyframes fadeOverlay {
    from {
        opacity: 0;
        transform: translate(-50%, -46%);
    }

    to {
        opacity: 1;
        transform: translate(-50%, -50%);
    }
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

st.markdown("""
<style>
:root {
    --calm-bg: #020617;
    --calm-panel: rgba(15, 23, 42, 0.76);
    --calm-panel-strong: rgba(15, 23, 42, 0.92);
    --calm-border: rgba(148, 163, 184, 0.18);
    --calm-blue: #60a5fa;
    --calm-mint: #5eead4;
    --calm-text: #e5edf8;
    --calm-muted: #9fb0c7;
}

.stApp {
    background:
        radial-gradient(circle at 18% 0%, rgba(96, 165, 250, 0.16), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(45, 212, 191, 0.11), transparent 30%),
        linear-gradient(135deg, #020617 0%, #07111f 46%, #020617 100%) !important;
}

.main .block-container {
    max-width: 980px;
    padding-top: 0 !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
}

.header-section {
    max-width: 880px;
    margin: 18px auto 22px !important;
    padding: 34px 24px 28px !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 26px !important;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(30, 41, 59, 0.54)),
        radial-gradient(circle at top, rgba(96, 165, 250, 0.16), transparent 52%) !important;
    box-shadow: 0 26px 80px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(18px);
}

.header-title {
    font-size: clamp(28px, 5vw, 44px) !important;
    color: var(--calm-text) !important;
    letter-spacing: 0 !important;
}

.header-subtitle {
    color: var(--calm-muted) !important;
    font-size: 15px !important;
}

.header-tagline {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #bfdbfe !important;
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 999px;
    padding: 8px 14px;
}

.chat-container {
    max-width: 820px !important;
    padding: 12px 12px 28px !important;
}

.chat-message-user,
.chat-message-bot {
    animation: messageRise 260ms ease-out !important;
}

.message-bubble-user,
.message-bubble-bot {
    max-width: min(76%, 620px) !important;
    border-radius: 22px !important;
    padding: 15px 18px !important;
    line-height: 1.62 !important;
    font-size: 15.5px !important;
}

.message-bubble-user {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.98), rgba(37, 99, 235, 0.9)) !important;
    box-shadow: 0 14px 36px rgba(37, 99, 235, 0.28) !important;
}

.message-bubble-bot {
    background: rgba(15, 23, 42, 0.86) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
    backdrop-filter: blur(14px);
}

div[data-testid="stButton"] button {
    border-radius: 999px !important;
    border: 1px solid rgba(96, 165, 250, 0.24) !important;
    background: rgba(15, 23, 42, 0.68) !important;
    color: #cfe2ff !important;
    min-height: 44px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    transition: transform 160ms ease, border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    border-color: rgba(96, 165, 250, 0.62) !important;
    background: rgba(30, 41, 59, 0.86) !important;
    box-shadow: 0 18px 38px rgba(37, 99, 235, 0.18);
}

[data-testid="stChatInput"] {
    border-radius: 24px !important;
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    box-shadow: 0 18px 54px rgba(0, 0, 0, 0.34) !important;
    backdrop-filter: blur(18px);
}

[data-testid="stChatInput"] textarea {
    color: var(--calm-text) !important;
    font-size: 15.5px !important;
}

.status-indicator {
    top: 18px !important;
    right: 18px !important;
    border-radius: 999px !important;
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid rgba(96, 165, 250, 0.28) !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(16px);
}

.alert-banner {
    max-width: 820px;
    margin: 0 auto 16px !important;
    border-radius: 20px !important;
    border: 1px solid rgba(248, 113, 113, 0.28) !important;
    box-shadow: 0 18px 48px rgba(220, 38, 38, 0.18);
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(96, 165, 250, 0.14), transparent 34%),
        rgba(2, 6, 23, 0.94) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--calm-text);
}

.analysis-card {
    background: rgba(15, 23, 42, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    padding: 14px 14px 12px;
    margin: 10px 0;
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.analysis-kicker {
    color: #93c5fd;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.analysis-value {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
}

.analysis-muted {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 4px;
}

.analysis-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 6px 10px;
    margin: 3px 4px 3px 0;
    background: rgba(96, 165, 250, 0.10);
    border: 1px solid rgba(96, 165, 250, 0.20);
    color: #bfdbfe;
    font-size: 12px;
}

.meter-track {
    height: 9px;
    border-radius: 999px;
    background: rgba(51, 65, 85, 0.76);
    overflow: hidden;
    margin-top: 10px;
}

.meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #60a5fa, #f97316);
}

@keyframes messageRise {
    from { opacity: 0; transform: translateY(8px) scale(0.99); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

@media (max-width: 720px) {
    .main .block-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    .header-section {
        margin-top: 10px !important;
        border-radius: 20px !important;
        padding: 26px 16px !important;
    }
    .message-bubble-user,
    .message-bubble-bot {
        max-width: 88% !important;
        font-size: 14.5px !important;
    }
    .status-indicator {
        position: static !important;
        width: fit-content;
        margin: 8px auto 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# ========== UI LAYOUT ==========
close_expired_breathing_popup()

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
        {"If this doesn't improve, please reach out to emergency services." if st.session_state.severity >= 5 else ""}
    </div>
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
chat_query = st.chat_input("Tell me what's on your mind...", key="main_input")
if chat_query:
    query = chat_query

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
    severity_analysis = analyze_severity(
        query,
        st.session_state.chat_history,
        st.session_state.severity_history
    )
    emotion = detect_contextual_emotion(query, st.session_state.chat_history, severity_analysis)
    update_severity_state(severity_analysis)
    record_emotional_timeline(emotion, severity_analysis)
    severity = severity_analysis["severity"]

    query_lower = query.lower()
    # Stage transition
    if emotion in ["anxiety", "stress", "panic"] and severity < 3:
        st.session_state.support_stage = "grounding"
    if emotion == "trauma":
        st.session_state.support_stage = "trauma_support"
    if emotion == "burnout":
        st.session_state.support_stage = "burnout_support"
    if emotion == "future_anxiety":
        st.session_state.support_stage = "future_support"
    if any(word in query_lower for word in [
        "scared to talk",
        "avoid people",
        "feel judged",
        "social anxiety",
        "nervous around people"
    ]):
        st.session_state.support_stage = "social_anxiety_support"

    if any(word in query_lower for word in [
        "burden",
        "people are tired of me",
        "i trouble everyone",
        "everyone hates me",
        "i'm unwanted"
    ]):
        st.session_state.support_stage = "burden_support"

    if any(word in query_lower for word in [
        "angry",
        "frustrated",
        "irritated",
        "annoyed",
        "furious",
        "rage"
    ]):
        st.session_state.support_stage = "anger_support"

    if any(word in query_lower for word in [
        "nothing feels real",
        "feel disconnected",
        "outside my body",
        "not real",
        "dissociating",
        "unreal"
    ]):
        st.session_state.support_stage = "dissociation_support"

    if severity >= 5:
        st.session_state.support_stage = "critical"
    if emotion == "crisis":
        st.session_state.support_stage = "suicide_support"
    if any(word in query_lower for word in RECOVERY_PHRASES):
        st.session_state.support_stage = "recovery"
    if query_lower.strip() in SILENT_DISTRESS:
        st.session_state.support_stage = "silent_support"


    # Retrieve context
    context, scores = retrieve(query, 3)
    confidence = np.mean(scores) * 100 if scores is not None and len(scores) > 0 else 0
    confidence_label = (
        "High" if confidence > 70
        else "Medium" if confidence > 40
        else "Low"
        
    )
    
    st.session_state.last_response_strategy = "general"

    # Small talk first
    small_talk = handle_small_talk(query)
    if small_talk:
        answer = small_talk
    else:
        # Serious cases → controlled logic
        answer = generate_llm_response(
            query,
            context,
            intent,
            emotion,
            severity
        )
    
    # Check if breathing guidance was provided
    popup_blocked_strategies = {"emotional_support", "grounding", "stabilization", "trusted_help"}
    popup_cooldown_elapsed = time.time() - st.session_state.get("last_breathing_popup_at", 0.0) > 45
    if (
        detect_breathing_trigger(answer)
        and st.session_state.get("last_response_strategy") not in popup_blocked_strategies
        and popup_cooldown_elapsed
    ):
        start_breathing_popup()
        st.session_state.last_breathing_popup_at = time.time()
    else:
        st.session_state.show_breathing_popup = False
        st.session_state.breathing_popup_active = False
    
    # Add to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_bot_response = answer
    
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

# ========== BREATHING POPUP COMPONENT ==========
if st.session_state.show_breathing_popup:
    popup_id = st.session_state.get("breathing_popup_id", "breathing-popup")
    popup_expires_at = st.session_state.get("breathing_popup_expires_at", 0.0)
    remaining_ms = max(500, int((popup_expires_at - time.time()) * 1000))
    breathing_cycle = "8.5s" if st.session_state.get("severity", 0) >= 4 else "7s"
    breathing_html = f"""
    <script>
    (function() {{
        const popupId = "{popup_id}";
        const durationMs = {remaining_ms};
        const breathingCycle = "{breathing_cycle}";
        const doc = window.parent.document;
        const storageKey = "breathingPopupDismissed:" + popupId;

        function removePopup() {{
            const existing = doc.getElementById("breathing-popup-root");
            if (existing) existing.remove();
            window.parent.localStorage.setItem(storageKey, "1");
        }}

        if (window.parent.localStorage.getItem(storageKey) === "1") {{
            removePopup();
            return;
        }}

        const oldPopup = doc.getElementById("breathing-popup-root");
        if (oldPopup) oldPopup.remove();

        const styleId = "breathing-popup-style";
        let style = doc.getElementById(styleId);
        if (!style) {{
            style = doc.createElement("style");
            style.id = styleId;
            style.textContent = `
                #breathing-popup-root {{
                    position: fixed;
                    inset: 0;
                    z-index: 2147483000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background:
                        radial-gradient(circle at 50% 42%, rgba(96, 165, 250, 0.18), transparent 34%),
                        rgba(2, 6, 23, 0.58);
                    backdrop-filter: blur(10px) brightness(1.16) saturate(1.08);
                    -webkit-backdrop-filter: blur(10px) brightness(1.16) saturate(1.08);
                    animation: breathingOverlayIn 360ms ease-out;
                }}

                #breathing-popup-root.breathing-popup-closing {{
                    animation: breathingOverlayOut 360ms ease-in forwards;
                }}

                .breathing-popup-card {{
                    width: min(410px, calc(100vw - 30px));
                    min-height: 440px;
                    border-radius: 34px;
                    background:
                        linear-gradient(145deg, rgba(15, 23, 42, 0.90), rgba(30, 41, 59, 0.72)),
                        radial-gradient(circle at top, rgba(96, 165, 250, 0.16), transparent 54%);
                    border: 1px solid rgba(147, 197, 253, 0.32);
                    box-shadow:
                        0 24px 70px rgba(0, 0, 0, 0.42),
                        0 0 90px rgba(96, 165, 250, 0.26),
                        inset 0 1px 0 rgba(255, 255, 255, 0.06);
                    color: #e2e8f0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px 34px;
                    position: relative;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    animation: breathingCardIn 420ms cubic-bezier(0.2, 0.8, 0.2, 1), breathingFloat 5s ease-in-out infinite;
                }}

                .breathing-popup-close {{
                    position: absolute;
                    top: 16px;
                    right: 16px;
                    width: 34px;
                    height: 34px;
                    border-radius: 50%;
                    border: 1px solid rgba(148, 163, 184, 0.28);
                    background: rgba(30, 41, 59, 0.8);
                    color: #cbd5e1;
                    cursor: pointer;
                    font-size: 22px;
                    line-height: 1;
                }}

                .breathing-popup-close:hover {{
                    background: rgba(51, 65, 85, 0.95);
                }}

                .breathing-popup-circle {{
                    width: 144px;
                    height: 144px;
                    border-radius: 50%;
                    background: radial-gradient(circle at 35% 30%, #dbeafe 0%, #93c5fd 22%, #60a5fa 45%, #2563eb 72%, #1e3a8a 100%);
                    box-shadow:
                        0 0 52px rgba(96, 165, 250, 0.62),
                        0 0 110px rgba(37, 99, 235, 0.34),
                        inset 12px 12px 28px rgba(255, 255, 255, 0.12),
                        inset -22px -22px 54px rgba(15, 23, 42, 0.34);
                    margin-bottom: 26px;
                    animation: breathingCircle var(--breathing-cycle, 7s) ease-in-out infinite;
                }}

                .breathing-popup-title {{
                    color: #93c5fd;
                    font-size: 21px;
                    font-weight: 650;
                    margin-bottom: 10px;
                }}

                .breathing-popup-copy {{
                    color: #cbd5e1;
                    font-size: 14px;
                    line-height: 1.6;
                    text-align: center;
                    margin-bottom: 18px;
                }}

                .breathing-popup-rhythm {{
                    color: #bfdbfe;
                    font-size: 13px;
                    font-weight: 700;
                    letter-spacing: 0;
                    text-align: center;
                    animation: breathingText var(--breathing-cycle, 7s) ease-in-out infinite;
                }}

                .breathing-popup-grounding {{
                    margin-top: 18px;
                    color: #94a3b8;
                    font-size: 12.5px;
                    line-height: 1.55;
                    text-align: center;
                    max-width: 290px;
                }}

                @keyframes breathingOverlayIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}

                @keyframes breathingOverlayOut {{
                    from {{ opacity: 1; }}
                    to {{ opacity: 0; }}
                }}

                @keyframes breathingCardIn {{
                    from {{ opacity: 0; transform: translateY(12px) scale(0.96); }}
                    to {{ opacity: 1; transform: translateY(0) scale(1); }}
                }}

                @keyframes breathingFloat {{
                    0%, 100% {{ transform: translateY(0); }}
                    50% {{ transform: translateY(-4px); }}
                }}

                @keyframes breathingCircle {{
                    0%, 100% {{ transform: scale(0.80); opacity: 0.74; }}
                    38% {{ transform: scale(1.18); opacity: 1; }}
                    52% {{ transform: scale(1.18); opacity: 0.96; }}
                }}

                @keyframes breathingText {{
                    0%, 100% {{ opacity: 0.64; }}
                    50% {{ opacity: 1; }}
                }}
            `;
            doc.head.appendChild(style);
        }}

        const root = doc.createElement("div");
        root.id = "breathing-popup-root";
        root.setAttribute("role", "dialog");
        root.setAttribute("aria-modal", "true");
        root.style.setProperty("--breathing-cycle", breathingCycle);
        root.innerHTML = `
            <div class="breathing-popup-card">
                <button class="breathing-popup-close" type="button" aria-label="Close breathing exercise">&times;</button>
                <div class="breathing-popup-circle" aria-hidden="true"></div>
                <div class="breathing-popup-title">Breathe Slowly</div>
                <div class="breathing-popup-copy">Follow the circle gently. No forcing, no perfect timing.</div>
                <div class="breathing-popup-rhythm">INHALE &middot; HOLD &middot; EXHALE</div>
                <div class="breathing-popup-grounding">Notice one thing you can see, one thing you can feel, and let your shoulders soften.</div>
            </div>
        `;

        function closeWithAnimation() {{
            const current = doc.getElementById("breathing-popup-root");
            if (!current) return;
            current.classList.add("breathing-popup-closing");
            window.parent.localStorage.setItem(storageKey, "1");
            setTimeout(() => current.remove(), 360);
        }}

        root.addEventListener("click", function(event) {{
            if (event.target === root || event.target.closest(".breathing-popup-close")) {{
                closeWithAnimation();
            }}
        }});

        doc.body.appendChild(root);
        window.setTimeout(closeWithAnimation, durationMs);
    }})();
    </script>
    """
    components.html(breathing_html, height=0, width=0)

if False and st.session_state.show_breathing_popup:
    breathing_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            html, body {
                background: transparent !important;
                overflow: hidden !important;
            }
            
            .breathing-popup-container {
                position: fixed !important;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 99999;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.4s ease-in;
            }
            
            .breathing-popup-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(4px);
                z-index: -1;
            }
            
            .breathing-modal {
                position: relative;
                width: 360px;
                height: 420px;
                border-radius: 32px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(16px);
                border: 1.5px solid rgba(96, 165, 250, 0.3);
                box-shadow: 
                    0 0 60px rgba(96, 165, 250, 0.25),
                    0 0 120px rgba(59, 130, 246, 0.15),
                    0 20px 60px rgba(0, 0, 0, 0.4);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 40px;
                z-index: 100000;
                animation: popupAppear 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
            }
            
            @keyframes popupAppear {
                from {
                    opacity: 0;
                    transform: scale(0.7) translateY(-30px);
                }
                to {
                    opacity: 1;
                    transform: scale(1) translateY(0);
                }
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .breathing-circle {
                width: 140px;
                height: 140px;
                border-radius: 50%;
                background: radial-gradient(circle at 35% 35%, rgba(147, 197, 253, 0.8), #3b82f6, #1e40af);
                animation: breatheAnimation 6s ease-in-out infinite;
                box-shadow: 
                    0 0 50px rgba(96, 165, 250, 0.6),
                    0 0 100px rgba(59, 130, 246, 0.3),
                    inset -20px -20px 60px rgba(0, 0, 0, 0.2),
                    inset 10px 10px 30px rgba(255, 255, 255, 0.1);
                margin-bottom: 24px;
                position: relative;
            }
            
            .breathing-circle::before {
                content: '';
                position: absolute;
                top: -8px;
                left: -8px;
                right: -8px;
                bottom: -8px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(96, 165, 250, 0.3), transparent);
                animation: pulseRing 6s ease-in-out infinite;
                z-index: -1;
            }
            
            @keyframes breatheAnimation {
                0%, 100% {
                    transform: scale(0.85);
                    opacity: 0.7;
                }
                50% {
                    transform: scale(1.15);
                    opacity: 1;
                }
            }
            
            @keyframes pulseRing {
                0%, 100% {
                    transform: scale(0.8);
                    opacity: 0;
                }
                50% {
                    transform: scale(1.3);
                    opacity: 0.5;
                }
            }
            
            .breathing-text {
                text-align: center;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                z-index: 1;
            }
            
            .breathing-title {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #60a5fa;
                letter-spacing: 0.5px;
            }
            
            .breathing-instruction {
                font-size: 14px;
                color: #cbd5e1;
                margin-bottom: 20px;
                line-height: 1.6;
            }
            
            .breathing-rhythm {
                font-size: 16px;
                color: #93c5fd;
                font-weight: 500;
                letter-spacing: 3px;
                animation: fadeInOut 6s ease-in-out infinite;
            }
            
            @keyframes fadeInOut {
                0%, 100% { opacity: 0.6; }
                25% { opacity: 1; }
                50% { opacity: 0.6; }
                75% { opacity: 1; }
            }
            
            .breathing-hint {
                font-size: 12px;
                color: #64748b;
                margin-top: 16px;
                font-style: italic;
            }
            
            .close-button {
                position: absolute;
                top: 16px;
                right: 16px;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: rgba(100, 116, 139, 0.5);
                border: 1px solid rgba(148, 163, 184, 0.3);
                color: #94a3b8;
                font-size: 18px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }
            
            .close-button:hover {
                background: rgba(100, 116, 139, 0.8);
                color: #cbd5e1;
                border-color: rgba(148, 163, 184, 0.6);
            }
        </style>
    </head>
    <body>
        <div class="breathing-popup-container" id="popupContainer">
            <div class="breathing-popup-backdrop" id="backdrop"></div>
            <div class="breathing-modal">
                <button class="close-button" onclick="closePopup()">×</button>
                <div class="breathing-circle"></div>
                <div class="breathing-text">
                    <div class="breathing-title">Breathe Slowly</div>
                    <div class="breathing-instruction">
                        Follow the circle and match its rhythm
                    </div>
                    <div class="breathing-rhythm">INHALE • HOLD • EXHALE</div>
                    <div class="breathing-hint">You are doing great. I'm here with you.</div>
                </div>
            </div>
        </div>
        
        <script>
            let popupTimeout;
            const POPUP_DURATION = 14000; // 14 seconds
            
            function closePopup() {
                const container = document.getElementById('popupContainer');
                if (container) {
                    container.style.animation = 'fadeOut 0.4s ease-out';
                    setTimeout(() => {
                        container.remove();
                        // Signal to Streamlit that popup should close
                        window.parent.postMessage({type: 'breathing-popup-close'}, '*');
                    }, 400);
                }
            }
            
            function autoClosePopup() {
                closePopup();
            }
            
            // Start auto-close timer
            popupTimeout = setTimeout(autoClosePopup, POPUP_DURATION);
            
            // Close on backdrop click
            const backdrop = document.getElementById('backdrop');
            if (backdrop) {
                backdrop.addEventListener('click', closePopup);
            }
            
            // Add fadeOut animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes fadeOut {
                    from { opacity: 1; }
                    to { opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        </script>
    </body>
    </html>
    """
    
    components.html(
    breathing_html,
    height=700,
    width=1200,
)

# Sidebar debug info
with st.sidebar:
    st.markdown("### System Analysis")
    
    if st.session_state.chat_history:
        last_query = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        if last_query:
            intent = get_intent(last_query, "")
            severity_analysis = st.session_state.get("last_severity_analysis") or analyze_severity(
                last_query,
                st.session_state.chat_history,
                st.session_state.severity_history
            )
            severity = severity_analysis["severity"]
            emotion = detect_contextual_emotion(last_query, st.session_state.chat_history, severity_analysis)
            trend_label, trend_color = get_trend_indicator()
            stability = get_stability_score()
            intensity = st.session_state.emotional_intensity
            support_stage = get_support_stage_label(st.session_state.support_flow_stage)
            severity_width = min(100, int((severity / 5) * 100))
            physical = st.session_state.physical_symptoms[:4]
            physical_html = "".join(f'<span class="analysis-pill">{symptom}</span>' for symptom in physical) or '<span class="analysis-pill">No acute physical pattern</span>'
            timeline = format_timeline_summary()

            st.markdown(f"""
            <div class="analysis-card">
                <div class="analysis-kicker">Current State</div>
                <div class="analysis-value">{get_severity_label(severity).title()}</div>
                <div class="analysis-muted">{emotion.title()} · {intent.title()} · {len(st.session_state.chat_history)} messages</div>
                <div class="meter-track"><div class="meter-fill" style="width:{severity_width}%"></div></div>
            </div>
            <div class="analysis-card">
                <div class="analysis-kicker">Support Flow</div>
                <div class="analysis-value">{support_stage}</div>
                <div class="analysis-muted">Strategy: {st.session_state.last_response_strategy.replace('_', ' ').title()}</div>
                <span class="analysis-pill" style="border-color:{trend_color}; color:{trend_color};">{trend_label}</span>
                <span class="analysis-pill">Intensity {intensity}/10</span>
            </div>
            <div class="analysis-card">
                <div class="analysis-kicker">Stability</div>
                <div class="analysis-value">{stability}%</div>
                <div class="analysis-muted">Higher means the recent pattern is calmer and less volatile.</div>
                <div class="meter-track"><div class="meter-fill" style="width:{stability}%"></div></div>
            </div>
            <div class="analysis-card">
                <div class="analysis-kicker">Physical Signals</div>
                <div>{physical_html}</div>
            </div>
            <div class="analysis-card">
                <div class="analysis-kicker">Timeline</div>
                <div class="analysis-muted">{timeline}</div>
            </div>
            """, unsafe_allow_html=True)
            
            
