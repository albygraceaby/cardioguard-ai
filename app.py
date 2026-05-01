"""
CardioGuard AI — Heart Disease Prediction System
Premium SaaS Healthcare Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from model import (
    load_model, predict, evaluate_model,
    get_top_features, get_all_feature_importances, FEATURE_LABELS,
)

matplotlib.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "#0d1321",
    "axes.edgecolor": "#1e293b", "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "text.color": "#e2e8f0", "grid.color": "#1e293b",
})

# ── Cached model ──
@st.cache_resource
def init_model():
    return load_model()

def _init_session():
    if "model" not in st.session_state:
        m, fn, xt, yt = init_model()
        st.session_state.model = m
        st.session_state.feature_names = fn
        st.session_state.X_test = xt
        st.session_state.y_test = yt
    if "patients" not in st.session_state:
        st.session_state.patients = _gen_patients()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def _gen_patients():
    rng = np.random.RandomState(7)
    names = ["Aarav Mehta","Priya Sharma","Rohan Gupta","Sneha Iyer",
             "Vikram Singh","Ananya Das","Karthik Reddy","Meera Nair",
             "Arjun Patel","Divya Joshi","Sanjay Rao","Kavita Bhatt"]
    recs = []
    for n in names:
        v = [rng.randint(30,75), rng.randint(0,2), rng.randint(0,4),
             rng.randint(100,190), rng.randint(150,400), rng.randint(0,2),
             rng.randint(0,3), rng.randint(80,200), rng.randint(0,2),
             round(rng.uniform(0,5),1), rng.randint(0,3), rng.randint(0,4), rng.randint(0,4)]
        p, pb = predict(st.session_state.model, v)
        recs.append({"name":n,"values":v,"prediction":p,"probability":pb})
    return recs

# ── CSS ──
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #1b2838 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d19 0%, #0f1729 50%, #131d30 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* Glass card */
.glass {
    background: rgba(15,23,42,0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.glass-sm {
    background: rgba(15,23,42,0.5);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* Metric card */
.metric-card {
    background: rgba(15,23,42,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
}
.metric-card .label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; margin:0; font-weight:600; }
.metric-card .value { font-size:30px; font-weight:800; margin:6px 0 0; }

/* Risk cards */
.risk-high {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(185,28,28,0.15));
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 14px; padding: 22px; text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.15));
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 14px; padding: 22px; text-align: center;
}

/* Patient card */
.patient-card {
    background: rgba(15,23,42,0.5);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
    border-left: 4px solid;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}
.patient-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.35); }
.patient-high { border-left-color: #ef4444; border-top: 1px solid rgba(248,113,113,0.15); border-right: 1px solid rgba(248,113,113,0.08); border-bottom: 1px solid rgba(248,113,113,0.08); }
.patient-low { border-left-color: #22c55e; border-top: 1px solid rgba(74,222,128,0.15); border-right: 1px solid rgba(74,222,128,0.08); border-bottom: 1px solid rgba(74,222,128,0.08); }

/* Badge */
.badge { padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.5px; display:inline-block; }
.badge-high { background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.3); }
.badge-low { background:rgba(34,197,94,0.2); color:#4ade80; border:1px solid rgba(34,197,94,0.3); }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; padding: 12px 18px; border-radius: 18px 18px 4px 18px;
    margin: 8px 0; max-width: 80%; margin-left: auto; font-size: 14px;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3);
}
.chat-bot {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,102,241,0.1);
    color: #e2e8f0; padding: 12px 18px; border-radius: 18px 18px 18px 4px;
    margin: 8px 0; max-width: 80%; font-size: 14px;
    backdrop-filter: blur(8px);
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important; font-size: 15px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

/* Form & inputs */
.stForm { background: rgba(15,23,42,0.4); border-radius: 16px; border: 1px solid rgba(99,102,241,0.08); padding: 8px; }
.stNumberInput, .stSelectbox { margin-bottom: -8px; }

/* Chat input */
.stChatInput textarea { border-radius: 14px !important; background: rgba(15,23,42,0.6) !important; border: 1px solid rgba(99,102,241,0.15) !important; }

/* Section title */
.section-title {
    font-size: 28px; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.section-sub { color: #64748b; font-size: 14px; margin-top: 0; }

/* Progress bar */
.prog-wrap { background: rgba(15,23,42,0.6); border-radius: 8px; height: 20px; overflow: hidden; margin: 4px 0; border: 1px solid rgba(99,102,241,0.08); }
.prog-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

/* Headings override */
h2, h3 { background: linear-gradient(135deg, #818cf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }

/* Divider */
hr { border-color: rgba(99,102,241,0.1) !important; }

/* Hide branding */
#MainMenu {visibility:hidden;} footer {visibility:hidden;}

/* Radio pills */
.stRadio > div { gap: 0 !important; }
.stRadio label { transition: all 0.15s; }
</style>
"""

# ── Page: Patient ──
def page_patient():
    st.markdown("<p class='section-title'>🫀 Patient Risk Assessment</p><p class='section-sub'>Enter your health parameters to check heart disease risk.</p>", unsafe_allow_html=True)

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"][x])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak ST", [0,1,2], format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
            ca = st.selectbox("Major Vessels (0-4)", [0,1,2,3,4])
            thal = st.selectbox("Thalassemia", [0,1,2,3], format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Other"][x])
        submitted = st.form_submit_button("🔍  Analyze Risk", use_container_width=True)

    if submitted:
        inp = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        pred, prob = predict(st.session_state.model, inp)
        top = get_top_features(st.session_state.model, st.session_state.feature_names)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        if pred == 1:
            r1.markdown("<div class='risk-high'><p class='label' style='color:#fca5a5;font-size:11px;letter-spacing:1.5px;margin:0;'>RISK LEVEL</p><p style='font-size:28px;font-weight:800;color:#f87171;margin:6px 0 0;'>⚠️ HIGH RISK</p></div>", unsafe_allow_html=True)
        else:
            r1.markdown("<div class='risk-low'><p class='label' style='color:#86efac;font-size:11px;letter-spacing:1.5px;margin:0;'>RISK LEVEL</p><p style='font-size:28px;font-weight:800;color:#4ade80;margin:6px 0 0;'>✅ LOW RISK</p></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='metric-card'><p class='label'>Probability</p><p class='value' style='color:#818cf8;'>{prob*100:.1f}%</p></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='metric-card'><p class='label'>Confidence</p><p class='value' style='color:#c084fc;'>{max(prob,1-prob)*100:.1f}%</p></div>", unsafe_allow_html=True)

        st.markdown("<div class='glass' style='margin-top:20px;'>", unsafe_allow_html=True)
        st.markdown("### 🔑 Top Contributing Factors")
        colors = ["#6366f1","#8b5cf6","#06b6d4"]
        for i,(feat,imp) in enumerate(top):
            pct = imp*100
            st.markdown(f"<p style='color:#cbd5e1;font-weight:600;margin:12px 0 2px;'>{feat}</p><div class='prog-wrap'><div class='prog-fill' style='width:{pct}%;background:linear-gradient(90deg,{colors[i]},transparent);'></div></div><p style='color:#64748b;font-size:12px;margin:2px 0;'>{pct:.1f}% importance</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Doctor ──
def page_doctor():
    st.markdown("<p class='section-title'>🩺 Doctor Dashboard</p><p class='section-sub'>Patient records and risk overview.</p>", unsafe_allow_html=True)

    patients = st.session_state.patients
    total = len(patients)
    high = sum(1 for p in patients if p["prediction"]==1)
    low = total - high

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><p class='label'>Total Patients</p><p class='value' style='color:#e2e8f0;'>{total}</p></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card' style='border:1px solid rgba(239,68,68,0.2);'><p class='label'>High Risk</p><p class='value' style='color:#f87171;'>{high}</p></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card' style='border:1px solid rgba(34,197,94,0.2);'><p class='label'>Low Risk</p><p class='value' style='color:#4ade80;'>{low}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    filt = st.radio("Filter", ["All","High Risk","Low Risk"], horizontal=True)
    filtered = patients
    if filt == "High Risk":
        filtered = [p for p in patients if p["prediction"]==1]
    elif filt == "Low Risk":
        filtered = [p for p in patients if p["prediction"]==0]

    fn = st.session_state.feature_names
    for p in filtered:
        cls = "patient-high" if p["prediction"]==1 else "patient-low"
        bcls = "badge-high" if p["prediction"]==1 else "badge-low"
        btxt = "HIGH RISK" if p["prediction"]==1 else "LOW RISK"
        details = " · ".join(f"<b>{FEATURE_LABELS.get(fn[i],fn[i])}</b>: {p['values'][i]}" for i in range(len(fn)))
        prob_color = "#f87171" if p["prediction"]==1 else "#4ade80"
        st.markdown(
            f"<div class='patient-card {cls}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:17px;font-weight:700;color:#e2e8f0;'>{p['name']}</span>"
            f"<span class='badge {bcls}'>{btxt}</span></div>"
            f"<p style='color:#64748b;margin:8px 0 0;font-size:12px;line-height:1.8;'>{details}</p>"
            f"<p style='color:{prob_color};margin:6px 0 0;font-size:13px;font-weight:600;'>Probability: {p['probability']*100:.1f}%</p>"
            f"</div>", unsafe_allow_html=True)


# ── Page: Analytics ──
def page_analytics():
    st.markdown("<p class='section-title'>📊 Model Analytics</p><p class='section-sub'>Performance metrics and feature analysis.</p>", unsafe_allow_html=True)

    model = st.session_state.model
    X_test, y_test = st.session_state.X_test, st.session_state.y_test
    fn = st.session_state.feature_names
    acc, cm, report = evaluate_model(model, X_test, y_test)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><p class='label'>Accuracy</p><p class='value' style='color:#4ade80;'>{acc*100:.1f}%</p></div>", unsafe_allow_html=True)
    prec = report.get("1",{}).get("precision",0)
    rec = report.get("1",{}).get("recall",0)
    c2.markdown(f"<div class='metric-card'><p class='label'>Precision</p><p class='value' style='color:#818cf8;'>{prec*100:.1f}%</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><p class='label'>Recall</p><p class='value' style='color:#c084fc;'>{rec*100:.1f}%</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(cm, cmap="PuBu", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=24,fontweight="bold",
                        color="#fff" if cm[i,j]>cm.max()/2 else "#818cf8")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Low Risk","High Risk"]); ax.set_yticklabels(["Low Risk","High Risk"])
        ax.set_xlabel("Predicted",fontsize=11); ax.set_ylabel("Actual",fontsize=11)
        plt.tight_layout(); st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Feature Importance")
        fi = get_all_feature_importances(model, fn)
        fig2, ax2 = plt.subplots(figsize=(5,4))
        colors = plt.cm.cool(np.linspace(0.2,0.8,len(fi)))
        ax2.barh(fi["Feature"], fi["Importance"], color=colors, height=0.65)
        ax2.invert_yaxis(); ax2.set_xlabel("Importance",fontsize=11)
        ax2.tick_params(axis="y",labelsize=8)
        for spine in ax2.spines.values(): spine.set_visible(False)
        ax2.grid(axis="x", alpha=0.15)
        plt.tight_layout(); st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Page: Chatbot ──
def page_chatbot():
    st.markdown("<p class='section-title'>🤖 CardioGuard Assistant</p><p class='section-sub'>Ask about heart health. General guidance only — not medical advice.</p>", unsafe_allow_html=True)

    RESP = {
        "risk": "🔴 **Common Risk Factors:**\n\n- High blood pressure & cholesterol\n- Smoking, excessive alcohol\n- Obesity, sedentary lifestyle\n- Diabetes, family history\n- Stress, poor sleep",
        "diet": "🥗 **Heart-Healthy Diet:**\n\n- More fruits, vegetables, whole grains\n- Lean proteins (fish, legumes)\n- Limit saturated fats, sodium, sugar\n- Use olive oil, eat nuts in moderation\n- Stay hydrated with water",
        "symptom": "🫀 **Warning Signs:**\n\n- Chest pain, tightness, pressure\n- Shortness of breath\n- Dizziness or fainting\n- Pain radiating to arm, jaw, back\n- Unusual fatigue, leg swelling\n\n**Seek emergency help for sudden severe symptoms.**",
        "exercise": "🏃 **Exercise Tips:**\n\n- 150 min moderate aerobic activity/week\n- Brisk walking, cycling, swimming\n- Strength training 2 days/week\n- Start slowly, increase gradually\n- Consult doctor before starting",
        "bp": "💉 **Blood Pressure:**\n\n- Normal: < 120/80\n- Elevated: 120–129 / < 80\n- High Stage 1: 130–139 / 80–89\n- High Stage 2: ≥ 140 / ≥ 90",
        "cholesterol": "🧪 **Cholesterol Levels:**\n\n- Total: < 200 mg/dL desirable\n- LDL: < 100 optimal\n- HDL: ≥ 60 protective\n- Triglycerides: < 150 normal",
        "stress": "🧘 **Stress Management:**\n\n- Deep breathing, meditation\n- Exercise regularly\n- Sleep 7–9 hours\n- Maintain social connections",
        "hello": "👋 Hello! Ask about **risk factors**, **diet**, **symptoms**, **exercise**, **blood pressure**, **cholesterol**, or **stress**.",
        "hi": "👋 Hi! How can I help with heart health? Try **risk**, **diet**, or **symptoms**.",
        "help": "ℹ️ Topics: **risk** · **diet** · **symptoms** · **exercise** · **bp** · **cholesterol** · **stress**",
    }
    DISC = "\n\n---\n⚕️ *This is not medical advice. Consult a qualified healthcare professional.*"

    st.markdown("<div class='glass' style='min-height:300px;max-height:500px;overflow-y:auto;padding:20px 24px;'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='display:flex;justify-content:flex-end;'><div class='chat-user'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{msg['content']}</div>", unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown("<p style='color:#475569;text-align:center;margin-top:40px;'>💬 Start a conversation about heart health...</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask about heart health…")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        lower = user_input.lower()
        response = None
        for kw, rsp in RESP.items():
            if kw in lower:
                response = rsp; break
        if response is None:
            response = "🤔 Try asking about **risk**, **diet**, **symptoms**, **exercise**, **bp**, **cholesterol**, or **stress**."
        response += DISC
        st.session_state.chat_history.append({"role":"assistant","content":response})
        st.rerun()


# ── Main ──
def main():
    st.set_page_config(page_title="CardioGuard AI", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_session()

    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:24px 0 8px;'>"
            "<p style='font-size:48px;margin:0;'>🫀</p>"
            "<h1 style='background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;"
            "-webkit-text-fill-color:transparent;font-size:24px;margin:4px 0 0;font-weight:900;letter-spacing:-0.5px;'>CardioGuard AI</h1>"
            "<p style='color:#475569;font-size:12px;margin-top:4px;letter-spacing:0.5px;'>Heart Disease Prediction</p>"
            "</div>", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("", ["🫀  Patient View","🩺  Doctor Dashboard","📊  Analytics","🤖  Chatbot"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("<div style='text-align:center;'><p style='color:#334155;font-size:10px;letter-spacing:0.5px;'>Powered by RandomForest ML<br>© 2026 CardioGuard AI</p></div>", unsafe_allow_html=True)

    if "Patient" in page: page_patient()
    elif "Doctor" in page: page_doctor()
    elif "Analytics" in page: page_analytics()
    elif "Chatbot" in page: page_chatbot()

if __name__ == "__main__":
    main()
