# src/ui.py
# ============================================================
# MedBot: Interactive Doctor-Patient Oncology AI Simulation
# Features:
#   1. Clinical Consultation Protocol (Answers + Inquiries + Precautions)
#   2. Real-Time Patient Medical Notes & Symptom Tracking
#   3. Contextual Memory Layer (Pronoun / Context resolution)
#   4. Bi-directional Simulation: Patient Mode & Doctor Mode
# ============================================================

import os
import sys
import json
from datetime import datetime
import streamlit as st

# Automatic .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
SRC  = os.path.abspath(os.path.dirname(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title = "🏥 Doctor-Patient Simulation",
    page_icon  = "🏥",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)


# ==================================================
# HISTORY FOLDER
# ==================================================

HISTORY_FOLDER = "chat_histories"
os.makedirs(HISTORY_FOLDER, exist_ok=True)


# ==================================================
# PATIENT CASES FOR DOCTOR MODE
# ==================================================

PATIENT_CASES = {
    "lung_case": {
        "title": "🫁 Robert Miller — Suspected Lung Cancer (Age 58)",
        "name": "Robert Miller",
        "age": 58,
        "gender": "Male",
        "cancer_type": "lung cancer",
        "stage": "Suspected NSCLC (Under Evaluation)",
        "chief_complaint": "Persistent dry cough for 3 months, progressive shortness of breath on exertion, rust-colored sputum, and 12 lb unintentional weight loss.",
        "medical_history": "35 pack-year tobacco smoking history, retired pipefitter, mild COPD.",
        "emotional_state": "Anxious and worried about lung cancer; hoping it is just chronic bronchitis.",
        "vitals": "BP: 138/86 mmHg | HR: 82 bpm | SpO2: 95% on room air"
    },
    "breast_case": {
        "title": "🎗️ Sarah Jenkins — Breast Lump Assessment (Age 46)",
        "name": "Sarah Jenkins",
        "age": 46,
        "gender": "Female",
        "cancer_type": "breast cancer",
        "stage": "Palpable Breast Mass (BIRADS-4/5 suspect)",
        "chief_complaint": "Firm, painless, non-mobile 2.5cm mass in upper-outer quadrant of right breast discovered during self-exam 3 weeks ago.",
        "medical_history": "Menarche at age 12, G1P1, maternal aunt diagnosed with premenopausal breast cancer at 44.",
        "emotional_state": "Frightened, tearful, seeking reassurance and clear immediate next steps.",
        "vitals": "BP: 124/78 mmHg | HR: 74 bpm | Afebrile"
    },
    "colon_case": {
        "title": "🩸 David Chen — Colorectal Concerns (Age 54)",
        "name": "David Chen",
        "age": 54,
        "gender": "Male",
        "cancer_type": "colorectal cancer",
        "stage": "Suspected Colorectal Neoplasm",
        "chief_complaint": "Intermittent rectal bleeding with dark maroon blood mixed in stool, alternating constipation and diarrhea for 2 months, severe fatigue.",
        "medical_history": "No prior screening colonoscopy, sedentary accountant, low-fiber Western diet.",
        "emotional_state": "Embarrassed about rectal symptoms; delayed visit for months but now exhausted.",
        "vitals": "BP: 118/72 mmHg | HR: 88 bpm | Mild pallor noted"
    },
    "prostate_case": {
        "title": "🧬 James Wilson — Prostate Symptoms (Age 67)",
        "name": "James Wilson",
        "age": 67,
        "gender": "Male",
        "cancer_type": "prostate cancer",
        "stage": "Under Investigation (Elevated PSA 8.4 ng/mL)",
        "chief_complaint": "Urinary hesitancy, weak stream, nocturia waking 4 times nightly, dull lower back ache.",
        "medical_history": "Hypertension, brother had prostate cancer at age 65.",
        "emotional_state": "Pragmatic, wanting to know if biopsy or surgery is urgently required.",
        "vitals": "BP: 142/88 mmHg | HR: 70 bpm | PSA: 8.4 ng/mL"
    },
    "custom_case": {
        "title": "🧪 Custom Cancer Case",
        "name": "Alex Taylor",
        "age": 52,
        "gender": "Patient",
        "cancer_type": "cancer",
        "stage": "Under Evaluation",
        "chief_complaint": "Experiencing new-onset uncharacteristic localized pain, abnormal fatigue, and unexplained weight loss.",
        "medical_history": "No major prior surgical history.",
        "emotional_state": "Searching for clear clinical answers and guidance.",
        "vitals": "BP: 126/80 mmHg | HR: 76 bpm"
    }
}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_greeting() -> str:
    hour = datetime.now().hour
    if   5  <= hour < 12 : return "Good Morning"
    elif 12 <= hour < 17 : return "Good Afternoon"
    else                  : return "Good Evening"


def save_chat(
    messages      : list,
    session_id    : str,
    patient_name  : str,
    doctor_name   : str = "Dr. Oncologist",
    mode          : str = "patient",
    patient_notes : dict = None
) -> str:

    if not messages:
        return ""

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c for c in patient_name if c.isalnum() or c == "_"
    ) or "Patient"
    filename  = f"chat_{safe_name}_{ts}.json"
    filepath  = os.path.join(HISTORY_FOLDER, filename)

    clean_messages = []
    for m in messages:
        entry = {
            "role"    : m.get("role",    ""),
            "content" : m.get("content", ""),
            "action"  : m.get("action",  ""),
            "mode"    : m.get("mode",    mode)
        }
        meta = m.get("metadata", {})
        if meta:
            try:
                hall = meta.get("hallucination", {})
                entry["metadata"] = {
                    "question_type"     : str(meta.get("question_type", "")),
                    "cancer_type"       : str(meta.get("cancer_type",   "")),
                    "action"            : str(meta.get("action",        "")),
                    "resolved_question" : str(meta.get("resolved_question", "")),
                    "was_resolved"      : bool(meta.get("was_resolved", False)),
                    "confidence"        : float(meta.get("confidence",  0.0)),
                    "score"             : float(hall.get("score",       0.0)),
                    "verdict"           : str(hall.get("verdict",       "")),
                    "safety"            : str(hall.get("safety",        "LOW")),
                }
            except Exception:
                pass
        clean_messages.append(entry)

    data = {
        "filename"      : filename,
        "session_id"    : session_id,
        "patient_name"  : patient_name,
        "doctor_name"   : doctor_name,
        "mode"          : mode,
        "patient_notes" : patient_notes or {},
        "saved_at"      : datetime.now().isoformat(),
        "total_msgs"    : len(clean_messages),
        "messages"      : clean_messages,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"  Save error: {e}")
        return ""


def load_all_chats() -> list:
    if not os.path.exists(HISTORY_FOLDER):
        return []

    chats = []
    for f in sorted(os.listdir(HISTORY_FOLDER), reverse=True):
        if not f.endswith(".json"):
            continue
        try:
            path = os.path.join(HISTORY_FOLDER, f)
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["filename"] = f
                chats.append(data)
        except Exception:
            continue

    return chats


# ==================================================
# CSS STYLES
# ==================================================

st.markdown("""
<style>
.main { background-color:#0d0d14; color:#ffffff; }

.main-header {
    text-align:center; padding:18px;
    background:linear-gradient(135deg,#1a2a4f 0%,#1a3a2a 100%);
    border-radius:14px; margin-bottom:16px;
    border:1px solid #2a3a5f;
}
.main-header h1 { font-size:1.8rem; font-weight:700; color:#fff; margin:0; }
.main-header p  { color:#aaaacc; margin:5px 0 0 0; font-size:.88rem; }

.patient-msg {
    background:#1a2a4f; border-left:4px solid #4a9eff;
    border-radius:10px; padding:12px 16px; margin:8px 0; color:#fff;
}
.patient-label { font-size:12px; color:#4a9eff; font-weight:600; margin-bottom:3px; }
.patient-text  { font-size:15px; color:#fff; margin:0; }

.doctor-msg {
    background:#13261a; border-left:4px solid #4caf50;
    border-radius:10px; padding:14px 18px; margin:8px 0; color:#fff;
}
.doctor-label { font-size:12px; color:#4caf50; font-weight:600; margin-bottom:4px; }

.notes-card {
    background:#16202c; border:1px solid #2a4060;
    border-radius:10px; padding:13px 16px; margin:8px 0;
}
.notes-title { font-size:13px; font-weight:700; color:#4a9eff; margin-bottom:5px; }
.notes-item  { font-size:12px; color:#d0d8e8; margin:2px 0; }

.patient-ai-msg {
    background:#16243b; border-left:4px solid #00bcd4;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}
.patient-ai-label { font-size:12px; color:#00bcd4; font-weight:600; margin-bottom:3px; }

.specialist-msg {
    background:#2b1b3d; border-left:4px solid #ba68c8;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}

.case-card {
    background:#131d2e; border:1px solid #2c446e;
    border-radius:10px; padding:14px 18px; margin-bottom:15px;
}
.case-title { font-size:15px; font-weight:700; color:#4a9eff; margin-bottom:6px; }
.case-detail { font-size:13px; color:#c0cce0; margin:3px 0; }

.reject-msg {
    background:#3a1a1a; border-left:4px solid #ff5252;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}

.chat-card {
    background:#13132a; border:1px solid #2a2a5f;
    border-radius:10px; padding:13px 15px; margin:6px 0;
}
.chat-card:hover { border-color:#4a9eff; }

.mem-badge {
    background:#1e1e3a; border:1px solid #9c27b0;
    border-radius:5px; padding:3px 9px; font-size:12px;
    color:#cc99ff; display:inline-block; margin-bottom:5px;
}
.mem-resolution-badge {
    background:#142814; border:1px solid #4caf50;
    border-radius:5px; padding:3px 8px; font-size:12px;
    color:#81c784; display:inline-block; margin-top:4px;
}
.src-card {
    background:#1e2a1e; border-left:3px solid #ff9800;
    border-radius:5px; padding:5px 10px; margin:3px 0;
    font-size:12px; color:#ccddcc;
}
.score-good {
    background:#1a3a2a; border:1px solid #4caf50; border-radius:7px;
    padding:7px; text-align:center; color:#4caf50; font-weight:bold;
}
.score-warn {
    background:#3a2a1a; border:1px solid #ff9800; border-radius:7px;
    padding:7px; text-align:center; color:#ff9800; font-weight:bold;
}
.score-bad {
    background:#3a1a1a; border:1px solid #ff5252; border-radius:7px;
    padding:7px; text-align:center; color:#ff5252; font-weight:bold;
}
.edit-box {
    background:#1e1e2e; border:1px solid #4a9eff;
    border-radius:8px; padding:10px; margin:5px 0;
}

hr { border-color:#2a2a3a !important; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD AGENT
# ==================================================

@st.cache_resource
def load_agent():
    try:
        from agents.query_agent import QueryAgent
    except ImportError:
        from src.agents.query_agent import QueryAgent
    return QueryAgent()


# ==================================================
# SESSION STATE INIT
# ==================================================

def _new_sid() -> str:
    return f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def init_session():
    if "initialized" not in st.session_state:
        st.session_state.initialized           = True
        st.session_state.agent                 = None
        st.session_state.session_id            = _new_sid()
        st.session_state.messages              = []
        st.session_state.patient_notes         = {"symptoms": [], "duration": "Not specified"}
        st.session_state.total_queries         = 0
        st.session_state.total_passed          = 0
        st.session_state.all_scores            = []
        st.session_state.patient_name          = "Patient"
        st.session_state.doctor_name           = "Dr. Giridhari"
        st.session_state.sim_mode              = "patient"     # 'patient', 'doctor', 'doctor_qa'
        st.session_state.doctor_response_style = "auto"
        st.session_state.patient_case          = "lung_case"
        st.session_state.last_cancer           = ""
        st.session_state.turn_count            = 0
        st.session_state.editing_idx           = None
        st.session_state.quick_q               = None
        st.session_state.page                  = "chat"
        st.session_state.view_hist_data        = None
        st.session_state.save_msg              = ""


# ==================================================
# NEW CHAT RESET
# ==================================================

def start_new_chat():
    if st.session_state.get("messages"):
        save_chat(
            messages      = st.session_state.messages,
            session_id    = st.session_state.session_id,
            patient_name  = st.session_state.patient_name,
            doctor_name   = st.session_state.doctor_name,
            mode          = st.session_state.sim_mode,
            patient_notes = st.session_state.get("patient_notes", {})
        )
        if st.session_state.get("agent"):
            try:
                st.session_state.agent.clear_memory(
                    st.session_state.session_id
                )
            except Exception:
                pass

    st.session_state.session_id    = _new_sid()
    st.session_state.messages      = []
    st.session_state.patient_notes = {"symptoms": [], "duration": "Not specified"}
    st.session_state.total_queries = 0
    st.session_state.total_passed  = 0
    st.session_state.all_scores    = []
    st.session_state.last_cancer   = ""
    st.session_state.turn_count    = 0
    st.session_state.editing_idx   = None
    st.session_state.quick_q       = None
    st.session_state.page          = "chat"
    st.session_state.save_msg      = "✅ Previous consultation saved!"


# ==================================================
# PROCESS INQUIRY
# ==================================================

def process_question(
    question : str,
    agent,
    is_edit  : bool = False,
    edit_idx : int  = None
):
    mode = st.session_state.sim_mode
    active_case = PATIENT_CASES.get(st.session_state.patient_case, PATIENT_CASES["lung_case"])

    if is_edit and edit_idx is not None:
        st.session_state.messages      = st.session_state.messages[:edit_idx]
        st.session_state.total_queries = len(st.session_state.messages) // 2

    user_role = "doctor" if mode in ["doctor", "doctor_qa"] else "patient"
    user_name = st.session_state.doctor_name if user_role == "doctor" else st.session_state.patient_name

    st.session_state.messages.append({
        "role"      : user_role,
        "name"      : user_name,
        "content"   : question,
        "is_edit"   : is_edit,
        "mode"      : mode,
    })

    spinner_label = "👤 Patient thinking..." if mode == "doctor" else "🩺 Doctor AI reviewing clinical oncology guidelines..."

    with st.spinner(spinner_label):
        try:
            result = agent.process(
                question              = question,
                session_id            = st.session_state.session_id,
                mode                  = mode,
                patient_profile       = active_case if mode == "doctor" else None,
                doctor_response_style = st.session_state.doctor_response_style
            )
            answer      = result.get("answer", "")
            qtype       = result.get("question_type", "general")
            cancer      = result.get("cancer_type", "")
            action      = result.get("action", "answer")
            notes       = result.get("patient_notes", {})

            if notes:
                st.session_state.patient_notes = notes

            st.session_state.total_queries += 1
            st.session_state.turn_count    += 1

            if cancer and cancer not in ["cancer", "N/A", ""]:
                st.session_state.last_cancer = cancer

            if qtype != "non_medical":
                score = result.get("hallucination", {}).get("score", 0.0)
                st.session_state.all_scores.append(score)
                if not result.get("hallucination", {}).get("is_hallucinated", False):
                    st.session_state.total_passed += 1

            ai_role = "patient_ai" if mode == "doctor" else ("specialist" if mode == "doctor_qa" else "doctor")
            ai_name = active_case.get("name", "Patient AI") if mode == "doctor" else ("Oncology Specialist AI" if mode == "doctor_qa" else "Doctor AI")

            st.session_state.messages.append({
                "role"     : ai_role,
                "name"     : ai_name,
                "content"  : answer,
                "action"   : action,
                "metadata" : result,
                "mode"     : mode,
            })

        except Exception as e:
            err_role = "patient_ai" if mode == "doctor" else "doctor"
            st.session_state.messages.append({
                "role"    : err_role,
                "name"    : "Simulation Assistant",
                "content" : f"An error occurred: {e}. Please check connection or enter a Groq API key in the sidebar.",
            })

    st.session_state.editing_idx = None
    st.rerun()


# ==================================================
# SIDEBAR
# ==================================================

def render_sidebar():

    with st.sidebar:

        # Header Logo
        st.markdown("""
        <div style='text-align:center;padding:6px 0 4px;'>
        <h2 style='color:#4caf50;margin:0;'>🏥 MedBot</h2>
        <p style='color:#aaaacc;font-size:12px;margin:2px 0 0;'>
        Cancer QA · Doctor-Patient Clinical Simulation
        </p></div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── SIMULATION MODE SELECTOR ──────────────────
        st.markdown("**🎭 Simulation Mode**")
        mode_options = {
            "patient"   : "👤 Patient Mode (Consult Doctor AI)",
            "doctor"    : "🩺 Doctor Mode (Interview Patient AI)",
            "doctor_qa" : "🔬 Clinical Copilot (Physician QA)"
        }
        
        current_mode = st.session_state.sim_mode
        selected_mode = st.radio(
            "Select Role / Mode",
            options = list(mode_options.keys()),
            format_func = lambda x: mode_options[x],
            index = list(mode_options.keys()).index(current_mode) if current_mode in mode_options else 0,
            label_visibility = "collapsed"
        )
        if selected_mode != current_mode:
            st.session_state.sim_mode = selected_mode
            st.rerun()

        # ── PATIENT CASE SELECTION (FOR DOCTOR MODE) ──
        if st.session_state.sim_mode == "doctor":
            st.markdown("**🗂️ Patient Case Profile**")
            case_keys = list(PATIENT_CASES.keys())
            case_titles = [PATIENT_CASES[k]["title"] for k in case_keys]
            curr_case_idx = case_keys.index(st.session_state.patient_case) if st.session_state.patient_case in case_keys else 0
            
            chosen_title = st.selectbox(
                "Patient Case",
                options = case_titles,
                index = curr_case_idx,
                label_visibility = "collapsed"
            )
            chosen_key = case_keys[case_titles.index(chosen_title)]
            if chosen_key != st.session_state.patient_case:
                st.session_state.patient_case = chosen_key
                st.rerun()

        st.divider()

        # ── NEW CHAT BUTTON ───────────────────────────
        if st.button(
            "➕  New Consultation",
            key                 = "sb_new",
            type                = "primary",
            use_container_width = True
        ):
            start_new_chat()
            st.rerun()

        if st.session_state.get("save_msg"):
            st.success(st.session_state.save_msg)
            st.session_state.save_msg = ""

        # ── SAVE CHAT ─────────────────────────────────
        has_msgs = len(st.session_state.get("messages", [])) > 0
        if st.button(
            "💾  Save Consultation",
            key                 = "sb_save",
            use_container_width = True,
            disabled            = not has_msgs
        ):
            fname = save_chat(
                messages      = st.session_state.messages,
                session_id    = st.session_state.session_id,
                patient_name  = st.session_state.patient_name,
                doctor_name   = st.session_state.doctor_name,
                mode          = st.session_state.sim_mode,
                patient_notes = st.session_state.patient_notes
            )
            if fname:
                st.session_state.save_msg = "✅ Saved successfully!"
                st.rerun()
            else:
                st.error("Save failed.")

        # ── VIEW HISTORY ──────────────────────────────
        if st.button(
            "📂  Consultation History",
            key                 = "sb_history",
            use_container_width = True
        ):
            st.session_state.page          = "history"
            st.session_state.view_hist_data = None
            st.rerun()

        if st.session_state.page != "chat":
            if st.button(
                "💬  Back to Room",
                key                 = "sb_back",
                use_container_width = True
            ):
                st.session_state.page          = "chat"
                st.session_state.view_hist_data = None
                st.rerun()

        st.divider()

        # ── API CONFIGURATION ─────────────────────────
        st.markdown("**🔑 Groq API Key (Optional)**")
        current_key = os.environ.get("GROQ_API_KEY", "")
        key_input = st.text_input(
            "groq_api_key_ui",
            value            = current_key,
            type             = "password",
            label_visibility = "collapsed",
            placeholder      = "Paste Groq API Key (gsk_...)"
        )
        if key_input and key_input.strip() != current_key:
            os.environ["GROQ_API_KEY"] = key_input.strip()
            st.success("✅ Groq API Key connected!")

        st.divider()

        # ── IDENTITY SETTINGS ─────────────────────────
        if st.session_state.sim_mode in ["doctor", "doctor_qa"]:
            st.markdown("**🩺 Doctor Name**")
            doc_in = st.text_input(
                "doctor_name_input",
                value = st.session_state.doctor_name,
                label_visibility = "collapsed",
                placeholder = "Dr. Name..."
            )
            if doc_in: st.session_state.doctor_name = doc_in
        else:
            st.markdown("**👤 Patient Name**")
            p_in = st.text_input(
                "patient_name_input",
                value = st.session_state.patient_name,
                label_visibility = "collapsed",
                placeholder = "Patient Name..."
            )
            if p_in: st.session_state.patient_name = p_in

        st.divider()

        # ── QUICK PROMPTS ─────────────────────────────
        if st.session_state.page == "chat":
            st.markdown("**💡 Quick Prompts**")
            
            if st.session_state.sim_mode == "doctor":
                quick_qs = [
                    "How long have you had this cough?",
                    "Have you coughed up any blood or rust-colored phlegm?",
                    "Do you have a personal or family history of smoking?",
                    "Let's schedule a low-dose chest CT and sputum test.",
                ]
            elif st.session_state.sim_mode == "doctor_qa":
                quick_qs = [
                    "What is first-line therapy for metastatic NSCLC with PD-L1 > 50%?",
                    "What are the NCCN adjuvant criteria for Stage III colon cancer?",
                    "Explain the management of immune checkpoint inhibitor toxicities.",
                ]
            else:
                quick_qs = [
                    "What is lung cancer?",
                    "What are symptoms for this?",
                    "How do I recover from this quickly?",
                    "What is brain cancer?",
                    "What are symptoms of breast cancer?",
                ]

            for q in quick_qs:
                if st.button(
                    q,
                    key                 = f"qq_{abs(hash(q))}",
                    use_container_width = True
                ):
                    st.session_state.quick_q = q
                    st.rerun()

            st.divider()

        st.caption("⚠️ Medical educational & clinical consultation simulation.")


# ==================================================
# RENDER MESSAGES
# ==================================================

def render_messages(agent):

    for idx, msg in enumerate(st.session_state.messages):

        role    = msg.get("role", "")
        name    = msg.get("name", "")
        content = msg.get("content", "")
        meta    = msg.get("metadata", {})
        is_edit = msg.get("is_edit", False)

        # ── PATIENT USER MESSAGE ───────────────────────
        if role == "patient":
            is_editing = st.session_state.editing_idx == idx
            if is_editing:
                st.markdown("<div class='edit-box'>", unsafe_allow_html=True)
                edited = st.text_area("Edit inquiry:", value=content, key=f"ea_{idx}", height=75)
                cs, cc = st.columns(2)
                with cs:
                    if st.button("✅ Resend", key=f"esave_{idx}", type="primary", use_container_width=True):
                        if edited.strip():
                            process_question(edited.strip(), agent, is_edit=True, edit_idx=idx)
                with cc:
                    if st.button("❌ Cancel", key=f"ecancel_{idx}", use_container_width=True):
                        st.session_state.editing_idx = None
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                cm, cb = st.columns([11, 1])
                with cm:
                    if is_edit:
                        st.markdown("<div class='edited-tag'>✏️ Edited</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='patient-msg'>"
                        f"<div class='patient-label'>👤 {name or st.session_state.patient_name}</div>"
                        f"<div class='patient-text'>{content}</div></div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.write("")
                    if st.button("✏️", key=f"eb_{idx}", help="Edit"):
                        st.session_state.editing_idx = idx
                        st.rerun()

        # ── DOCTOR USER MESSAGE (DOCTOR MODE) ─────────
        elif role == "doctor" and not meta:
            is_editing = st.session_state.editing_idx == idx
            if is_editing:
                st.markdown("<div class='edit-box'>", unsafe_allow_html=True)
                edited = st.text_area("Edit inquiry:", value=content, key=f"ea_{idx}", height=75)
                cs, cc = st.columns(2)
                with cs:
                    if st.button("✅ Resend", key=f"esave_{idx}", type="primary", use_container_width=True):
                        if edited.strip():
                            process_question(edited.strip(), agent, is_edit=True, edit_idx=idx)
                with cc:
                    if st.button("❌ Cancel", key=f"ecancel_{idx}", use_container_width=True):
                        st.session_state.editing_idx = None
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                cm, cb = st.columns([11, 1])
                with cm:
                    if is_edit:
                        st.markdown("<div class='edited-tag'>✏️ Edited</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='doctor-msg'>"
                        f"<div class='doctor-label'>🩺 {name or st.session_state.doctor_name}</div>"
                        f"<div style='font-size:15px;color:#fff;'>{content}</div></div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.write("")
                    if st.button("✏️", key=f"eb_{idx}", help="Edit"):
                        st.session_state.editing_idx = idx
                        st.rerun()

        # ── PATIENT AI RESPONSE (DOCTOR MODE) ─────────
        elif role == "patient_ai":
            st.markdown(
                f"<div class='patient-ai-msg'>"
                f"<div class='patient-ai-label'>👤 {name} (Simulated Patient)</div>"
                f"<div style='font-size:15px;color:#fff;'>{content}</div></div>",
                unsafe_allow_html=True
            )
            if meta:
                _render_meta(meta, is_patient_ai=True)

        # ── DOCTOR AI RESPONSE (PATIENT MODE) ─────────
        elif role == "doctor":
            qtype = meta.get("question_type", "")
            if qtype == "non_medical":
                css  = "reject-msg"
                icon = "⚠️ MedBot"
            else:
                css  = "doctor-msg"
                icon = "🩺 Doctor AI"

            # Render formatted response
            formatted_content = content.replace("\n", "<br>")
            st.markdown(
                f"<div class='{css}'>"
                f"<b>{icon}:</b><br>{formatted_content}</div>",
                unsafe_allow_html=True
            )
            if meta and qtype != "non_medical":
                _render_meta(meta)

        # ── SPECIALIST COPILOT RESPONSE ───────────────
        elif role == "specialist":
            st.markdown(
                f"<div class='specialist-msg'>"
                f"<b>🔬 Oncology Specialist Copilot:</b><br>{content}</div>",
                unsafe_allow_html=True
            )
            if meta:
                _render_meta(meta)


# ==================================================
# RENDER METADATA EXPANDER
# ==================================================

def _render_meta(meta: dict, is_patient_ai: bool = False):

    header_label = "📋 Clinical Case Simulation Insights" if is_patient_ai else "📋 Consultation Diagnostics & Knowledge Grounding"

    with st.expander(header_label, expanded=False):

        score        = meta.get("hallucination", {}).get("score", 0.0)
        verdict      = meta.get("hallucination", {}).get("verdict", "")
        cancer       = meta.get("cancer_type", "")
        resolved     = meta.get("resolved_question", "")
        original     = meta.get("question", "")
        was_resolved = meta.get("was_resolved", False)
        mem_ctx      = meta.get("memory_context", "")

        c1, c2, c3 = st.columns(3)
        with c1:
            if score >= 4.0:
                st.markdown(f"<div class='score-good'>✅ {score}/5</div>", unsafe_allow_html=True)
            elif score >= 3.0:
                st.markdown(f"<div class='score-warn'>⚠️ {score}/5</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='score-bad'>❌ {score}/5</div>", unsafe_allow_html=True)
        with c2:
            st.metric("Clinical Intent", meta.get("question_type", "?").upper()[:14])
        with c3:
            st.metric("Quality Score", meta.get("confidence", 5.0))

        if was_resolved and resolved != original:
            st.markdown(
                f"<div class='mem-resolution-badge'>"
                f"🧠 <b>Context Resolved:</b> <i>'{original}'</i> → <b>'{resolved}'</b>"
                f"</div>",
                unsafe_allow_html=True
            )

        if cancer and cancer not in ["cancer", ""]:
            st.markdown(f"<div class='mem-badge'>🎯 CANCER TOPIC: {cancer.upper()}</div>", unsafe_allow_html=True)

        if "PASS" in str(verdict):
            st.success(f"✅ {verdict}")
        elif "FAIL" in str(verdict):
            st.error(f"❌ {verdict}")

        if mem_ctx:
            with st.expander("🧠 Conversation Dialogue History"):
                st.text(mem_ctx)

        sources = meta.get("sources", [])
        if sources:
            st.markdown("**📚 Oncology Textbook References (45,384 Knowledge Records):**")
            for i, s in enumerate(sources):
                st.markdown(
                    f"<div class='src-card'>{i+1}. {s.get('source','?')} — Score: {s.get('score',0)}</div>",
                    unsafe_allow_html=True
                )


# ==================================================
# CHAT PAGE
# ==================================================

def page_chat(agent):

    mode = st.session_state.sim_mode
    active_case = PATIENT_CASES.get(st.session_state.patient_case, PATIENT_CASES["lung_case"])

    if mode == "doctor":
        # Doctor Simulation Header & Medical Chart
        st.markdown(f"### 🩺 Clinical Examination Room — {st.session_state.doctor_name}")
        
        with st.container():
            st.markdown(
                f"""<div class='case-card'>
                <div class='case-title'>📋 PATIENT MEDICAL CHART: {active_case['name']} ({active_case['age']} y/o {active_case['gender']})</div>
                <div class='case-detail'><b>Condition:</b> {active_case['cancer_type'].upper()} ({active_case['stage']})</div>
                <div class='case-detail'><b>Chief Complaint:</b> {active_case['chief_complaint']}</div>
                <div class='case-detail'><b>Medical History:</b> {active_case['medical_history']}</div>
                <div class='case-detail'><b>Vitals:</b> {active_case.get('vitals', 'Stable')}</div>
                <div class='case-detail'><b>Patient Mood:</b> <i>{active_case.get('emotional_state', '')}</i></div>
                </div>""",
                unsafe_allow_html=True
            )

        if not st.session_state.messages:
            st.markdown(
                f"""<div class='patient-ai-msg'>
                <b>👤 {active_case['name']} (Patient):</b><br><br>
                <i>"Hello Doctor... Thank you for seeing me today. I've been feeling really uneasy about these symptoms and I'm hoping you can help me figure out what is going on."</i>
                </div>""",
                unsafe_allow_html=True
            )

    elif mode == "doctor_qa":
        st.markdown(f"### 🔬 Oncology Clinical Copilot — {st.session_state.doctor_name}")
        if not st.session_state.messages:
            st.markdown(
                """<div class='specialist-msg'>
                <b>🔬 Oncology Specialist Copilot:</b><br><br>
                Welcome, Doctor. I am your oncology decision copilot backed by 25 oncology textbooks and NCCN/ESMO guidelines.
                Ask any question regarding staging, molecular biomarkers, systemic regimens, toxicity management, or clinical trials.
                </div>""",
                unsafe_allow_html=True
            )

    else:
        # Patient Mode
        greeting = get_greeting()
        pname    = st.session_state.patient_name
        st.markdown("### 💬 Interactive Oncology Consultation Room")
        if not st.session_state.messages:
            st.markdown(
                f"""<div class='doctor-msg'>
                <b>🩺 Doctor AI:</b><br><br>
                <b>{greeting}!</b> {"Welcome, " + pname + "!" if pname != "Patient" else "Welcome!"}<br><br>
                I am your dedicated oncology AI physician assistant, backed by 25 oncology textbooks.
                Feel free to ask me anything about cancer, symptoms, staging, or treatments — or tell me about any symptoms you are feeling so I can note them down and provide tailored precautions.<br><br>
                <i>Please share what is on your mind. How can I help you today?</i>
                </div>""",
                unsafe_allow_html=True
            )

    render_messages(agent)

    st.divider()

    # Input Box Placeholder
    if mode == "doctor":
        placeholder = f"Ask {active_case['name']} a clinical question (e.g. 'When did the cough start?', 'Any blood in sputum?')..."
    elif mode == "doctor_qa":
        placeholder = "Ask clinical oncology questions (e.g. 'Adjuvant therapy for Stage 3 colon cancer')..."
    else:
        placeholder = "Ask about cancer, symptoms, or share your symptoms... (Enter to send)"

    user_input = st.chat_input(placeholder)

    if st.session_state.quick_q:
        q = st.session_state.quick_q
        st.session_state.quick_q = None
        process_question(q, agent)

    if user_input and user_input.strip():
        process_question(user_input.strip(), agent)


# ==================================================
# HISTORY LIST PAGE
# ==================================================

def page_history():

    st.markdown("## 📂 Saved Consultation Histories")
    chats = load_all_chats()

    if not chats:
        st.info("No saved consultations yet. Complete a consultation and click **💾 Save Consultation**.")
        if st.button("💬 Go to Consultation Room", type="primary"):
            st.session_state.page = "chat"
            st.rerun()
        return

    search = st.text_input(
        "hist_search",
        placeholder      = "🔍 Search by patient name or date...",
        label_visibility = "collapsed"
    )

    filtered = [
        c for c in chats
        if not search or search.lower() in (c.get("patient_name", "") + c.get("saved_at", "")).lower()
    ]

    st.markdown(f"**{len(filtered)} consultation(s)**" + (f" (filtered from {len(chats)})" if search else ""))
    st.divider()

    for chat in filtered:
        filename = chat.get("filename", "")
        name     = chat.get("patient_name", "Patient")
        doc_name = chat.get("doctor_name", "Doctor")
        saved_at = chat.get("saved_at", "")[:19].replace("T", " ")
        total    = chat.get("total_msgs", 0)
        mode     = chat.get("mode", "patient")
        msgs     = chat.get("messages", [])

        first_q = "—"
        for m in msgs:
            if m.get("content"):
                first_q = m.get("content")[:90]
                break

        mode_badge = "🩺 Doctor Interview" if mode == "doctor" else ("🔬 Clinical Copilot" if mode == "doctor_qa" else "👤 Patient Consultation")

        st.markdown(
            f"""<div class='chat-card'>
            <span style='background:#1e2a3a;color:#4a9eff;padding:2px 8px;border-radius:4px;font-size:11px;'>{mode_badge}</span>&nbsp;
            <b style='color:#ffffff;'>👤 {name} &nbsp;|&nbsp; 🩺 {doc_name}</b>
            &nbsp;
            <span style='color:#888899;font-size:12px;'>🕐 {saved_at} &nbsp;|&nbsp; 💬 {total} messages</span><br>
            <span style='color:#aaaacc;font-size:13px;'>{first_q}...</span></div>""",
            unsafe_allow_html=True
        )

        cv, cc, cd = st.columns([2, 2, 1])
        with cv:
            if st.button("👁️ View", key=f"hv_{filename}", use_container_width=True):
                st.session_state.view_hist_data = chat
                st.session_state.page           = "view_chat"
                st.rerun()

        with cc:
            if st.button("▶️ Continue", key=f"hc_{filename}", use_container_width=True, type="primary"):
                st.session_state.messages      = msgs
                st.session_state.patient_name  = name
                st.session_state.doctor_name   = doc_name
                st.session_state.sim_mode      = mode
                st.session_state.session_id    = chat.get("session_id", _new_sid())
                st.session_state.total_queries = len(msgs) // 2
                st.session_state.turn_count    = len(msgs)
                st.session_state.page          = "chat"
                st.session_state.view_hist_data= None
                st.rerun()

        with cd:
            if st.button("🗑️", key=f"hd_{filename}", help="Delete"):
                path = os.path.join(HISTORY_FOLDER, filename)
                try:
                    os.remove(path)
                    st.rerun()
                except Exception:
                    st.error("Could not delete.")

        st.divider()


# ==================================================
# VIEW SINGLE CHAT PAGE
# ==================================================

def page_view_chat():

    chat = st.session_state.view_hist_data
    if not chat:
        st.session_state.page = "history"
        st.rerun()
        return

    name     = chat.get("patient_name", "Patient")
    doc_name = chat.get("doctor_name", "Doctor")
    saved_at = chat.get("saved_at", "")[:19].replace("T", " ")
    total    = chat.get("total_msgs", 0)
    msgs     = chat.get("messages", [])
    mode     = chat.get("mode", "patient")

    cb, cc = st.columns([1, 1])
    with cb:
        if st.button("← Back", key="vcb_back", type="secondary"):
            st.session_state.page           = "history"
            st.session_state.view_hist_data = None
            st.rerun()

    with cc:
        if st.button("▶️ Continue This Consultation", key="vcb_cont", type="primary"):
            st.session_state.messages      = msgs
            st.session_state.patient_name  = name
            st.session_state.doctor_name   = doc_name
            st.session_state.sim_mode      = mode
            st.session_state.session_id    = chat.get("session_id", _new_sid())
            st.session_state.total_queries = len(msgs) // 2
            st.session_state.turn_count    = len(msgs)
            st.session_state.page          = "chat"
            st.session_state.view_hist_data= None
            st.rerun()

    st.markdown(f"## 📄 Consultation Transcript: {name} & {doc_name}")
    st.caption(f"🕐 Saved: {saved_at}  |  💬 {total} messages  |  Mode: {mode.upper()}")
    st.divider()

    for msg in msgs:
        role    = msg.get("role", "")
        mname   = msg.get("name", "")
        content = msg.get("content", "")

        if role == "patient":
            st.markdown(f"<div class='patient-msg'><div class='patient-label'>👤 {mname or name}</div><div class='patient-text'>{content}</div></div>", unsafe_allow_html=True)
        elif role == "patient_ai":
            st.markdown(f"<div class='patient-ai-msg'><div class='patient-ai-label'>👤 {mname or name} (Patient AI)</div><div style='font-size:15px;color:#fff;'>{content}</div></div>", unsafe_allow_html=True)
        elif role == "doctor":
            st.markdown(f"<div class='doctor-msg'><b>🩺 {mname or doc_name}:</b><br>{content.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        elif role == "specialist":
            st.markdown(f"<div class='specialist-msg'><b>🔬 Oncology Specialist AI:</b><br>{content}</div>", unsafe_allow_html=True)


# ==================================================
# MAIN
# ==================================================

def main():

    init_session()
    render_sidebar()

    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 Doctor-Patient Oncology Simulation</h1>
        <p>Powered by Agentic RAG + Clinical Consultation Protocol + LLaMA3 + 25 Oncology Textbooks</p>
        <p style='color:#888899;font-size:12px;'>
        ⚠️ For educational & training simulation only. Always consult a certified oncologist.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load agent
    if st.session_state.agent is None:
        with st.spinner("🔄 Initializing Oncology Medical AI Engine..."):
            try:
                st.session_state.agent = load_agent()
                st.success("✅ Oncology AI ready!")
            except Exception as e:
                st.error(f"❌ Initialization error: {e}")
                st.stop()

    agent = st.session_state.agent

    # Layout
    chat_col, info_col = st.columns([3, 1])

    with chat_col:
        page = st.session_state.page
        if page == "chat":
            page_chat(agent)
        elif page == "history":
            page_history()
        elif page == "view_chat":
            page_view_chat()

    # Info Column
    with info_col:

        # ── DOCTOR'S CLINICAL NOTES ───────────────────
        st.markdown("### 📋 Clinical Patient Chart")
        pnotes = st.session_state.patient_notes
        symptoms_list = pnotes.get("symptoms", [])
        
        st.markdown(
            f"""<div class='notes-card'>
            <div class='notes-title'>📝 ACTIVE CONSULTATION NOTES</div>
            <div class='notes-item'><b>Suspected Focus:</b> {st.session_state.last_cancer.upper() if st.session_state.last_cancer else 'General Evaluation'}</div>
            <div class='notes-item'><b>Reported Symptoms:</b> {', '.join(symptoms_list) if symptoms_list else 'None reported yet'}</div>
            <div class='notes-item'><b>Duration:</b> {pnotes.get('duration', 'Not specified')}</div>
            <div class='notes-item'><b>Status:</b> {'⚠️ Action Recommended' if symptoms_list else '🟢 Information Gathering'}</div>
            </div>""",
            unsafe_allow_html=True
        )

        st.divider()

        # ── STATS ─────────────────────────────────────
        st.markdown("### 📊 Live Stats")
        total  = st.session_state.total_queries
        passed = st.session_state.total_passed
        scores = st.session_state.all_scores

        st.metric("Consultation Turns", total)
        st.metric("Passed Verification", f"{passed}/{total}" if total > 0 else "0/0")

        if scores:
            avg = round(sum(scores)/len(scores), 2)
            st.metric("Avg Clinical Score", f"{avg}/5.0")
            st.progress(min(avg/5.0, 1.0))

        st.divider()

        # ── SAVED CONSULTATIONS ───────────────────────
        st.markdown("### 💾 Saved Records")
        chats = load_all_chats()
        st.metric("Total Records", len(chats))
        if chats:
            if st.button("📂 View All Records", key="info_hist", use_container_width=True):
                st.session_state.page = "history"
                st.rerun()

        st.divider()

        # ── ABOUT ─────────────────────────────────────
        st.markdown("### 🔬 Knowledge Base")
        st.markdown("""
        **Data Sources:**
        - 25 Oncology Textbooks
        - 45,384 Knowledge Chunks

        **Clinical Features:**
        - 🩺 Interactive Medical Consultation
        - 📋 Real-Time Symptom & Note Tracking
        - 🛡️ Evidence-Based Precautions & Next Steps
        - 🧠 Pronoun & Context Memory
        """)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()
