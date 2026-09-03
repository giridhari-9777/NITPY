# src/ui.py
# ============================================================
# MedBot: Doctor-Patient Oncology AI Simulation
# Clean UI with Integrated Contextual Memory & RAG
# ============================================================

import os
import sys
import json
from datetime import datetime
import streamlit as st

# Get absolute path of project root and src
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
# HELPER FUNCTIONS
# ==================================================

def get_greeting() -> str:
    hour = datetime.now().hour
    if   5  <= hour < 12 : return "Good Morning"
    elif 12 <= hour < 17 : return "Good Afternoon"
    else                  : return "Good Evening"


def save_chat(
    messages     : list,
    session_id   : str,
    patient_name : str
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
        "filename"     : filename,
        "session_id"   : session_id,
        "patient_name" : patient_name,
        "saved_at"     : datetime.now().isoformat(),
        "total_msgs"   : len(clean_messages),
        "messages"     : clean_messages,
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
    text-align:center; padding:22px;
    background:linear-gradient(135deg,#1a2a4f 0%,#1a3a2a 100%);
    border-radius:14px; margin-bottom:18px;
    border:1px solid #2a3a5f;
}
.main-header h1 { font-size:2rem; font-weight:700; color:#fff; margin:0; }
.main-header p  { color:#aaaacc; margin:6px 0 0 0; font-size:.9rem; }

.patient-msg {
    background:#1a2a4f; border-left:4px solid #4a9eff;
    border-radius:10px; padding:12px 16px; margin:8px 0; color:#fff;
}
.patient-label { font-size:12px; color:#4a9eff; font-weight:600; margin-bottom:3px; }
.patient-text  { font-size:15px; color:#fff; margin:0; }

.doctor-msg {
    background:#1a3a2a; border-left:4px solid #4caf50;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}
.doctor-q-msg {
    background:#223828; border-left:4px solid #8bc34a;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}
.followup-msg {
    background:#2a1a3a; border-left:4px solid #9c27b0;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}
.reject-msg {
    background:#3a1a1a; border-left:4px solid #ff5252;
    border-radius:10px; padding:13px 17px; margin:8px 0; color:#fff;
}

.chat-card {
    background:#13132a; border:1px solid #2a2a5f;
    border-radius:10px; padding:13px 15px; margin:6px 0;
}
.chat-card:hover { border-color:#4a9eff; }

.edited-tag {
    background:#2a1a0a; border:1px solid #ff9800;
    border-radius:4px; padding:1px 7px; font-size:11px;
    color:#ff9800; display:inline-block; margin-bottom:3px;
}
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
        st.session_state.initialized    = True
        st.session_state.agent          = None
        st.session_state.session_id     = _new_sid()
        st.session_state.messages       = []
        st.session_state.total_queries  = 0
        st.session_state.total_passed   = 0
        st.session_state.all_scores     = []
        st.session_state.patient_name   = "Patient"
        st.session_state.last_cancer    = ""
        st.session_state.turn_count     = 0
        st.session_state.editing_idx    = None
        st.session_state.quick_q        = None
        st.session_state.page           = "chat"
        st.session_state.view_hist_data = None
        st.session_state.save_msg       = ""


# ==================================================
# NEW CHAT RESET
# ==================================================

def start_new_chat():
    if st.session_state.get("messages"):
        save_chat(
            messages     = st.session_state.messages,
            session_id   = st.session_state.session_id,
            patient_name = st.session_state.patient_name
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
    if is_edit and edit_idx is not None:
        st.session_state.messages      = st.session_state.messages[:edit_idx]
        st.session_state.total_queries = len(st.session_state.messages) // 2

    st.session_state.messages.append({
        "role"    : "patient",
        "content" : question,
        "is_edit" : is_edit,
    })

    with st.spinner("🩺 Doctor AI consulting knowledge base & memory..."):
        try:
            result = agent.process(
                question              = question,
                session_id            = st.session_state.session_id,
                mode                  = "patient",
                doctor_response_style = "auto"
            )
            answer = result.get("answer", "")
            qtype  = result.get("question_type", "general")
            cancer = result.get("cancer_type", "")
            action = result.get("action", "answer")

            st.session_state.total_queries += 1
            st.session_state.turn_count    += 1

            if cancer and cancer not in ["cancer", "N/A", ""]:
                st.session_state.last_cancer = cancer

            if qtype != "non_medical":
                score = result.get("hallucination", {}).get("score", 0.0)
                st.session_state.all_scores.append(score)
                if not result.get("hallucination", {}).get("is_hallucinated", False):
                    st.session_state.total_passed += 1

            st.session_state.messages.append({
                "role"     : "doctor",
                "content"  : answer,
                "action"   : action,
                "metadata" : result,
            })

        except Exception as e:
            st.session_state.messages.append({
                "role"    : "doctor",
                "content" : f"I encountered an error processing your inquiry: {e}. Please try again.",
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
        Cancer QA · Doctor-Patient AI
        </p></div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── NEW CHAT BUTTON ───────────────────────────
        if st.button(
            "➕  New Chat",
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
            "💾  Save Chat",
            key                 = "sb_save",
            use_container_width = True,
            disabled            = not has_msgs
        ):
            fname = save_chat(
                messages     = st.session_state.messages,
                session_id   = st.session_state.session_id,
                patient_name = st.session_state.patient_name
            )
            if fname:
                st.session_state.save_msg = "✅ Saved!"
                st.rerun()
            else:
                st.error("Save failed.")

        # ── VIEW HISTORY ──────────────────────────────
        if st.button(
            "📂  Chat History",
            key                 = "sb_history",
            use_container_width = True
        ):
            st.session_state.page          = "history"
            st.session_state.view_hist_data = None
            st.rerun()

        if st.session_state.page != "chat":
            if st.button(
                "💬  Back to Chat",
                key                 = "sb_back",
                use_container_width = True
            ):
                st.session_state.page          = "chat"
                st.session_state.view_hist_data = None
                st.rerun()

        st.divider()

        # ── PATIENT NAME ──────────────────────────────
        st.markdown("**👤 Patient Name**")
        name_input = st.text_input(
            "name_input",
            value            = st.session_state.patient_name,
            label_visibility = "collapsed",
            placeholder      = "Your name..."
        )
        if name_input and name_input != st.session_state.patient_name:
            st.session_state.patient_name = name_input

        st.divider()

        # ── CONTEXTUAL MEMORY ─────────────────────────
        st.markdown("**🧠 Memory**")
        if st.session_state.last_cancer:
            st.markdown(
                f"<div class='mem-badge'>🎯 {st.session_state.last_cancer.upper()}</div>",
                unsafe_allow_html=True
            )
            st.caption("Ask follow-up questions like:\n• *'what is the treatment for this?'*\n• *'what is the survival rate?'*")
        else:
            st.caption("No context yet. Memory activates when you mention a cancer.")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Turns",     st.session_state.turn_count)
        with c2:
            st.metric("Questions", st.session_state.total_queries)

        st.divider()

        # ── QUICK QUESTIONS ───────────────────────────
        if st.session_state.page == "chat":
            st.markdown("**💡 Quick Questions**")
            quick_qs = [
                "What is lung cancer?",
                "What is the treatment for this?",
                "What are symptoms of breast cancer?",
                "What is colon cancer survival rate?",
                "What are chemotherapy side effects?",
                "How to prevent cervical cancer?",
                "How is leukemia diagnosed?",
                "Is cancer hereditary?",
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

        st.caption("⚠️ Educational use only.")


# ==================================================
# RENDER MESSAGES
# ==================================================

def render_messages(agent):

    for idx, msg in enumerate(st.session_state.messages):

        role    = msg.get("role", "")
        content = msg.get("content", "")
        action  = msg.get("action", "answer")
        meta    = msg.get("metadata", {})
        is_edit = msg.get("is_edit", False)

        # ── PATIENT USER MESSAGE ───────────────────────
        if role == "patient":
            is_editing = st.session_state.editing_idx == idx
            if is_editing:
                st.markdown("<div class='edit-box'>", unsafe_allow_html=True)
                edited = st.text_area("Edit question:", value=content, key=f"ea_{idx}", height=75)
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
                        f"<div class='patient-label'>👤 {st.session_state.patient_name}</div>"
                        f"<div class='patient-text'>{content}</div></div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.write("")
                    if st.button("✏️", key=f"eb_{idx}", help="Edit"):
                        st.session_state.editing_idx = idx
                        st.rerun()

        # ── DOCTOR AI RESPONSE ────────────────────────
        elif role == "doctor":
            qtype  = meta.get("question_type", "")
            is_fup = meta.get("is_followup", False) or (action == "question")

            if qtype == "non_medical":
                css  = "reject-msg"
                icon = "⚠️ MedBot"
            elif action == "question":
                css  = "doctor-q-msg"
                icon = "❓ Doctor AI (Question to Patient)"
            else:
                css  = "doctor-msg"
                icon = "🩺 Doctor AI"

            st.markdown(
                f"<div class='{css}'>"
                f"<b>{icon}:</b><br>{content}</div>",
                unsafe_allow_html=True
            )
            if meta and qtype != "non_medical":
                _render_meta(meta)


# ==================================================
# RENDER METADATA EXPANDER
# ==================================================

def _render_meta(meta: dict):

    with st.expander("📋 Response Details & RAG Verification", expanded=False):

        score        = meta.get("hallucination", {}).get("score", 0.0)
        verdict      = meta.get("hallucination", {}).get("verdict", "")
        cancer       = meta.get("cancer_type", "")
        resolved     = meta.get("resolved_question", "")
        original     = meta.get("question", "")
        was_resolved = meta.get("was_resolved", False)
        action       = meta.get("action", "answer")
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
            action_label = "1-LINE ANSWER" if action == "answer" else "QUESTION"
            st.metric("Response Format", action_label)
        with c3:
            st.metric("Confidence", meta.get("confidence", 0.0))

        if was_resolved and resolved != original:
            st.markdown(
                f"<div class='mem-resolution-badge'>"
                f"🧠 <b>Memory Resolved:</b> <i>'{original}'</i> → <b>'{resolved}'</b>"
                f"</div>",
                unsafe_allow_html=True
            )

        if cancer and cancer not in ["cancer", ""]:
            st.markdown(f"<div class='mem-badge'>🎯 CANCER: {cancer.upper()}</div>", unsafe_allow_html=True)

        if "PASS" in str(verdict):
            st.success(f"✅ {verdict}")
        elif "FAIL" in str(verdict):
            st.error(f"❌ {verdict}")

        if mem_ctx:
            with st.expander("🧠 Conversation Memory History"):
                st.text(mem_ctx)

        sources = meta.get("sources", [])
        if sources:
            st.markdown("**📚 Oncology Textbook References:**")
            for i, s in enumerate(sources):
                st.markdown(
                    f"<div class='src-card'>{i+1}. {s.get('source','?')} — Score: {s.get('score',0)}</div>",
                    unsafe_allow_html=True
                )


# ==================================================
# CHAT PAGE
# ==================================================

def page_chat(agent):

    greeting = get_greeting()
    name     = st.session_state.patient_name

    st.markdown("### 💬 Consultation Room")

    if not st.session_state.messages:
        st.markdown(
            f"""<div class='doctor-msg'>
            <b>🩺 Doctor AI:</b><br><br>
            <b>{greeting}!</b> {"Welcome, " + name + "!" if name != "Patient" else "Welcome!"}<br><br>
            I know that cancer concerns can feel overwhelming — <b>you are not alone</b>.
            I am your dedicated oncology AI assistant powered by Agentic RAG and 25 oncology textbooks.<br><br>
            I can help with:<br>
            • Cancer <b>symptoms</b> and warning signs<br>
            • <b>Diagnosis</b> methods and tests<br>
            • <b>Treatment</b> options and side effects<br>
            • <b>Prognosis</b> and survival information<br>
            • Cancer <b>staging</b> and classification<br>
            • <b>Prevention</b> and risk factors<br><br>
            <i>Please share what's on your mind. I am here for you.</i>
            </div>""",
            unsafe_allow_html=True
        )

    render_messages(agent)

    st.divider()

    user_input = st.chat_input("Ask about cancer... (Enter to send)")

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

    st.markdown("## 📂 Saved Chat Histories")
    chats = load_all_chats()

    if not chats:
        st.info("No saved chats yet.\n\nClick **💾 Save Chat** or **➕ New Chat** to save a conversation.")
        if st.button("💬 Go to Chat", type="primary"):
            st.session_state.page = "chat"
            st.rerun()
        return

    search = st.text_input(
        "hist_search",
        placeholder      = "🔍 Search by name or date...",
        label_visibility = "collapsed"
    )

    filtered = [
        c for c in chats
        if not search or search.lower() in (c.get("patient_name", "") + c.get("saved_at", "")).lower()
    ]

    st.markdown(f"**{len(filtered)} conversation(s)**" + (f" (filtered from {len(chats)})" if search else ""))
    st.divider()

    for chat in filtered:
        filename = chat.get("filename", "")
        name     = chat.get("patient_name", "Patient")
        saved_at = chat.get("saved_at", "")[:19].replace("T", " ")
        total    = chat.get("total_msgs", 0)
        msgs     = chat.get("messages", [])

        first_q = "—"
        for m in msgs:
            if m.get("role") == "patient":
                first_q = m.get("content", "")[:90]
                break

        st.markdown(
            f"""<div class='chat-card'>
            <b style='color:#4a9eff;'>👤 {name}</b>
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
                st.session_state.session_id    = chat.get("session_id", _new_sid())
                st.session_state.total_queries = len([m for m in msgs if m.get("role") == "patient"])
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
    saved_at = chat.get("saved_at", "")[:19].replace("T", " ")
    total    = chat.get("total_msgs", 0)
    msgs     = chat.get("messages", [])

    cb, cc = st.columns([1, 1])
    with cb:
        if st.button("← Back", key="vcb_back", type="secondary"):
            st.session_state.page           = "history"
            st.session_state.view_hist_data = None
            st.rerun()

    with cc:
        if st.button("▶️ Continue This Chat", key="vcb_cont", type="primary"):
            st.session_state.messages      = msgs
            st.session_state.patient_name  = name
            st.session_state.session_id    = chat.get("session_id", _new_sid())
            st.session_state.total_queries = len([m for m in msgs if m.get("role") == "patient"])
            st.session_state.turn_count    = len(msgs)
            st.session_state.page          = "chat"
            st.session_state.view_hist_data= None
            st.rerun()

    st.markdown(f"## 📄 {name}'s Consultation")
    st.caption(f"🕐 Saved: {saved_at}  |  💬 {total} messages")
    st.divider()

    for msg in msgs:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        meta    = msg.get("metadata", {})

        if role == "patient":
            st.markdown(f"<div class='patient-msg'><div class='patient-label'>👤 {name}</div><div class='patient-text'>{content}</div></div>", unsafe_allow_html=True)
        elif role == "doctor":
            qtype  = meta.get("question_type", "")
            is_fup = meta.get("is_followup", False)

            if qtype == "non_medical":
                css  = "reject-msg"
                icon = "⚠️ MedBot"
            elif is_fup:
                css  = "followup-msg"
                icon = "🔄 Doctor AI"
            else:
                css  = "doctor-msg"
                icon = "🩺 Doctor AI"

            st.markdown(f"<div class='{css}'><b>{icon}:</b><br>{content}</div>", unsafe_allow_html=True)


# ==================================================
# MAIN
# ==================================================

def main():

    init_session()
    render_sidebar()

    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 Doctor-Patient Simulation</h1>
        <p>Powered by Agentic RAG + Memory Layer + LLaMA3 + 25 Oncology Textbooks</p>
        <p style='color:#888899;font-size:12px;'>
        ⚠️ For educational purposes only. Always consult a real oncologist.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load agent
    if st.session_state.agent is None:
        with st.spinner("🔄 Loading Medical AI..."):
            try:
                st.session_state.agent = load_agent()
                st.success("✅ Medical AI loaded!")
            except Exception as e:
                st.error(f"❌ Failed: {e}")
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

        st.markdown("### 📊 Live Stats")
        total  = st.session_state.total_queries
        passed = st.session_state.total_passed
        scores = st.session_state.all_scores

        st.metric("Questions", total)
        st.metric("Passed", f"{passed}/{total}" if total > 0 else "0/0")

        if scores:
            avg = round(sum(scores)/len(scores), 2)
            st.metric("Avg Score", f"{avg}/5.0")
            st.progress(min(avg/5.0, 1.0))
            if avg >= 4.4:
                st.success("🎯 Target!")
            elif avg >= 4.0:
                st.info("📈 Almost!")
            else:
                st.warning("⚠️ Below target")

        st.divider()

        # Memory
        st.markdown("### 🧠 Memory")
        if st.session_state.last_cancer:
            st.markdown(
                f"<div class='mem-badge'>🎯 {st.session_state.last_cancer.upper()}</div>",
                unsafe_allow_html=True
            )
            st.caption(
                "Ask follow-ups like:\n"
                "• What is the treatment for this?\n"
                "• What is the survival rate?"
            )
        else:
            st.caption("Ask about a cancer to activate memory.")

        st.divider()

        # Saved chats
        st.markdown("### 💾 Saved Chats")
        chats = load_all_chats()
        st.metric("Total", len(chats))
        if chats:
            if st.button("📂 View History", key="info_hist", use_container_width=True):
                st.session_state.page = "history"
                st.rerun()

        st.divider()

        # About
        st.markdown("### 🔬 About")
        st.markdown("""
        **Sources:**
        - 25 Oncology Textbooks
        - 45,384 Knowledge Chunks

        **Features:**
        - Contextual Memory Layer
        - 1-Line Answers & Clinical Questions
        - LLaMA3 via Groq API
        - Agentic RAG
        - ✏️ Edit Questions
        - 💾 Save Chat
        - 📂 View History
        - ▶️ Continue Old Chats
        """)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    main()
