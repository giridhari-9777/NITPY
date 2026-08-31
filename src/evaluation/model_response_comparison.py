# src/evaluation/model_response_comparison.py
# Ask same complex questions to all models
# Save responses as screenshot images

import os
import sys
import json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

os.makedirs("results/response_screenshots", exist_ok=True)


# ==================================================
# COLORS
# ==================================================

COLORS = {
    "bg"      : "#0d0d14",
    "card"    : "#1a1a2e",
    "border"  : "#2a2a4f",
    "llama3"  : "#4a9eff",
    "mistral" : "#4caf50",
    "gemma3"  : "#ff9800",
    "qwen"    : "#e91e63",
    "deepseek": "#9c27b0",
    "phi4"    : "#00bcd4",
    "header"  : "#1a2a4f",
    "text"    : "#ffffff",
    "subtext" : "#aaaacc",
    "green"   : "#4caf50",
    "red"     : "#ff5252",
    "gold"    : "#ffd700",
}

# Model display info
MODELS = {
    "llama3"     : {"label": "LLaMA3",     "color": COLORS["llama3"]},
    "mistral"    : {"label": "Mistral 7B", "color": COLORS["mistral"]},
    "gemma3"     : {"label": "Gemma3",     "color": COLORS["gemma3"]},
    "qwen2.5"    : {"label": "Qwen2.5",    "color": COLORS["qwen"]},
    "deepseek-r1": {"label": "DeepSeek-R1","color": COLORS["deepseek"]},
    "phi4"       : {"label": "Phi-4",      "color": COLORS["phi4"]},
}


# ==================================================
# COMPLEX MEDICAL QUESTIONS
# ==================================================

QUESTIONS = [
    {
        "id"      : "Q1",
        "category": "Treatment",
        "question": (
            "What is the recommended treatment protocol "
            "for stage III non-small cell lung cancer "
            "in a patient with EGFR mutation, and how "
            "does targeted therapy differ from "
            "traditional chemotherapy in this case?"
        ),
        "why_complex": (
            "Requires knowledge of EGFR mutations, "
            "targeted therapy, staging, and treatment comparison"
        )
    },
    {
        "id"      : "Q2",
        "category": "Diagnosis",
        "question": (
            "How do you differentiate between Hodgkin "
            "lymphoma and Non-Hodgkin lymphoma using "
            "histopathological findings, and what are "
            "the key Reed-Sternberg cell characteristics "
            "used in diagnosis?"
        ),
        "why_complex": (
            "Requires knowledge of histopathology, "
            "Reed-Sternberg cells, and differential diagnosis"
        )
    },
    {
        "id"      : "Q3",
        "category": "Prognosis",
        "question": (
            "What factors determine the 5-year survival "
            "rate in triple-negative breast cancer, and "
            "how does the presence of BRCA1 mutation "
            "affect treatment options and prognosis?"
        ),
        "why_complex": (
            "Requires understanding of BRCA mutations, "
            "triple-negative subtype, survival statistics"
        )
    },
    {
        "id"      : "Q4",
        "category": "Mechanism",
        "question": (
            "Explain the mechanism of action of CAR-T "
            "cell therapy in treating B-cell acute "
            "lymphoblastic leukemia and what are the "
            "major side effects including cytokine "
            "release syndrome?"
        ),
        "why_complex": (
            "Advanced immunotherapy mechanism, "
            "CAR-T cell engineering, toxicity management"
        )
    },
    {
        "id"      : "Q5",
        "category": "Staging",
        "question": (
            "Describe the TNM staging system for "
            "colorectal cancer and explain how the "
            "stage determines whether adjuvant "
            "chemotherapy with FOLFOX regimen is "
            "indicated after surgical resection?"
        ),
        "why_complex": (
            "Requires TNM staging knowledge, "
            "surgical oncology, and chemotherapy protocols"
        )
    },
]


# ==================================================
# RETRIEVE CONTEXT FROM CHROMADB
# ==================================================

def get_context(question: str, collection, model) -> tuple:

    q_emb = model.encode(
        question,
        normalize_embeddings = True,
        convert_to_numpy     = True
    )

    result = collection.query(
        query_embeddings = [q_emb.tolist()],
        n_results        = 5,
        include          = ["documents","distances"]
    )

    chunks = []
    for i in range(len(result["ids"][0])):
        score = 1 - result["distances"][0][i]
        chunks.append({
            "text"  : result["documents"][0][i],
            "score" : round(score, 4)
        })

    context = "\n\n".join([c["text"] for c in chunks])
    return context, chunks


# ==================================================
# GENERATE FROM EACH MODEL
# ==================================================

def get_model_prompt(
    model_name : str,
    question   : str,
    context    : str
) -> str:

    base_context = f"""You are an expert oncologist with 20 years
of experience. Answer the medical question based
ONLY on the provided context. Be thorough,
accurate and empathetic.

CONTEXT:
{context[:1500]}

QUESTION: {question}

Provide a comprehensive answer in 2-4 sentences:"""

    if model_name == "mistral":
        return f"<s>[INST] {base_context} [/INST]"

    elif model_name in ["qwen2.5"]:
        return f"""<|im_start|>system
You are an expert oncologist.
<|im_end|>
<|im_start|>user
{base_context}
<|im_end|>
<|im_start|>assistant
"""

    elif model_name == "phi4":
        return f"""<|system|>
You are an expert oncologist.
<|end|>
<|user|>
{base_context}
<|end|>
<|assistant|>"""

    else:
        # llama3, gemma3, deepseek-r1
        return base_context


def clean_response(model_name: str, raw: str) -> str:

    import re

    # Strip DeepSeek thinking tags
    if model_name == "deepseek-r1":
        raw = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.DOTALL
        ).strip()

    # Strip special tokens
    for token in [
        "<|im_end|>","<|im_start|>","<|end|>",
        "<|assistant|>","<|user|>","<|system|>",
        "[/INST]","[INST]","</s>"
    ]:
        raw = raw.replace(token, "")

    # Strip HTML tags
    raw = re.sub(r"<[^>]+>", "", raw)

    return raw.strip()


def generate_response(
    model_name : str,
    question   : str,
    context    : str,
    timeout    : int = 180
) -> dict:

    prompt = get_model_prompt(model_name, question, context)

    start_time = datetime.now()

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json    = {
                "model"  : model_name,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature" : 0.1,
                    "num_predict" : 300
                }
            },
            timeout = timeout
        )

        elapsed = (
            datetime.now() - start_time
        ).total_seconds()

        if resp.status_code == 200:
            raw    = resp.json().get("response","").strip()
            answer = clean_response(model_name, raw)

            return {
                "success"  : True,
                "answer"   : answer,
                "elapsed"  : round(elapsed, 2),
                "error"    : ""
            }
        else:
            return {
                "success"  : False,
                "answer"   : "",
                "elapsed"  : round(elapsed, 2),
                "error"    : f"HTTP {resp.status_code}"
            }

    except requests.exceptions.Timeout:
        return {
            "success" : False,
            "answer"  : "",
            "elapsed" : timeout,
            "error"   : "Timeout"
        }
    except Exception as e:
        return {
            "success" : False,
            "answer"  : "",
            "elapsed" : 0.0,
            "error"   : str(e)
        }


# ==================================================
# CHECK WHICH MODELS ARE AVAILABLE
# ==================================================

def get_available_models() -> list:

    try:
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        if resp.status_code == 200:
            data   = resp.json()
            models = [
                m["name"].split(":")[0]
                for m in data.get("models", [])
            ]
            # Match to our model keys
            available = []
            for key in MODELS.keys():
                for m in models:
                    if key in m or m in key:
                        available.append(key)
                        break
            print(f"  Available: {available}")
            return available
    except Exception as e:
        print(f"  Error checking models: {e}")

    # Default — try all
    return list(MODELS.keys())


# ==================================================
# SCREENSHOT 1 — Single Question, All Models
# ==================================================

def save_question_screenshot(
    question_data : dict,
    responses     : dict,
    available_models: list
):

    q_id      = question_data["id"]
    question  = question_data["question"]
    category  = question_data["category"]

    n_models  = len(available_models)
    fig_height= max(14, n_models * 4.5 + 4)

    fig = plt.figure(
        figsize       = (18, fig_height),
        facecolor     = COLORS["bg"]
    )

    # ── Title Block ───────────────────────────────
    title_ax = fig.add_axes([0.02, 0.94, 0.96, 0.05])
    title_ax.set_facecolor(COLORS["header"])
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")

    title_ax.text(
        0.5, 0.7,
        f"NITPY — Doctor-Patient Cancer QA  |  "
        f"After RL Optimization  |  {q_id}: {category}",
        ha         = "center",
        va         = "center",
        fontsize   = 13,
        fontweight = "bold",
        color      = "white",
        transform  = title_ax.transAxes
    )
    title_ax.text(
        0.5, 0.2,
        f"Question: {question[:120]}...",
        ha         = "center",
        va         = "center",
        fontsize   = 10,
        color      = COLORS["subtext"],
        transform  = title_ax.transAxes,
        style      = "italic"
    )

    # ── Question Box ─────────────────────────────
    q_ax = fig.add_axes([0.02, 0.87, 0.96, 0.06])
    q_ax.set_facecolor("#1a2a4f")
    q_ax.set_xlim(0, 1)
    q_ax.set_ylim(0, 1)
    q_ax.axis("off")

    for spine in ["top","bottom","left","right"]:
        q_ax.spines[spine].set_visible(False)

    q_ax.text(
        0.01, 0.75,
        "👤 Patient Question:",
        ha         = "left",
        va         = "center",
        fontsize   = 10,
        fontweight = "bold",
        color      = "#4a9eff",
        transform  = q_ax.transAxes
    )
    q_ax.text(
        0.01, 0.25,
        question,
        ha         = "left",
        va         = "center",
        fontsize   = 10,
        color      = "white",
        transform  = q_ax.transAxes,
        wrap       = True
    )

    # ── Model Response Cards ──────────────────────
    card_height = 0.82 / n_models
    card_gap    = 0.005

    for idx, model_key in enumerate(available_models):

        info = MODELS.get(model_key, {
            "label": model_key.upper(),
            "color": "#ffffff"
        })
        resp = responses.get(model_key, {})

        y_pos = 0.86 - (idx + 1) * (card_height + card_gap)

        card_ax = fig.add_axes([
            0.02,
            y_pos,
            0.96,
            card_height
        ])
        card_ax.set_facecolor(COLORS["card"])
        card_ax.set_xlim(0, 1)
        card_ax.set_ylim(0, 1)
        card_ax.axis("off")

        # Colored left border
        border_ax = fig.add_axes([
            0.02,
            y_pos,
            0.005,
            card_height
        ])
        border_ax.set_facecolor(info["color"])
        border_ax.axis("off")

        # Model name header
        card_ax.text(
            0.01, 0.88,
            f"🩺 {info['label']}",
            ha         = "left",
            va         = "center",
            fontsize   = 11,
            fontweight = "bold",
            color      = info["color"],
            transform  = card_ax.transAxes
        )

        # Response time
        elapsed = resp.get("elapsed", 0)
        success = resp.get("success", False)

        card_ax.text(
            0.85, 0.88,
            f"⏱ {elapsed}s",
            ha         = "left",
            va         = "center",
            fontsize   = 9,
            color      = COLORS["subtext"],
            transform  = card_ax.transAxes
        )

        # Status badge
        status_color = COLORS["green"] if success else COLORS["red"]
        status_text  = "✅ Success" if success else "❌ Failed"
        card_ax.text(
            0.92, 0.88,
            status_text,
            ha         = "left",
            va         = "center",
            fontsize   = 9,
            color      = status_color,
            fontweight = "bold",
            transform  = card_ax.transAxes
        )

        # Answer text
        answer = resp.get("answer","")
        error  = resp.get("error", "")

        if success and answer:
            # Wrap answer text
            display_text = answer
            if len(display_text) > 600:
                display_text = display_text[:600] + "..."

            card_ax.text(
                0.01, 0.55,
                display_text,
                ha          = "left",
                va          = "center",
                fontsize    = 9.5,
                color       = "white",
                transform   = card_ax.transAxes,
                wrap        = True,
                linespacing = 1.5
            )
        else:
            card_ax.text(
                0.01, 0.55,
                f"❌ Error: {error}",
                ha        = "left",
                va        = "center",
                fontsize  = 10,
                color     = COLORS["red"],
                transform = card_ax.transAxes
            )

        # Word count
        wc = len(answer.split()) if answer else 0
        card_ax.text(
            0.01, 0.08,
            f"Words: {wc}",
            ha        = "left",
            va        = "center",
            fontsize  = 8,
            color     = COLORS["subtext"],
            transform = card_ax.transAxes
        )

    # ── Footer ────────────────────────────────────
    footer_ax = fig.add_axes([0.02, 0.00, 0.96, 0.02])
    footer_ax.set_facecolor(COLORS["header"])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"NITPY | Oncology QA | Agentic RAG + RL | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha        = "center",
        va        = "center",
        fontsize  = 8,
        color     = COLORS["subtext"],
        transform = footer_ax.transAxes
    )

    path = (
        f"results/response_screenshots/"
        f"{q_id}_all_models.png"
    )
    plt.savefig(
        path, dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ Saved: {path}")


# ==================================================
# SCREENSHOT 2 — All Questions Summary per Model
# ==================================================

def save_model_summary_screenshot(
    all_responses    : dict,
    available_models : list
):

    for model_key in available_models:

        info = MODELS.get(model_key, {
            "label": model_key.upper(),
            "color": "#ffffff"
        })

        n_q       = len(QUESTIONS)
        fig_height= max(16, n_q * 5 + 4)

        fig = plt.figure(
            figsize   = (16, fig_height),
            facecolor = COLORS["bg"]
        )

        # Title
        title_ax = fig.add_axes([0.02, 0.96, 0.96, 0.04])
        title_ax.set_facecolor(info["color"])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.5,
            f"🩺 {info['label']} — All Responses "
            f"| NITPY Oncology QA | After RL",
            ha         = "center",
            va         = "center",
            fontsize   = 14,
            fontweight = "bold",
            color      = "white",
            transform  = title_ax.transAxes
        )

        card_height = 0.88 / n_q
        card_gap    = 0.005

        for idx, qdata in enumerate(QUESTIONS):

            q_id     = qdata["id"]
            question = qdata["question"]
            category = qdata["category"]
            resp     = all_responses.get(q_id, {}).get(
                model_key, {}
            )
            answer   = resp.get("answer",  "")
            elapsed  = resp.get("elapsed", 0)
            success  = resp.get("success", False)

            y_pos = 0.95 - (idx + 1) * (card_height + card_gap)

            # Question header
            qh_ax = fig.add_axes([
                0.02, y_pos + card_height * 0.72,
                0.96, card_height * 0.28
            ])
            qh_ax.set_facecolor("#1a2a4f")
            qh_ax.axis("off")

            qh_ax.text(
                0.01, 0.70,
                f"{q_id} [{category}]  ⏱ {elapsed}s",
                ha         = "left",
                va         = "center",
                fontsize   = 10,
                fontweight = "bold",
                color      = info["color"],
                transform  = qh_ax.transAxes
            )
            qh_ax.text(
                0.01, 0.20,
                f"Q: {question[:130]}...",
                ha        = "left",
                va        = "center",
                fontsize  = 9,
                color     = COLORS["subtext"],
                transform = qh_ax.transAxes,
                style     = "italic"
            )

            # Answer body
            ans_ax = fig.add_axes([
                0.02, y_pos,
                0.96, card_height * 0.70
            ])
            ans_ax.set_facecolor(COLORS["card"])
            ans_ax.axis("off")

            # Left border
            border_ax = fig.add_axes([
                0.02, y_pos,
                0.005, card_height * 0.70
            ])
            border_ax.set_facecolor(info["color"])
            border_ax.axis("off")

            if success and answer:
                display = (
                    answer[:500] + "..."
                    if len(answer) > 500
                    else answer
                )
                ans_ax.text(
                    0.02, 0.50,
                    display,
                    ha          = "left",
                    va          = "center",
                    fontsize    = 9.5,
                    color       = "white",
                    transform   = ans_ax.transAxes,
                    linespacing = 1.5,
                    wrap        = True
                )
                wc = len(answer.split())
                ans_ax.text(
                    0.85, 0.08,
                    f"Words: {wc}  ✅",
                    ha        = "left",
                    va        = "center",
                    fontsize  = 8,
                    color     = COLORS["green"],
                    transform = ans_ax.transAxes
                )
            else:
                ans_ax.text(
                    0.02, 0.50,
                    f"❌ {resp.get('error', 'No response')}",
                    ha        = "left",
                    va        = "center",
                    fontsize  = 10,
                    color     = COLORS["red"],
                    transform = ans_ax.transAxes
                )

        # Footer
        footer_ax = fig.add_axes([0.02, 0.00, 0.96, 0.02])
        footer_ax.set_facecolor(COLORS["header"])
        footer_ax.axis("off")
        footer_ax.text(
            0.5, 0.5,
            f"NITPY | {info['label']} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha        = "center",
            va        = "center",
            fontsize  = 8,
            color     = COLORS["subtext"],
            transform = footer_ax.transAxes
        )

        path = (
            f"results/response_screenshots/"
            f"model_{model_key.replace('-','_')}_all_questions.png"
        )
        plt.savefig(
            path, dpi=150,
            bbox_inches="tight",
            facecolor=COLORS["bg"]
        )
        plt.close()
        print(f"  ✅ Saved: {path}")


# ==================================================
# SCREENSHOT 3 — Side by Side Comparison Table
# ==================================================

def save_comparison_table(
    all_responses    : dict,
    available_models : list
):

    fig, ax = plt.subplots(
        figsize   = (22, len(QUESTIONS) * 2.5 + 4),
        facecolor = COLORS["bg"]
    )
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.98,
        "NITPY — Model Response Comparison | "
        "Complex Medical Questions | After RL",
        ha         = "center",
        va         = "top",
        fontsize   = 16,
        fontweight = "bold",
        color      = "white",
        transform  = ax.transAxes
    )
    ax.text(
        0.5, 0.95,
        "Same Questions Asked to All Models — "
        "Agentic RAG + RL Optimization",
        ha        = "center",
        va        = "top",
        fontsize  = 11,
        color     = COLORS["subtext"],
        transform = ax.transAxes
    )

    # Build table data
    col_labels = ["Question\n& Category"] + [
        MODELS.get(m, {"label": m})["label"]
        for m in available_models
    ]

    table_data = []
    for qdata in QUESTIONS:
        q_id     = qdata["id"]
        category = qdata["category"]
        question = qdata["question"][:60] + "..."
        row      = [f"{q_id}\n[{category}]\n{question}"]

        for model_key in available_models:
            resp   = all_responses.get(q_id, {}).get(
                model_key, {}
            )
            answer = resp.get("answer", "")
            elapsed= resp.get("elapsed", 0)

            if answer:
                preview = answer[:80] + "..."
                row.append(f"{preview}\n⏱{elapsed}s")
            else:
                row.append(
                    f"❌ {resp.get('error','Failed')}"
                )

        table_data.append(row)

    table = ax.table(
        cellText  = table_data,
        colLabels = col_labels,
        cellLoc   = "left",
        loc       = "center",
        bbox      = [0, 0.02, 1, 0.90]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 4.5)

    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1a2a4f")
        cell.set_text_props(
            color="white", fontweight="bold"
        )
        if j > 0:
            model_key = available_models[j-1]
            color     = MODELS.get(
                model_key, {"color": "#ffffff"}
            )["color"]
            cell.set_facecolor(color)
            cell.set_text_props(
                color="white", fontweight="bold"
            )

    # Row styling
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = table[i+1, j]
            if j == 0:
                cell.set_facecolor("#1a1a3a")
                cell.set_text_props(
                    color=COLORS["subtext"]
                )
            else:
                bg = "#1a2e1a" if "❌" not in str(
                    table_data[i][j]
                ) else "#2e1a1a"
                cell.set_facecolor(bg)
                cell.set_text_props(color="white")

    path = (
        "results/response_screenshots/"
        "00_comparison_table.png"
    )
    plt.savefig(
        path, dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ Saved: {path}")


# ==================================================
# SCREENSHOT 4 — Response Time Comparison
# ==================================================

def save_response_time_chart(
    all_responses    : dict,
    available_models : list
):

    fig, ax = plt.subplots(
        figsize   = (14, 7),
        facecolor = COLORS["bg"]
    )
    ax.set_facecolor("#1a1a2e")

    x     = np.arange(len(QUESTIONS))
    width = 0.12
    n     = len(available_models)
    start = -(n-1)/2 * width

    for i, model_key in enumerate(available_models):

        info   = MODELS.get(model_key, {
            "label": model_key,
            "color": "#ffffff"
        })
        times  = []
        for qdata in QUESTIONS:
            resp = all_responses.get(
                qdata["id"], {}
            ).get(model_key, {})
            times.append(resp.get("elapsed", 0))

        offset = start + i * width
        bars   = ax.bar(
            x + offset, times,
            width     = width * 0.9,
            label     = info["label"],
            color     = info["color"],
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.4
        )

        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{t:.1f}s",
                    ha       = "center",
                    va       = "bottom",
                    fontsize = 7,
                    color    = info["color"]
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Q{i+1}\n{q['category']}"
         for i, q in enumerate(QUESTIONS)],
        fontsize=10, color="white"
    )
    ax.set_ylabel("Response Time (seconds)", fontsize=12,
                  color="white")
    ax.set_title(
        "Response Time Comparison — All Models",
        fontsize=14, fontweight="bold",
        color="white", pad=15
    )
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.3, color="#2a2a3a")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2a2a3a")
    ax.spines["left"].set_color("#2a2a3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = (
        "results/response_screenshots/"
        "00_response_time.png"
    )
    plt.savefig(
        path, dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ Saved: {path}")


# ==================================================
# SCREENSHOT 5 — Word Count Comparison
# ==================================================

def save_word_count_chart(
    all_responses    : dict,
    available_models : list
):

    fig, ax = plt.subplots(
        figsize   = (14, 7),
        facecolor = COLORS["bg"]
    )
    ax.set_facecolor("#1a1a2e")

    x     = np.arange(len(QUESTIONS))
    width = 0.12
    n     = len(available_models)
    start = -(n-1)/2 * width

    for i, model_key in enumerate(available_models):

        info  = MODELS.get(model_key, {
            "label": model_key,
            "color": "#ffffff"
        })
        wcs   = []
        for qdata in QUESTIONS:
            resp   = all_responses.get(
                qdata["id"], {}
            ).get(model_key, {})
            answer = resp.get("answer", "")
            wcs.append(len(answer.split()) if answer else 0)

        offset = start + i * width
        bars   = ax.bar(
            x + offset, wcs,
            width     = width * 0.9,
            label     = info["label"],
            color     = info["color"],
            alpha     = 0.85,
            edgecolor = "white",
            linewidth = 0.4
        )

        for bar, wc in zip(bars, wcs):
            if wc > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    str(wc),
                    ha       = "center",
                    va       = "bottom",
                    fontsize = 7,
                    color    = info["color"]
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Q{i+1}\n{q['category']}"
         for i, q in enumerate(QUESTIONS)],
        fontsize=10, color="white"
    )
    ax.set_ylabel("Word Count", fontsize=12, color="white")
    ax.set_title(
        "Response Length (Word Count) — All Models",
        fontsize=14, fontweight="bold",
        color="white", pad=15
    )
    ax.axhline(
        y=50,  color="#ff9800", linestyle="--",
        alpha=0.6, linewidth=1, label="Min target (50)"
    )
    ax.axhline(
        y=200, color="#4caf50", linestyle="--",
        alpha=0.6, linewidth=1, label="Max target (200)"
    )
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3, color="#2a2a3a")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2a2a3a")
    ax.spines["left"].set_color("#2a2a3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = (
        "results/response_screenshots/"
        "00_word_count.png"
    )
    plt.savefig(
        path, dpi=150,
        bbox_inches="tight",
        facecolor=COLORS["bg"]
    )
    plt.close()
    print(f"  ✅ Saved: {path}")


# ==================================================
# SAVE JSON REPORT
# ==================================================

def save_json_report(
    all_responses    : dict,
    available_models : list
):

    report = {
        "generated_at"     : datetime.now().isoformat(),
        "total_questions"  : len(QUESTIONS),
        "total_models"     : len(available_models),
        "models_tested"    : available_models,
        "questions"        : [],
    }

    for qdata in QUESTIONS:
        q_id = qdata["id"]
        q_entry = {
            "id"          : q_id,
            "category"    : qdata["category"],
            "question"    : qdata["question"],
            "why_complex" : qdata["why_complex"],
            "responses"   : {}
        }

        for model_key in available_models:
            resp = all_responses.get(q_id, {}).get(
                model_key, {}
            )
            q_entry["responses"][model_key] = {
                "model"   : MODELS.get(
                    model_key,{"label":model_key}
                )["label"],
                "answer"  : resp.get("answer",  ""),
                "elapsed" : resp.get("elapsed", 0),
                "success" : resp.get("success", False),
                "words"   : len(
                    resp.get("answer","").split()
                ),
                "error"   : resp.get("error", "")
            }

        report["questions"].append(q_entry)

    path = (
        "results/response_screenshots/"
        "response_report.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  ✅ JSON saved: {path}")
    return path


# ==================================================
# PRINT CONSOLE SUMMARY
# ==================================================

def print_console_summary(
    all_responses    : dict,
    available_models : list
):

    print("\n" + "="*70)
    print("  RESPONSE SUMMARY — ALL MODELS")
    print("="*70)

    for qdata in QUESTIONS:
        q_id     = qdata["id"]
        question = qdata["question"]
        category = qdata["category"]

        print(f"\n{'─'*70}")
        print(f"  {q_id} [{category}]")
        print(f"  Q: {question[:80]}...")
        print(f"{'─'*70}")

        for model_key in available_models:
            info   = MODELS.get(model_key, {
                "label": model_key
            })
            resp   = all_responses.get(q_id, {}).get(
                model_key, {}
            )
            answer = resp.get("answer",  "")
            elapsed= resp.get("elapsed", 0)
            success= resp.get("success", False)

            status = "✅" if success else "❌"
            label  = info["label"]

            print(f"\n  {status} {label} ({elapsed}s):")
            if answer:
                preview = answer[:200]
                if len(answer) > 200:
                    preview += "..."
                print(f"  {preview}")
            else:
                print(f"  ❌ {resp.get('error','No response')}")

    print("\n" + "="*70)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  NITPY — MODEL RESPONSE COMPARISON")
    print("  Complex Medical Questions — After RL")
    print("="*70)

    # Check Ollama
    print("\nChecking Ollama...")
    try:
        requests.get("http://localhost:11434", timeout=3)
        print("  Ollama running ✅")
    except Exception:
        print("  ❌ Ollama not running! Run: ollama serve")
        sys.exit(1)

    # Get available models
    print("\nChecking available models...")
    available_models = get_available_models()

    if not available_models:
        print("  ❌ No models found!")
        print("  Pull models with: ollama pull llama3")
        sys.exit(1)

    print(f"  Testing {len(available_models)} models")

    # Load ChromaDB + embedding model
    print("\nLoading ChromaDB...")
    import chromadb
    from sentence_transformers import SentenceTransformer

    client     = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name     = "medical_rag",
        metadata = {"hnsw:space": "cosine"}
    )
    print(f"  Records: {collection.count()} ✅")

    print("\nLoading embedding model...")
    emb_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("  Ready ✅")

    # ── Ask all questions to all models ──────────
    print("\n" + "="*70)
    print("  ASKING QUESTIONS TO ALL MODELS...")
    print("="*70)

    all_responses = {}

    for qdata in QUESTIONS:

        q_id     = qdata["id"]
        question = qdata["question"]
        category = qdata["category"]

        print(f"\n{'─'*70}")
        print(f"  {q_id} [{category}]")
        print(f"  Q: {question[:70]}...")
        print(f"{'─'*70}")

        # Get context once for this question
        print("  Retrieving context from ChromaDB...")
        context, chunks = get_context(
            question, collection, emb_model
        )
        print(f"  Got {len(chunks)} chunks ✅")

        all_responses[q_id] = {}

        for model_key in available_models:

            info  = MODELS.get(model_key, {
                "label": model_key
            })
            label = info["label"]

            print(f"\n  🔄 {label}...", end=" ", flush=True)

            resp = generate_response(
                model_name = model_key,
                question   = question,
                context    = context,
                timeout    = 180
            )

            all_responses[q_id][model_key] = resp

            if resp["success"]:
                wc = len(resp["answer"].split())
                print(
                    f"✅ {resp['elapsed']}s | "
                    f"{wc} words"
                )
            else:
                print(f"❌ {resp['error']}")

    # ── Generate all screenshots ──────────────────
    print("\n" + "="*70)
    print("  GENERATING SCREENSHOTS...")
    print("="*70)

    # Per question — all models
    print("\n  Per-question screenshots...")
    for qdata in QUESTIONS:
        save_question_screenshot(
            question_data    = qdata,
            responses        = all_responses.get(
                qdata["id"], {}
            ),
            available_models = available_models
        )

    # Per model — all questions
    print("\n  Per-model screenshots...")
    save_model_summary_screenshot(
        all_responses    = all_responses,
        available_models = available_models
    )

    # Comparison table
    print("\n  Comparison table...")
    save_comparison_table(
        all_responses    = all_responses,
        available_models = available_models
    )

    # Response time chart
    print("\n  Response time chart...")
    save_response_time_chart(
        all_responses    = all_responses,
        available_models = available_models
    )

    # Word count chart
    print("\n  Word count chart...")
    save_word_count_chart(
        all_responses    = all_responses,
        available_models = available_models
    )

    # JSON report
    print("\n  JSON report...")
    save_json_report(
        all_responses    = all_responses,
        available_models = available_models
    )

    # Console summary
    print_console_summary(
        all_responses    = all_responses,
        available_models = available_models
    )

    # ── Final Summary ─────────────────────────────
    print("\n" + "="*70)
    print("  ALL SCREENSHOTS SAVED ✅")
    print(f"  Location: results/response_screenshots/")
    print("="*70)

    files = sorted(
        os.listdir("results/response_screenshots/")
    )
    print(f"\n  {len(files)} files generated:")
    for f in files:
        size = os.path.getsize(
            f"results/response_screenshots/{f}"
        )
        print(f"  📸 {f} ({size//1024}KB)")

    print("\n  To open all screenshots:")
    print("  open results/response_screenshots/")