import os
import io
import re
import json
import time
import uuid
import math
import zipfile
import hashlib
import tempfile
import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import yaml
from pypdf import PdfReader

# Optional SDK imports (guarded so UI can still load without all providers installed)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


# ----------------------------
# 0) Constants / Defaults
# ----------------------------

APP_TITLE = "Automated PDF Discovery & Agentic Intelligence System — WOW UI"
DEFAULT_MAX_TOKENS = 12000
DEFAULT_TEMPERATURE = 0.2

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
}

SUPPORTED_MODELS = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    # Anthropic (examples; you can change to match your account)
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

# Grok OpenAI-compatible base URL (xAI)
GROK_BASE_URL = "https://api.x.ai/v1"

# Coral default for keyword highlighting
DEFAULT_KEYWORD_COLOR = "#FF7F50"  # coral


# ----------------------------
# 1) WOW UI: Themes / Styles / i18n
# ----------------------------

PAINTER_STYLES = [
    {"id": "monet", "name": "Claude Monet", "accent": "#2E86AB", "accent2": "#A23B72", "bg": "#F7F7FB", "fg": "#12151A"},
    {"id": "vangogh", "name": "Vincent van Gogh", "accent": "#F2A900", "accent2": "#1B3A57", "bg": "#FFF7E6", "fg": "#12151A"},
    {"id": "picasso", "name": "Pablo Picasso", "accent": "#D7263D", "accent2": "#1B998B", "bg": "#FAFAFA", "fg": "#111827"},
    {"id": "dali", "name": "Salvador Dalí", "accent": "#6D28D9", "accent2": "#F59E0B", "bg": "#F5F3FF", "fg": "#0F172A"},
    {"id": "kahlo", "name": "Frida Kahlo", "accent": "#0EA5A4", "accent2": "#E11D48", "bg": "#ECFEFF", "fg": "#0B1220"},
    {"id": "rembrandt", "name": "Rembrandt", "accent": "#7C2D12", "accent2": "#B45309", "bg": "#FFF7ED", "fg": "#111827"},
    {"id": "vermeer", "name": "Johannes Vermeer", "accent": "#1D4ED8", "accent2": "#F59E0B", "bg": "#EFF6FF", "fg": "#0F172A"},
    {"id": "kandinsky", "name": "Wassily Kandinsky", "accent": "#EF4444", "accent2": "#3B82F6", "bg": "#FFFFFF", "fg": "#0F172A"},
    {"id": "klimt", "name": "Gustav Klimt", "accent": "#D4AF37", "accent2": "#111827", "bg": "#FFFBEB", "fg": "#0F172A"},
    {"id": "pollock", "name": "Jackson Pollock", "accent": "#111827", "accent2": "#F97316", "bg": "#FAFAFA", "fg": "#0B1220"},
    {"id": "hopper", "name": "Edward Hopper", "accent": "#0F766E", "accent2": "#B91C1C", "bg": "#F0FDFA", "fg": "#0F172A"},
    {"id": "okeeffe", "name": "Georgia O’Keeffe", "accent": "#7C3AED", "accent2": "#10B981", "bg": "#F5F3FF", "fg": "#0F172A"},
    {"id": "matisse", "name": "Henri Matisse", "accent": "#F97316", "accent2": "#2563EB", "bg": "#FFF7ED", "fg": "#0F172A"},
    {"id": "cezanne", "name": "Paul Cézanne", "accent": "#166534", "accent2": "#A16207", "bg": "#F7FEE7", "fg": "#0F172A"},
    {"id": "renoir", "name": "Pierre-Auguste Renoir", "accent": "#DB2777", "accent2": "#0284C7", "bg": "#FFF1F2", "fg": "#0F172A"},
    {"id": "caravaggio", "name": "Caravaggio", "accent": "#111827", "accent2": "#EAB308", "bg": "#FFFBEB", "fg": "#0B1220"},
    {"id": "turner", "name": "J. M. W. Turner", "accent": "#F59E0B", "accent2": "#1D4ED8", "bg": "#FFFBEB", "fg": "#0F172A"},
    {"id": "hokusai", "name": "Hokusai", "accent": "#0EA5E9", "accent2": "#111827", "bg": "#F0F9FF", "fg": "#0F172A"},
    {"id": "warhol", "name": "Andy Warhol", "accent": "#EC4899", "accent2": "#22C55E", "bg": "#FFF1F2", "fg": "#0F172A"},
    {"id": "basquiat", "name": "Jean-Michel Basquiat", "accent": "#FACC15", "accent2": "#111827", "bg": "#FFFBEB", "fg": "#0B1220"},
]

I18N = {
    "en": {
        "nav_workspace": "Workspace (PDF ToC)",
        "nav_agent_studio": "Agent Studio",
        "nav_note_keeper": "AI Note Keeper",
        "nav_dashboard": "Dashboard",
        "nav_settings": "Settings",
        "title": APP_TITLE,
        "subtitle": "Scan PDFs → Summarize → Build Master ToC → Run Agents / Keep Notes",
        "api_status": "API Status",
        "connected_env": "Connected (env)",
        "connected_ui": "Connected (session)",
        "missing": "Missing",
        "provider": "Provider",
        "key_source": "Key Source",
        "enter_key": "Enter API key",
        "save_key": "Save key",
        "clear_keys": "Clear session keys",
        "ui_theme": "Theme",
        "ui_lang": "Language",
        "ui_style": "Painter Style",
        "jackpot": "Jackpot",
        "light": "Light",
        "dark": "Dark",
        "english": "English",
        "zh_tw": "Traditional Chinese",
        "pipeline": "PDF Processing Pipeline",
        "upload_pdfs": "Upload PDFs",
        "upload_zip": "Upload ZIP Folder",
        "path_scan": "Scan Path (if accessible)",
        "scan": "Scan",
        "build_toc": "Build / Refresh Master ToC",
        "trim_first_page": "Trim first page (cover/metadata)",
        "summary_prompt": "Summary prompt",
        "model": "Model",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "run": "Run",
        "cancel": "Cancel",
        "toc": "Master Table of Contents (editable)",
        "toc_preview": "ToC Preview",
        "agent_single": "Single Agent Run",
        "agent_pipeline": "Agent Pipeline (Step-by-step)",
        "select_agent": "Select agent",
        "add_step": "Add step",
        "remove_last_step": "Remove last step",
        "run_step": "Run current step",
        "run_all": "Run all steps",
        "reset_pipeline": "Reset pipeline",
        "input_source": "Input source",
        "from_toc": "Master ToC",
        "from_prev": "Previous step output",
        "custom": "Custom",
        "output": "Output",
        "text_view": "Text view",
        "markdown_view": "Markdown view",
        "note_input": "Paste note (text or markdown)",
        "transform_note": "Transform into organized markdown",
        "note_prompt": "Note transform prompt",
        "keywords": "Keywords",
        "keyword_color": "Keyword color",
        "highlight_preview": "Highlighted preview",
        "raw_markdown": "Raw markdown",
        "ai_magics": "AI Magics",
        "apply_magic": "Apply magic",
        "replace_note": "Replace note",
        "append_section": "Append section",
        "download_md": "Download Markdown",
        "load_example": "Load Example (HTNF / K250507)",
        "example_loaded": "Example loaded into Workspace as an uploaded PDF-like text sample.",
        "status": "Status",
        "stage": "Stage",
    },
    "zh-TW": {
        "nav_workspace": "工作區（PDF ToC）",
        "nav_agent_studio": "代理工作室",
        "nav_note_keeper": "AI 筆記管家",
        "nav_dashboard": "儀表板",
        "nav_settings": "設定",
        "title": APP_TITLE,
        "subtitle": "掃描 PDF → 摘要 → 建立主目錄（ToC）→ 執行代理 / 管理筆記",
        "api_status": "API 狀態",
        "connected_env": "已連線（環境變數）",
        "connected_ui": "已連線（工作階段）",
        "missing": "未提供",
        "provider": "供應商",
        "key_source": "金鑰來源",
        "enter_key": "輸入 API 金鑰",
        "save_key": "儲存金鑰",
        "clear_keys": "清除工作階段金鑰",
        "ui_theme": "主題",
        "ui_lang": "語言",
        "ui_style": "畫家風格",
        "jackpot": "幸運抽獎",
        "light": "亮色",
        "dark": "暗色",
        "english": "英文",
        "zh_tw": "繁體中文",
        "pipeline": "PDF 處理流程",
        "upload_pdfs": "上傳 PDFs",
        "upload_zip": "上傳 ZIP 資料夾",
        "path_scan": "掃描路徑（若可存取）",
        "scan": "掃描",
        "build_toc": "建立 / 更新 主目錄（ToC）",
        "trim_first_page": "裁切第一頁（封面/中繼資料）",
        "summary_prompt": "摘要提示詞",
        "model": "模型",
        "max_tokens": "最大 tokens",
        "temperature": "溫度",
        "run": "執行",
        "cancel": "取消",
        "toc": "主目錄（可編輯）",
        "toc_preview": "ToC 預覽",
        "agent_single": "單一代理執行",
        "agent_pipeline": "代理流程（逐步）",
        "select_agent": "選擇代理",
        "add_step": "新增步驟",
        "remove_last_step": "移除最後一步",
        "run_step": "執行目前步驟",
        "run_all": "執行所有步驟",
        "reset_pipeline": "重設流程",
        "input_source": "輸入來源",
        "from_toc": "主 ToC",
        "from_prev": "前一步輸出",
        "custom": "自訂",
        "output": "輸出",
        "text_view": "文字檢視",
        "markdown_view": "Markdown 檢視",
        "note_input": "貼上筆記（文字或 Markdown）",
        "transform_note": "轉換為有組織的 Markdown",
        "note_prompt": "筆記轉換提示詞",
        "keywords": "關鍵字",
        "keyword_color": "關鍵字顏色",
        "highlight_preview": "高亮預覽",
        "raw_markdown": "原始 Markdown",
        "ai_magics": "AI 魔法",
        "apply_magic": "套用魔法",
        "replace_note": "取代筆記",
        "append_section": "附加章節",
        "download_md": "下載 Markdown",
        "load_example": "載入範例（HTNF / K250507）",
        "example_loaded": "範例已載入工作區，作為類 PDF 文字樣本。",
        "status": "狀態",
        "stage": "階段",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


def _pick_style_by_id(style_id: str) -> Dict[str, str]:
    for s in PAINTER_STYLES:
        if s["id"] == style_id:
            return s
    return PAINTER_STYLES[0]


def inject_wow_css():
    theme = st.session_state.get("ui_theme", "dark")
    style = _pick_style_by_id(st.session_state.get("ui_style", "monet"))

    # Derive theme variables
    if theme == "dark":
        base_bg = "#0B1220"
        base_fg = "#E5E7EB"
        surface = "#111827"
        border = "rgba(255,255,255,0.10)"
        muted = "rgba(229,231,235,0.70)"
    else:
        base_bg = style["bg"]
        base_fg = style["fg"]
        surface = "#FFFFFF"
        border = "rgba(15,23,42,0.10)"
        muted = "rgba(17,24,39,0.70)"

    accent = style["accent"]
    accent2 = style["accent2"]

    css = f"""
    <style>
      :root {{
        --wow-bg: {base_bg};
        --wow-fg: {base_fg};
        --wow-surface: {surface};
        --wow-border: {border};
        --wow-muted: {muted};
        --wow-accent: {accent};
        --wow-accent2: {accent2};
        --wow-radius: 16px;
      }}

      .stApp {{
        background: radial-gradient(1200px 800px at 10% 0%,
          color-mix(in srgb, var(--wow-accent) 18%, transparent),
          transparent 60%),
          radial-gradient(1200px 800px at 90% 10%,
          color-mix(in srgb, var(--wow-accent2) 14%, transparent),
          transparent 60%),
          var(--wow-bg);
        color: var(--wow-fg);
      }}

      .block-container {{
        padding-top: 1.2rem;
      }}

      .wow-card {{
        background: color-mix(in srgb, var(--wow-surface) 92%, transparent);
        border: 1px solid var(--wow-border);
        border-radius: var(--wow-radius);
        padding: 14px 14px;
      }}

      .wow-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 6px 10px;
        border: 1px solid var(--wow-border);
        background: color-mix(in srgb, var(--wow-surface) 92%, transparent);
        font-size: 0.85rem;
      }}

      .wow-dot {{
        width: 10px; height: 10px;
        border-radius: 999px;
        background: var(--wow-accent);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--wow-accent) 20%, transparent);
      }}

      .wow-dot.warn {{ background: #F59E0B; box-shadow: 0 0 0 3px rgba(245,158,11,0.2); }}
      .wow-dot.bad  {{ background: #EF4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.2); }}
      .wow-dot.ok   {{ background: #22C55E; box-shadow: 0 0 0 3px rgba(34,197,94,0.2); }}

      .wow-muted {{ color: var(--wow-muted); }}

      /* Buttons accent */
      .stButton > button {{
        border-radius: 12px !important;
        border: 1px solid var(--wow-border) !important;
      }}
      .stButton > button:hover {{
        border-color: color-mix(in srgb, var(--wow-accent) 55%, var(--wow-border)) !important;
      }}

      /* Inputs */
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
        border-radius: 12px !important;
      }}

      /* Markdown links */
      a {{ color: var(--wow-accent) !important; }}

      /* Make code blocks nicer */
      pre {{
        border-radius: 12px;
        border: 1px solid var(--wow-border);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------
# 2) Example PDF (Provided Content) — for quick functional checks
# ----------------------------

EXAMPLE_HTNF_TEXT = """U.S. Food & Drug Administration
September 11–12, 2025 — Apple Inc. — K250507
Trade/Device Name: Hypertension Notification Feature (HTNF)
Regulation Number: 21 CFR 870.2380
Regulation Name: Cardiovascular Machine Learning-Based Notification Software
Regulatory Class: Class II
Product Code: QXO (original SE letter), SFR (administrative update)

Device Description (510(k) Summary)
- HTNF is an OTC software-only mobile medical application.
- Uses Apple Watch PPG data over multiple days; ML model scores qualified signals.
- iPhone aggregates risk scores to identify patterns suggestive of hypertension and notifies users.
- Intended users: adults 22+ without prior hypertension diagnosis; not for pregnancy.
- Not a diagnosis; absence of notification does not imply absence of hypertension.

Indications for Use
- Analyzes PPG data opportunistically collected by Apple Watch to identify patterns suggestive of hypertension and provides a notification.
- OTC use by adults age 22+ not previously diagnosed with hypertension.
- Not intended to replace diagnosis methods, monitor treatment effect, or provide surveillance; not for pregnancy.

Clinical Performance (pivotal validation)
- Overall Sensitivity: 41.2% (95% CI [37.2, 45.3])
- Overall Specificity: 92.3% (95% CI [90.6, 93.7])
- PPV (prevalence 31.4%): 70.9% (95% CI [65.7, 75.7])
- Enrolled subjects: 2,229; usable data for primary endpoint: 1,863
- Subgroup analyses adjusted for age, sex, BMI, race, BP; no clinically meaningful differences after adjustment for sex/race/skin tone.

Predetermined Change Control Plan (PCCP)
- No continuous learning in field; changes trained/tuned/locked pre-release.
- Defines modifications to ML modules and notification logic; includes validation methods.
Conclusion
- HTNF substantially equivalent to predicate (Viz HCM) with no new safety/effectiveness questions.
"""


# ----------------------------
# 3) Utilities: Session state, metrics, logs
# ----------------------------

def ss_init():
    # UI defaults
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("ui_lang", "en")
    st.session_state.setdefault("ui_style", "monet")

    # API keys (session-only)
    st.session_state.setdefault("openai_key", "")
    st.session_state.setdefault("gemini_key", "")
    st.session_state.setdefault("anthropic_key", "")
    st.session_state.setdefault("grok_key", "")

    # Pipeline state
    st.session_state.setdefault("pipeline_stage", "Idle")
    st.session_state.setdefault("cancel_requested", False)
    st.session_state.setdefault("processing_log", [])
    st.session_state.setdefault("pdf_items", [])  # list of dict {id, name, source, bytes?, path?, text?, summary_md?}
    st.session_state.setdefault("summaries", {})  # id -> md summary
    st.session_state.setdefault("toc_markdown", "")

    # Settings
    st.session_state.setdefault("trim_first_page", True)
    st.session_state.setdefault("summary_model", "gemini-2.5-flash")
    st.session_state.setdefault("summary_max_tokens", 1500)
    st.session_state.setdefault("summary_temperature", 0.2)
    st.session_state.setdefault(
        "summary_prompt",
        "Summarize the following regulatory/medical device document in 5–8 concise bullet points. "
        "Focus on: device name, indications for use, intended users, regulatory classification/product code, "
        "key performance/clinical results, PCCP/change control plan (if any), and notable constraints."
    )

    # Agents
    st.session_state.setdefault("agents_catalog", [])
    st.session_state.setdefault("agent_single_id", "")
    st.session_state.setdefault("agent_single_model", "gpt-4o-mini")
    st.session_state.setdefault("agent_single_max_tokens", DEFAULT_MAX_TOKENS)
    st.session_state.setdefault("agent_single_temperature", DEFAULT_TEMPERATURE)
    st.session_state.setdefault("agent_single_prompt", "")

    st.session_state.setdefault("agent_steps", [])  # list of step dicts
    st.session_state.setdefault("agent_active_step", 0)
    st.session_state.setdefault("agent_run_history", [])

    # Note Keeper
    st.session_state.setdefault("note_raw_input", "")
    st.session_state.setdefault("note_markdown", "")
    st.session_state.setdefault("note_keywords", [])
    st.session_state.setdefault("note_keyword_color", DEFAULT_KEYWORD_COLOR)
    st.session_state.setdefault("note_model", "gpt-4o-mini")
    st.session_state.setdefault("note_max_tokens", DEFAULT_MAX_TOKENS)
    st.session_state.setdefault("note_temperature", DEFAULT_TEMPERATURE)
    st.session_state.setdefault(
        "note_prompt",
        "Transform the note into well-organized Markdown with clear headings and bullet points. "
        "Include sections: Summary, Details, Action Items, Questions, References (if applicable). "
        "Extract 8–15 keywords and return them in a 'Keywords' list. Keep content faithful to the original."
    )
    st.session_state.setdefault("notes", [])  # saved notes list

    # Metrics
    st.session_state.setdefault("metrics", {
        "pdf_found": 0,
        "pdf_summarized_ok": 0,
        "pdf_single_page": 0,
        "pdf_no_text": 0,
        "pdf_errors": 0,
        "llm_runs": 0,
        "agents_executed": 0,
        "notes_created": 0,
        "magics_applied": 0,
        "last_model": "",
        "last_run_at": None,
    })


def log_event(msg: str, level: str = "info", data: Optional[Dict[str, Any]] = None):
    item = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "msg": msg,
        "data": data or {},
    }
    st.session_state.processing_log.append(item)


def set_stage(stage: str):
    st.session_state.pipeline_stage = stage


def approx_tokens(text: str) -> int:
    # Rough heuristic; good enough for dashboard estimates
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip()
    return name[:120] if len(name) > 120 else name


# ----------------------------
# 4) API Keys: env-first, UI fallback
# ----------------------------

def env_key(provider: str) -> str:
    return os.environ.get(ENV_KEYS[provider], "").strip()


def get_effective_key(provider: str) -> Tuple[str, str]:
    """
    Returns: (key, source)
    source in: env | session | missing
    """
    ek = env_key(provider)
    if ek:
        return ek, "env"
    sk = st.session_state.get(f"{provider}_key", "").strip()
    if sk:
        return sk, "session"
    return "", "missing"


def clear_session_keys():
    for p in ENV_KEYS.keys():
        st.session_state[f"{p}_key"] = ""


# ----------------------------
# 5) agents.yaml / SKILL.md loading
# ----------------------------

def load_skill_md() -> str:
    try:
        with open("SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_agents_yaml() -> List[Dict[str, Any]]:
    # Expected flexible schema; we normalize minimal fields
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}

    agents = []
    if isinstance(raw, dict) and "agents" in raw and isinstance(raw["agents"], list):
        for a in raw["agents"]:
            if not isinstance(a, dict):
                continue
            agents.append({
                "id": a.get("id") or a.get("name") or str(uuid.uuid4()),
                "name": a.get("name", "Unnamed Agent"),
                "category": a.get("category", "General"),
                "system_prompt": a.get("system_prompt", ""),
                "user_prompt_template": a.get("user_prompt_template", ""),
                "default_model": a.get("default_model", ""),
                "default_max_tokens": a.get("default_max_tokens", None),
                "default_temperature": a.get("default_temperature", None),
            })
    elif isinstance(raw, list):
        for a in raw:
            if not isinstance(a, dict):
                continue
            agents.append({
                "id": a.get("id") or a.get("name") or str(uuid.uuid4()),
                "name": a.get("name", "Unnamed Agent"),
                "category": a.get("category", "General"),
                "system_prompt": a.get("system_prompt", ""),
                "user_prompt_template": a.get("user_prompt_template", ""),
                "default_model": a.get("default_model", ""),
                "default_max_tokens": a.get("default_max_tokens", None),
                "default_temperature": a.get("default_temperature", None),
            })

    # Provide a few safe built-ins if file is missing/empty
    if not agents:
        agents = [
            {
                "id": "trend_spotter",
                "name": "Trend Spotter",
                "category": "Synthesis",
                "system_prompt": "",
                "user_prompt_template": "Read the Master ToC and identify major recurring themes, constraints, and performance patterns across documents. Return a concise report with headings.",
                "default_model": "gpt-4o-mini",
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "default_temperature": 0.2,
            },
            {
                "id": "regulatory_extractor",
                "name": "Regulatory Extractor",
                "category": "Extraction",
                "system_prompt": "",
                "user_prompt_template": "From the Master ToC, extract for each document: (1) device name, (2) regulation number, (3) class, (4) product code, (5) intended use/indications. Return a markdown table.",
                "default_model": "gemini-2.5-flash",
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "default_temperature": 0.2,
            },
        ]
    return agents


# ----------------------------
# 6) LLM Gateway: OpenAI / Gemini / Anthropic / Grok
# ----------------------------

def provider_for_model(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("grok-"):
        return "grok"
    # fallback: treat as openai-like
    return "openai"


def llm_call(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    provider = provider_for_model(model)
    key, source = get_effective_key(provider)

    if not key:
        raise RuntimeError(f"Missing API key for provider '{provider}' required by model '{model}'.")

    started = time.time()

    # Track metrics
    st.session_state.metrics["llm_runs"] += 1
    st.session_state.metrics["last_model"] = model
    st.session_state.metrics["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai SDK is not installed. Add 'openai' to requirements.txt.")
        client = OpenAI(api_key=key)
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        out = resp.choices[0].message.content or ""

    elif provider == "grok":
        if OpenAI is None:
            raise RuntimeError("openai SDK is required for Grok (OpenAI-compatible). Add 'openai' to requirements.txt.")
        client = OpenAI(api_key=key, base_url=GROK_BASE_URL)
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        out = resp.choices[0].message.content or ""

    elif provider == "gemini":
        if genai is None:
            raise RuntimeError("google-generativeai SDK not installed. Add 'google-generativeai' to requirements.txt.")
        genai.configure(api_key=key)
        # Gemini expects "contents". We emulate system instructions by prepending.
        prompt = user_prompt
        if system_prompt.strip():
            prompt = f"[System Instructions]\n{system_prompt}\n\n[User]\n{user_prompt}"
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": int(max_tokens),
                "temperature": float(temperature),
            },
        )
        out = (getattr(resp, "text", None) or "").strip()

    elif provider == "anthropic":
        if Anthropic is None:
            raise RuntimeError("anthropic SDK not installed. Add 'anthropic' to requirements.txt.")
        client = Anthropic(api_key=key)
        # Anthropic supports system field directly
        resp = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        # resp.content is list of blocks; concatenate text blocks
        parts = []
        for b in getattr(resp, "content", []) or []:
            if getattr(b, "type", "") == "text":
                parts.append(getattr(b, "text", ""))
            else:
                # best-effort fallback
                parts.append(str(b))
        out = "".join(parts).strip()

    else:
        raise RuntimeError(f"Unknown provider '{provider}' for model '{model}'.")

    elapsed = time.time() - started
    log_event("LLM call completed", data={
        "provider": provider,
        "model": model,
        "key_source": source,
        "elapsed_sec": round(elapsed, 3),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "prompt_tokens_est": approx_tokens(system_prompt + "\n" + user_prompt),
        "output_tokens_est": approx_tokens(out),
    })
    return out


# ----------------------------
# 7) PDF Processing: read, trim, extract text
# ----------------------------

def read_pdf_text(pdf_bytes: bytes, trim_first_page: bool = True) -> Tuple[str, Dict[str, Any]]:
    meta = {"page_count": 0, "trimmed": False, "single_page": False, "no_text": False}
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages
    meta["page_count"] = len(pages)

    start_idx = 0
    if trim_first_page and len(pages) > 1:
        start_idx = 1
        meta["trimmed"] = True
    if len(pages) == 1:
        meta["single_page"] = True

    texts = []
    for i in range(start_idx, len(pages)):
        try:
            txt = pages[i].extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            texts.append(txt)

    full = "\n\n".join(texts).strip()
    if not full:
        meta["no_text"] = True
        full = "[Scanned content - Text unavailable without OCR]"
    return full, meta


def add_pdf_item_from_upload(name: str, pdf_bytes: bytes):
    item_id = hashlib.sha256((name + str(len(pdf_bytes)) + str(time.time())).encode("utf-8")).hexdigest()[:16]
    st.session_state.pdf_items.append({
        "id": item_id,
        "name": safe_filename(name),
        "source": "upload",
        "bytes": pdf_bytes,
        "path": None,
        "text": None,
        "meta": {},
    })
    st.session_state.metrics["pdf_found"] = len(st.session_state.pdf_items)


def discover_pdfs_from_zip(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    found = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".pdf"):
                found.append((info.filename, z.read(info.filename)))
    return found


def discover_pdfs_from_path(root: str) -> List[str]:
    pdfs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, fn))
    return pdfs


# ----------------------------
# 8) Summarization + ToC Builder
# ----------------------------

def summarize_document(doc_name: str, doc_text: str) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill else "") + "You are a precise regulatory document summarizer."
    user_prompt = f"{st.session_state.summary_prompt}\n\n[Document: {doc_name}]\n\n{doc_text}"
    return llm_call(
        model=st.session_state.summary_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=int(st.session_state.summary_max_tokens),
        temperature=float(st.session_state.summary_temperature),
    )


def build_master_toc() -> str:
    entries = []
    for idx, item in enumerate(st.session_state.pdf_items, start=1):
        sid = item["id"]
        name = item["name"]
        md = st.session_state.summaries.get(sid, "").strip()
        if not md:
            md = "_(No summary yet)_"
        entries.append(f"## {idx}. {name}\n\n{md}\n")
    return "# Master Table of Contents\n\n" + "\n".join(entries)


# ----------------------------
# 9) Agent Execution: single + pipeline
# ----------------------------

def normalize_agent_prompt(agent: Dict[str, Any], user_edited_prompt: str, context: str) -> Tuple[str, str]:
    # Compose system prompt from SKILL.md + agent system prompt
    skill = load_skill_md()
    system_prompt = ""
    if skill.strip():
        system_prompt += skill.strip() + "\n\n"
    if (agent.get("system_prompt") or "").strip():
        system_prompt += agent["system_prompt"].strip()

    # Compose user prompt
    template = user_edited_prompt.strip() if user_edited_prompt.strip() else (agent.get("user_prompt_template") or "")
    user_prompt = template.strip()
    if "{context}" in user_prompt:
        user_prompt = user_prompt.replace("{context}", context)
    else:
        user_prompt = f"{user_prompt}\n\n[Context]\n{context}"
    return system_prompt, user_prompt


def run_agent_once(agent_id: str, prompt: str, model: str, max_tokens: int, temperature: float, context: str) -> str:
    agent = next((a for a in st.session_state.agents_catalog if a["id"] == agent_id), None)
    if not agent:
        raise RuntimeError("Agent not found.")
    system_prompt, user_prompt = normalize_agent_prompt(agent, prompt, context)
    out = llm_call(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )
    st.session_state.metrics["agents_executed"] += 1
    st.session_state.agent_run_history.append({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "agent_id": agent_id,
        "agent_name": agent.get("name"),
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "context_tokens_est": approx_tokens(context),
        "output_tokens_est": approx_tokens(out),
    })
    return out


def default_step_dict() -> Dict[str, Any]:
    # Use first agent if exists
    agent_id = st.session_state.agents_catalog[0]["id"] if st.session_state.agents_catalog else ""
    agent = next((a for a in st.session_state.agents_catalog if a["id"] == agent_id), None)
    return {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "model": (agent.get("default_model") if agent else "") or st.session_state.agent_single_model,
        "max_tokens": int((agent.get("default_max_tokens") if agent else None) or DEFAULT_MAX_TOKENS),
        "temperature": float((agent.get("default_temperature") if agent else None) or DEFAULT_TEMPERATURE),
        "prompt": (agent.get("user_prompt_template") if agent else "") or "",
        "input_mode": "prev",  # prev | toc | custom
        "custom_input": "",
        "output_text": "",
        "output_md": "",
        "last_run_at": None,
    }


# ----------------------------
# 10) Note Keeper: transform + magics + keyword highlight
# ----------------------------

def extract_keywords_from_markdown(md: str) -> List[str]:
    # Heuristic: look for a Keywords section list, else fallback to empty
    # Accept patterns like "Keywords:" or "## Keywords"
    kw = []

    # "## Keywords" section
    m = re.search(r"(?is)^\s*##\s*Keywords\s*(.*?)(?:\n##\s|\Z)", md)
    if m:
        block = m.group(1)
        # extract list items
        for line in block.splitlines():
            line = line.strip()
            line = re.sub(r"^[\-\*\d\.\)]\s*", "", line)
            if line:
                # split comma-separated too
                parts = [p.strip() for p in re.split(r"[,\u3001]", line) if p.strip()]
                kw.extend(parts)

    # "Keywords:" inline
    if not kw:
        m2 = re.search(r"(?is)\bKeywords?\s*:\s*(.+)", md)
        if m2:
            parts = [p.strip() for p in re.split(r"[,\u3001]", m2.group(1)) if p.strip()]
            kw.extend(parts)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in kw:
        k2 = k.strip()
        if not k2:
            continue
        k_norm = k2.lower()
        if k_norm not in seen:
            seen.add(k_norm)
            out.append(k2)
    return out[:25]


def highlight_keywords_html(md: str, keywords: List[str], color: str) -> str:
    # Convert markdown to "pseudo HTML" by injecting span tags into the raw markdown.
    # Streamlit st.markdown can render HTML when unsafe_allow_html=True.
    if not md.strip() or not keywords:
        return md

    # Sort by length desc to avoid partial overlaps
    kws = sorted([k for k in keywords if k.strip()], key=lambda x: len(x), reverse=True)

    def repl(text: str) -> str:
        for k in kws:
            pat = re.escape(k)
            # Case-insensitive replace with word-boundary best effort for simple terms.
            # For multi-word phrases, don't enforce word boundaries.
            if re.search(r"\s", k):
                regex = re.compile(pat, re.IGNORECASE)
            else:
                regex = re.compile(rf"\b{pat}\b", re.IGNORECASE)
            text = regex.sub(lambda m: f"<span style='color:{color}; font-weight:700;'>{m.group(0)}</span>", text)
        return text

    return repl(md)


def note_transform(raw_note: str, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill else "") + "You are an expert note organizer. Preserve factual meaning."
    user_prompt = f"{prompt}\n\n[Note]\n{raw_note}"
    return llm_call(model=model, system_prompt=system_prompt, user_prompt=user_prompt,
                    max_tokens=int(max_tokens), temperature=float(temperature))


def magic_apply(kind: str, note_md: str, model: str, max_tokens: int, temperature: float, extra: Dict[str, Any]) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill else "") + "You are a careful editor. Output valid Markdown only."

    if kind == "ai_keywords":
        kws = extra.get("keywords", [])
        # This magic is presentation + optional insertion: return note with Keywords section appended if missing
        kws_clean = [k.strip() for k in kws if k.strip()]
        if not kws_clean:
            return note_md
        if re.search(r"(?im)^\s*##\s*Keywords\b", note_md):
            # replace existing keywords section
            note_md = re.sub(r"(?is)^\s*##\s*Keywords\s*.*?(?=\n##\s|\Z)",
                             "## Keywords\n" + "\n".join([f"- {k}" for k in kws_clean]) + "\n",
                             note_md)
            return note_md
        return note_md.rstrip() + "\n\n## Keywords\n" + "\n".join([f"- {k}" for k in kws_clean]) + "\n"

    if kind == "action_items":
        user_prompt = (
            "Extract actionable tasks from the note. Create/replace a section:\n"
            "## Action Items\n"
            "- [ ] Task (Owner: ..., Due: ...)\n\n"
            "If owner/due date not present, leave them blank but keep the fields.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "meeting_minutes":
        user_prompt = (
            "Rewrite the note as structured meeting minutes with sections:\n"
            "## Attendees\n## Agenda\n## Decisions\n## Risks\n## Next Steps\n\n"
            "Be faithful to content; do not invent details.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "flashcards":
        user_prompt = (
            "Generate 8–15 flashcards from the note as Markdown:\n"
            "## Flashcards\n**Q:** ...\n**A:** ...\n\n"
            "Focus on definitions, key numbers, constraints, and concepts. Do not invent facts.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "exec_summary":
        length = extra.get("length", 100)
        user_prompt = (
            f"Write an executive summary of about {length} words as Markdown section:\n"
            "## Executive Summary\n...\n\n"
            "Be faithful; do not invent.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "clarity_improve":
        user_prompt = (
            "Improve clarity and structure while preserving meaning. Output:\n"
            "1) A rewritten version of the note in clean Markdown\n"
            "2) A section at the end:\n"
            "## Clarity Improvements\n- Bullet list of what you changed and why\n\n"
            "Do not invent facts.\n\n"
            f"[Note]\n{note_md}"
        )
    else:
        return note_md

    out = llm_call(model=model, system_prompt=system_prompt, user_prompt=user_prompt,
                   max_tokens=int(max_tokens), temperature=float(temperature))
    st.session_state.metrics["magics_applied"] += 1
    return out


# ----------------------------
# 11) UI Components
# ----------------------------

def wow_header():
    st.markdown(f"## {t('title')}")
    st.markdown(f"<div class='wow-muted'>{t('subtitle')}</div>", unsafe_allow_html=True)

    # Status strip: provider connectivity + stage
    stage = st.session_state.get("pipeline_stage", "Idle")

    def provider_badge(provider: str, label: str) -> str:
        _, src = get_effective_key(provider)
        if src == "env":
            dot = "ok"
            txt = t("connected_env")
        elif src == "session":
            dot = "ok"
            txt = t("connected_ui")
        else:
            dot = "bad"
            txt = t("missing")
        return f"""
        <span class="wow-pill">
          <span class="wow-dot {dot}"></span>
          <b>{label}</b>
          <span class="wow-muted">{txt}</span>
        </span>
        """

    badges = [
        provider_badge("openai", "OpenAI"),
        provider_badge("gemini", "Gemini"),
        provider_badge("anthropic", "Anthropic"),
        provider_badge("grok", "Grok"),
        f"""
        <span class="wow-pill">
          <span class="wow-dot"></span>
          <b>{t('stage')}:</b>
          <span class="wow-muted">{stage}</span>
        </span>
        """,
    ]

    st.markdown("<div style='display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 18px 0;'>" +
                "".join(badges) + "</div>", unsafe_allow_html=True)


def settings_page():
    st.markdown("### " + t("nav_settings"))
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        theme = st.selectbox(t("ui_theme"), [t("dark"), t("light")],
                             index=0 if st.session_state.ui_theme == "dark" else 1)
        st.session_state.ui_theme = "dark" if theme == t("dark") else "light"

    with c2:
        lang = st.selectbox(t("ui_lang"), [t("english"), t("zh_tw")],
                            index=0 if st.session_state.ui_lang == "en" else 1)
        st.session_state.ui_lang = "en" if lang == t("english") else "zh-TW"

    with c3:
        style_names = [s["name"] for s in PAINTER_STYLES]
        style_idx = next((i for i, s in enumerate(PAINTER_STYLES) if s["id"] == st.session_state.ui_style), 0)
        picked = st.selectbox(t("ui_style"), style_names, index=style_idx)
        st.session_state.ui_style = next(s["id"] for s in PAINTER_STYLES if s["name"] == picked)

        if st.button(t("jackpot")):
            # pseudo-random but stable-ish: use current time
            seed = int(time.time() * 1000) % 10_000_000
            idx = seed % len(PAINTER_STYLES)
            st.session_state.ui_style = PAINTER_STYLES[idx]["id"]
            st.toast(f"{t('ui_style')}: {PAINTER_STYLES[idx]['name']}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### " + t("api_status"))
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)

    def key_row(provider: str, label: str):
        key, src = get_effective_key(provider)
        cols = st.columns([1.2, 1.2, 2.6, 1.2])
        with cols[0]:
            st.write(f"**{label}**")
        with cols[1]:
            st.write(src)
        with cols[2]:
            if src == "env":
                st.text_input(t("enter_key"), value="Provided via environment", disabled=True, type="password",
                              key=f"{provider}_env_masked")
            else:
                st.text_input(t("enter_key"), type="password", key=f"{provider}_key")
        with cols[3]:
            if src != "env":
                if st.button(t("save_key"), key=f"save_{provider}"):
                    # The text_input already stored into session_state via key
                    st.toast(f"{label}: {t('connected_ui') if st.session_state.get(f'{provider}_key') else t('missing')}")

    key_row("openai", "OpenAI")
    key_row("gemini", "Gemini")
    key_row("anthropic", "Anthropic")
    key_row("grok", "Grok")

    cA, cB = st.columns([1, 3])
    with cA:
        if st.button(t("clear_keys")):
            clear_session_keys()
            st.toast("Session keys cleared")

    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Tip: On Hugging Face Spaces, set API keys as Secrets to provide them via environment variables.")


def workspace_page():
    st.markdown("### " + t("nav_workspace"))

    # Controls
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    c0, c1, c2 = st.columns([1.4, 1.4, 1.0])

    with c0:
        up_pdfs = st.file_uploader(t("upload_pdfs"), type=["pdf"], accept_multiple_files=True)
    with c1:
        up_zip = st.file_uploader(t("upload_zip"), type=["zip"], accept_multiple_files=False)
    with c2:
        if st.button(t("load_example")):
            # Load example as a pseudo "PDF item" with text already available
            item_id = "example_htnf"
            exists = any(x["id"] == item_id for x in st.session_state.pdf_items)
            if not exists:
                st.session_state.pdf_items.append({
                    "id": item_id,
                    "name": "Example - K250507 HTNF (FDA letter + 510(k) summary)",
                    "source": "example_text",
                    "bytes": None,
                    "path": None,
                    "text": EXAMPLE_HTNF_TEXT,
                    "meta": {"page_count": None, "trimmed": None, "single_page": None, "no_text": False},
                })
                st.session_state.metrics["pdf_found"] = len(st.session_state.pdf_items)
                st.toast(t("example_loaded"))

    # Path scan (optional)
    path = st.text_input(t("path_scan"), value="")
    path_scan_btn = st.button(t("scan"))

    trim = st.toggle(t("trim_first_page"), value=st.session_state.trim_first_page)
    st.session_state.trim_first_page = trim

    st.markdown("</div>", unsafe_allow_html=True)

    # Ingest uploads
    if up_pdfs:
        for f in up_pdfs:
            add_pdf_item_from_upload(f.name, f.read())
        st.toast(f"Added {len(up_pdfs)} PDF(s)")

    if up_zip is not None:
        try:
            found = discover_pdfs_from_zip(up_zip.read())
            for name, b in found:
                add_pdf_item_from_upload(name, b)
            st.toast(f"ZIP: found {len(found)} PDF(s)")
        except Exception as e:
            log_event("ZIP extraction failed", level="error", data={"error": str(e)})
            st.error(f"ZIP extraction failed: {e}")

    if path_scan_btn and path.strip():
        try:
            set_stage("Scanning")
            pdf_paths = discover_pdfs_from_path(path.strip())
            # We store as paths (bytes not loaded until processing)
            for p in pdf_paths:
                item_id = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
                if any(x["id"] == item_id for x in st.session_state.pdf_items):
                    continue
                st.session_state.pdf_items.append({
                    "id": item_id,
                    "name": os.path.basename(p),
                    "source": "path",
                    "bytes": None,
                    "path": p,
                    "text": None,
                    "meta": {},
                })
            st.session_state.metrics["pdf_found"] = len(st.session_state.pdf_items)
            set_stage("Idle")
            st.toast(f"Path scan found {len(pdf_paths)} PDF(s)")
        except Exception as e:
            set_stage("Idle")
            log_event("Path scan failed", level="error", data={"error": str(e)})
            st.error(f"Path scan failed: {e}")

    # Pipeline settings
    st.markdown("<div class='wow-card' style='margin-top: 12px;'>", unsafe_allow_html=True)
    st.markdown(f"#### {t('pipeline')}")
    cA, cB, cC, cD = st.columns([1.4, 1.4, 1.4, 1.0])
    with cA:
        st.selectbox(t("model"), SUPPORTED_MODELS, key="summary_model",
                     index=max(0, SUPPORTED_MODELS.index(st.session_state.summary_model)) if st.session_state.summary_model in SUPPORTED_MODELS else 0)
    with cB:
        st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256, key="summary_max_tokens")
    with cC:
        st.slider(t("temperature"), min_value=0.0, max_value=1.0, value=float(st.session_state.summary_temperature),
                  step=0.05, key="summary_temperature")
    with cD:
        run_btn = st.button(t("run"), use_container_width=True)
        cancel_btn = st.button(t("cancel"), use_container_width=True)
        if cancel_btn:
            st.session_state.cancel_requested = True
            st.toast("Cancel requested")

    st.text_area(t("summary_prompt"), key="summary_prompt", height=120)

    st.markdown("</div>", unsafe_allow_html=True)

    # Run pipeline: extract text + summarize
    if run_btn:
        st.session_state.cancel_requested = False
        if not st.session_state.pdf_items:
            st.warning("No PDFs loaded yet.")
        else:
            try:
                set_stage("Trimming / Extracting")
                ok = 0
                no_text = 0
                single_page = 0
                errors = 0
                total = len(st.session_state.pdf_items)
                prog = st.progress(0)
                status = st.status("Processing PDFs...", expanded=True)

                for i, item in enumerate(st.session_state.pdf_items, start=1):
                    if st.session_state.cancel_requested:
                        status.update(label="Canceled by user", state="error")
                        break

                    sid = item["id"]
                    name = item["name"]

                    # If example_text already has text, skip extraction
                    if item["source"] == "example_text" and item.get("text"):
                        text = item["text"]
                        meta = item.get("meta", {})
                    else:
                        # Load bytes from upload or path
                        try:
                            if item["source"] == "upload":
                                pdf_bytes = item["bytes"]
                            elif item["source"] == "path":
                                with open(item["path"], "rb") as f:
                                    pdf_bytes = f.read()
                            else:
                                pdf_bytes = item.get("bytes")
                            text, meta = read_pdf_text(pdf_bytes, trim_first_page=st.session_state.trim_first_page)
                            item["text"] = text
                            item["meta"] = meta
                            if meta.get("no_text"):
                                no_text += 1
                            if meta.get("single_page"):
                                single_page += 1
                        except Exception as e:
                            errors += 1
                            log_event("PDF extraction failed", level="error", data={"name": name, "error": str(e)})
                            status.write(f"❌ Extraction failed: {name} — {e}")
                            prog.progress(i / total)
                            continue

                    # Summarize
                    try:
                        set_stage("Summarizing")
                        status.write(f"🧠 Summarizing: {name}")
                        md = summarize_document(name, text)
                        st.session_state.summaries[sid] = md
                        ok += 1
                    except Exception as e:
                        errors += 1
                        log_event("Summarization failed", level="error", data={"name": name, "error": str(e)})
                        status.write(f"❌ Summarization failed: {name} — {e}")
                    prog.progress(i / total)

                st.session_state.metrics["pdf_summarized_ok"] += ok
                st.session_state.metrics["pdf_no_text"] += no_text
                st.session_state.metrics["pdf_single_page"] += single_page
                st.session_state.metrics["pdf_errors"] += errors

                set_stage("Building ToC")
                st.session_state.toc_markdown = build_master_toc()
                set_stage("Idle")
                status.update(label="Done", state="complete")
                st.toast("ToC updated")
            except Exception as e:
                set_stage("Idle")
                log_event("Pipeline run failed", level="error", data={"error": str(e)})
                st.error(f"Pipeline failed: {e}")

    # ToC editor + preview
    st.markdown("<div class='wow-card' style='margin-top: 12px;'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### " + t("toc"))
        st.text_area("", key="toc_markdown", height=420)
        if st.button(t("build_toc")):
            st.session_state.toc_markdown = build_master_toc()
            st.toast("ToC refreshed")
    with c2:
        st.markdown("#### " + t("toc_preview"))
        st.markdown(st.session_state.toc_markdown or "_(empty)_")
    st.markdown("</div>", unsafe_allow_html=True)

    # Logs
    with st.expander("Logs"):
        for item in st.session_state.processing_log[-200:]:
            st.write(f"[{item['ts']}] {item['level'].upper()}: {item['msg']}  {json.dumps(item['data'], ensure_ascii=False)}")


def agent_studio_page():
    st.markdown("### " + t("nav_agent_studio"))

    toc = st.session_state.toc_markdown.strip() or "(Master ToC is empty. Build it in Workspace first.)"

    # Single Agent Run
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    st.markdown("#### " + t("agent_single"))

    agents = st.session_state.agents_catalog
    agent_names = [f"{a['name']}  ·  {a.get('category','')}" for a in agents]
    agent_ids = [a["id"] for a in agents]

    if agents:
        idx = 0
        if st.session_state.agent_single_id in agent_ids:
            idx = agent_ids.index(st.session_state.agent_single_id)

        picked = st.selectbox(t("select_agent"), agent_names, index=idx)
        st.session_state.agent_single_id = agent_ids[agent_names.index(picked)]
        agent = agents[agent_ids.index(st.session_state.agent_single_id)]

        # Auto-load template if empty
        if not st.session_state.agent_single_prompt.strip():
            st.session_state.agent_single_prompt = agent.get("user_prompt_template", "")

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        st.selectbox(t("model"), SUPPORTED_MODELS, key="agent_single_model",
                     index=SUPPORTED_MODELS.index(st.session_state.agent_single_model) if st.session_state.agent_single_model in SUPPORTED_MODELS else 0)
    with c2:
        st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256, key="agent_single_max_tokens")
    with c3:
        st.slider(t("temperature"), 0.0, 1.0, float(st.session_state.agent_single_temperature), 0.05, key="agent_single_temperature")

    st.text_area(t("summary_prompt"), key="agent_single_prompt", height=160)

    if st.button(t("run"), key="run_single_agent"):
        try:
            set_stage("Agent Running")
            out = run_agent_once(
                agent_id=st.session_state.agent_single_id,
                prompt=st.session_state.agent_single_prompt,
                model=st.session_state.agent_single_model,
                max_tokens=int(st.session_state.agent_single_max_tokens),
                temperature=float(st.session_state.agent_single_temperature),
                context=toc,
            )
            set_stage("Idle")
            st.markdown("#### " + t("output"))
            st.markdown(out)
        except Exception as e:
            set_stage("Idle")
            log_event("Agent run failed", level="error", data={"error": str(e)})
            st.error(f"Agent run failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Pipeline Mode
    st.markdown("<div class='wow-card' style='margin-top: 12px;'>", unsafe_allow_html=True)
    st.markdown("#### " + t("agent_pipeline"))

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button(t("add_step")):
            st.session_state.agent_steps.append(default_step_dict())
            st.session_state.agent_active_step = max(0, len(st.session_state.agent_steps) - 1)
    with cB:
        if st.button(t("remove_last_step")) and st.session_state.agent_steps:
            st.session_state.agent_steps.pop()
            st.session_state.agent_active_step = max(0, min(st.session_state.agent_active_step, len(st.session_state.agent_steps) - 1))
    with cC:
        if st.button(t("reset_pipeline")):
            st.session_state.agent_steps = []
            st.session_state.agent_active_step = 0

    if not st.session_state.agent_steps:
        st.info("Add steps to build a pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    step_labels = [f"Step {i+1}: {next((a['name'] for a in agents if a['id']==s['agent_id']), 'Agent')}" for i, s in enumerate(st.session_state.agent_steps)]
    st.session_state.agent_active_step = st.selectbox(
        "Active step",
        list(range(len(step_labels))),
        format_func=lambda i: step_labels[i],
        index=min(st.session_state.agent_active_step, len(step_labels)-1),
    )

    s = st.session_state.agent_steps[st.session_state.agent_active_step]

    # Step config UI
    agent_idx = 0
    if s["agent_id"] in agent_ids:
        agent_idx = agent_ids.index(s["agent_id"])
    picked_agent = st.selectbox(t("select_agent"), agent_names, index=agent_idx, key=f"pipe_agent_{s['id']}")
    s["agent_id"] = agent_ids[agent_names.index(picked_agent)]

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        picked_model = st.selectbox(t("model"), SUPPORTED_MODELS,
                                    index=SUPPORTED_MODELS.index(s["model"]) if s["model"] in SUPPORTED_MODELS else 0,
                                    key=f"pipe_model_{s['id']}")
        s["model"] = picked_model
    with c2:
        s["max_tokens"] = st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256,
                                          value=int(s["max_tokens"]), key=f"pipe_max_{s['id']}")
    with c3:
        s["temperature"] = st.slider(t("temperature"), 0.0, 1.0, float(s["temperature"]), 0.05, key=f"pipe_temp_{s['id']}")

    s["prompt"] = st.text_area("Prompt (editable)", value=s.get("prompt", ""), height=140, key=f"pipe_prompt_{s['id']}")

    s["input_mode"] = st.radio(
        t("input_source"),
        options=["prev", "toc", "custom"],
        index=["prev", "toc", "custom"].index(s.get("input_mode", "prev")),
        format_func=lambda x: { "prev": t("from_prev"), "toc": t("from_toc"), "custom": t("custom") }.get(x, x),
        key=f"pipe_inmode_{s['id']}"
    )
    if s["input_mode"] == "custom":
        s["custom_input"] = st.text_area("Custom input", value=s.get("custom_input", ""), height=120, key=f"pipe_custom_{s['id']}")

    # Determine input context for this step
    def step_input(i: int) -> str:
        step = st.session_state.agent_steps[i]
        if step["input_mode"] == "toc":
            return toc
        if step["input_mode"] == "custom":
            return step.get("custom_input", "")
        # prev
        if i == 0:
            return toc
        prev = st.session_state.agent_steps[i-1]
        # Prefer user-edited "output_text" if present; else "output_md"
        return (prev.get("output_text") or prev.get("output_md") or toc)

    cRun1, cRun2 = st.columns([1, 1])
    with cRun1:
        if st.button(t("run_step")):
            try:
                set_stage("Agent Running")
                i = st.session_state.agent_active_step
                inp = step_input(i)
                out = run_agent_once(
                    agent_id=s["agent_id"],
                    prompt=s["prompt"],
                    model=s["model"],
                    max_tokens=int(s["max_tokens"]),
                    temperature=float(s["temperature"]),
                    context=inp,
                )
                s["output_md"] = out
                s["output_text"] = out
                s["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                set_stage("Idle")
                st.toast("Step completed")
            except Exception as e:
                set_stage("Idle")
                log_event("Pipeline step failed", level="error", data={"error": str(e)})
                st.error(f"Step failed: {e}")

    with cRun2:
        if st.button(t("run_all")):
            try:
                set_stage("Agent Running")
                for i in range(len(st.session_state.agent_steps)):
                    step = st.session_state.agent_steps[i]
                    inp = step_input(i)
                    out = run_agent_once(
                        agent_id=step["agent_id"],
                        prompt=step["prompt"],
                        model=step["model"],
                        max_tokens=int(step["max_tokens"]),
                        temperature=float(step["temperature"]),
                        context=inp,
                    )
                    step["output_md"] = out
                    step["output_text"] = out
                    step["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                set_stage("Idle")
                st.toast("All steps completed")
            except Exception as e:
                set_stage("Idle")
                log_event("Pipeline run_all failed", level="error", data={"error": str(e)})
                st.error(f"Run all failed: {e}")

    # Output editing (text + markdown)
    st.markdown("#### " + t("output"))
    view = st.radio("View", ["text", "markdown"], index=1, horizontal=True,
                    format_func=lambda x: t("text_view") if x == "text" else t("markdown_view"),
                    key=f"pipe_view_{s['id']}")
    if view == "text":
        s["output_text"] = st.text_area("", value=s.get("output_text", ""), height=240, key=f"pipe_outtext_{s['id']}")
    else:
        s["output_md"] = st.text_area("Markdown (editable)", value=s.get("output_md", ""), height=240, key=f"pipe_outmd_{s['id']}")
        st.markdown("Preview:")
        st.markdown(s["output_md"] or "_(empty)_")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Agent run history"):
        st.json(st.session_state.agent_run_history[-50:])


def note_keeper_page():
    st.markdown("### " + t("nav_note_keeper"))

    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    st.text_area(t("note_input"), key="note_raw_input", height=220)

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        st.selectbox(t("model"), SUPPORTED_MODELS, key="note_model",
                     index=SUPPORTED_MODELS.index(st.session_state.note_model) if st.session_state.note_model in SUPPORTED_MODELS else 0)
    with c2:
        st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256, key="note_max_tokens")
    with c3:
        st.slider(t("temperature"), 0.0, 1.0, float(st.session_state.note_temperature), 0.05, key="note_temperature")

    st.text_area(t("note_prompt"), key="note_prompt", height=140)

    if st.button(t("transform_note")):
        try:
            set_stage("Note Keeper Running")
            md = note_transform(
                raw_note=st.session_state.note_raw_input,
                prompt=st.session_state.note_prompt,
                model=st.session_state.note_model,
                max_tokens=int(st.session_state.note_max_tokens),
                temperature=float(st.session_state.note_temperature),
            )
            st.session_state.note_markdown = md
            kws = extract_keywords_from_markdown(md)
            st.session_state.note_keywords = kws
            st.session_state.metrics["notes_created"] += 1
            set_stage("Idle")
            st.toast("Note transformed")
        except Exception as e:
            set_stage("Idle")
            log_event("Note transform failed", level="error", data={"error": str(e)})
            st.error(f"Note transform failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Editor + preview with highlighting
    st.markdown("<div class='wow-card' style='margin-top: 12px;'>", unsafe_allow_html=True)
    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("#### " + t("raw_markdown"))
        st.text_area("", key="note_markdown", height=360)
        st.color_picker(t("keyword_color"), key="note_keyword_color")
        st.text_input(t("keywords"), value=", ".join(st.session_state.note_keywords), key="note_keywords_input")
        # Update keywords from input
        raw_kws = st.session_state.note_keywords_input
        st.session_state.note_keywords = [k.strip() for k in re.split(r"[,;\n\u3001]", raw_kws) if k.strip()][:50]

        download_name = safe_filename("note.md")
        st.download_button(
            t("download_md"),
            data=(st.session_state.note_markdown or "").encode("utf-8"),
            file_name=download_name,
            mime="text/markdown",
        )

    with cB:
        st.markdown("#### " + t("highlight_preview"))
        highlighted = highlight_keywords_html(
            st.session_state.note_markdown or "",
            st.session_state.note_keywords or [],
            st.session_state.note_keyword_color or DEFAULT_KEYWORD_COLOR,
        )
        st.markdown(highlighted or "_(empty)_", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # AI Magics
    st.markdown("<div class='wow-card' style='margin-top: 12px;'>", unsafe_allow_html=True)
    st.markdown("#### " + t("ai_magics"))

    magic = st.selectbox(
        "Magic",
        options=[
            ("ai_keywords", "AI Keywords (custom keywords + color)"),
            ("action_items", "Action Items Extractor"),
            ("meeting_minutes", "Meeting Minutes Structurer"),
            ("flashcards", "Flashcards Generator"),
            ("exec_summary", "Executive Summary (50/100/200 words)"),
            ("clarity_improve", "Critique & Improve Clarity"),
        ],
        format_func=lambda x: x[1],
    )[0]

    apply_mode = st.radio("Apply mode", ["replace", "append"], horizontal=True,
                          format_func=lambda x: t("replace_note") if x == "replace" else t("append_section"))

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        magic_model = st.selectbox("Magic model", SUPPORTED_MODELS,
                                  index=SUPPORTED_MODELS.index(st.session_state.note_model) if st.session_state.note_model in SUPPORTED_MODELS else 0)
    with c2:
        magic_max = st.number_input("Magic max_tokens", min_value=256, max_value=20000, step=256, value=DEFAULT_MAX_TOKENS)
    with c3:
        magic_temp = st.slider("Magic temperature", 0.0, 1.0, float(st.session_state.note_temperature), 0.05)

    extra = {}
    if magic == "ai_keywords":
        kws = st.text_area("Keywords (comma or newline)", value="\n".join(st.session_state.note_keywords), height=90)
        extra["keywords"] = [k.strip() for k in re.split(r"[,;\n\u3001]", kws) if k.strip()]
        # color is already handled in preview; here only inserts/updates Keywords section
    elif magic == "exec_summary":
        length = st.selectbox("Length (words)", [50, 100, 200], index=1)
        extra["length"] = length

    if st.button(t("apply_magic")):
        try:
            set_stage("Note Keeper Running")
            out = magic_apply(
                kind=magic,
                note_md=st.session_state.note_markdown or "",
                model=magic_model,
                max_tokens=int(magic_max),
                temperature=float(magic_temp),
                extra=extra,
            )
            if apply_mode == "replace":
                st.session_state.note_markdown = out
            else:
                st.session_state.note_markdown = (st.session_state.note_markdown or "").rstrip() + "\n\n" + out.strip() + "\n"
            # refresh keywords heuristic after change
            st.session_state.note_keywords = extract_keywords_from_markdown(st.session_state.note_markdown) or st.session_state.note_keywords
            set_stage("Idle")
            st.toast("Magic applied")
        except Exception as e:
            set_stage("Idle")
            log_event("Magic failed", level="error", data={"error": str(e)})
            st.error(f"Magic failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def dashboard_page():
    st.markdown("### " + t("nav_dashboard"))
    m = st.session_state.metrics

    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("PDFs Found", m["pdf_found"])
    with c2:
        st.metric("Summarized OK", m["pdf_summarized_ok"])
    with c3:
        st.metric("No Text (scanned)", m["pdf_no_text"])
    with c4:
        st.metric("Errors", m["pdf_errors"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("LLM Runs", m["llm_runs"])
    with c2:
        st.metric("Agents Executed", m["agents_executed"])
    with c3:
        st.metric("Notes Created", m["notes_created"])
    with c4:
        st.metric("Magics Applied", m["magics_applied"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.write("Last model:", m.get("last_model") or "-")
    st.write("Last run at:", m.get("last_run_at") or "-")
    st.write("Current stage:", st.session_state.pipeline_stage)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.write("Recent log entries")
    for item in st.session_state.processing_log[-30:][::-1]:
        st.write(f"[{item['ts']}] {item['level'].upper()}: {item['msg']}")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 12) Main App
# ----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ss_init()
    inject_wow_css()

    # Load agents once per session (or refresh if needed)
    if not st.session_state.agents_catalog:
        st.session_state.agents_catalog = load_agents_yaml()
        # Initialize single agent selection
        st.session_state.agent_single_id = st.session_state.agents_catalog[0]["id"]
        st.session_state.agent_single_prompt = st.session_state.agents_catalog[0].get("user_prompt_template", "")

    wow_header()

    nav = st.radio(
        "Navigation",
        options=["workspace", "agent", "note", "dashboard", "settings"],
        horizontal=True,
        format_func=lambda x: {
            "workspace": t("nav_workspace"),
            "agent": t("nav_agent_studio"),
            "note": t("nav_note_keeper"),
            "dashboard": t("nav_dashboard"),
            "settings": t("nav_settings"),
        }[x],
        label_visibility="collapsed",
    )

    if nav == "workspace":
        workspace_page()
    elif nav == "agent":
        agent_studio_page()
    elif nav == "note":
        note_keeper_page()
    elif nav == "dashboard":
        dashboard_page()
    elif nav == "settings":
        settings_page()


if __name__ == "__main__":
    main()
