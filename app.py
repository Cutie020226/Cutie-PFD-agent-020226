import os
import io
import re
import json
import time
import uuid
import math
import zipfile
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import yaml
from pypdf import PdfReader

# Optional SDK imports (guarded so the UI can still load)
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
DEFAULT_KEYWORD_COLOR = "#FF7F50"  # coral

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
    # Anthropic (examples; adjust as needed)
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

GROK_BASE_URL = "https://api.x.ai/v1"


# ----------------------------
# 1) WOW UI: Painter Styles + i18n
# ----------------------------

PAINTER_STYLES = [
    {"id": "monet", "name": "Claude Monet", "accent": "#2E86AB", "accent2": "#A23B72", "bg": "#F7F7FB", "fg": "#111827"},
    {"id": "vangogh", "name": "Vincent van Gogh", "accent": "#F2A900", "accent2": "#1B3A57", "bg": "#FFF7E6", "fg": "#111827"},
    {"id": "picasso", "name": "Pablo Picasso", "accent": "#D7263D", "accent2": "#1B998B", "bg": "#FAFAFA", "fg": "#111827"},
    {"id": "dali", "name": "Salvador Dalí", "accent": "#6D28D9", "accent2": "#F59E0B", "bg": "#F5F3FF", "fg": "#0F172A"},
    {"id": "kahlo", "name": "Frida Kahlo", "accent": "#0EA5A4", "accent2": "#E11D48", "bg": "#ECFEFF", "fg": "#0F172A"},
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
        "workspace": "Workspace",
        "agents": "Agent Studio",
        "notes": "AI Note Keeper",
        "dashboard": "Dashboard",
        "control_center": "WOW Control Center",
        "theme": "Theme",
        "language": "Language",
        "style": "Painter Style",
        "style_arcade": "Style Arcade",
        "spin": "Spin",
        "jackpot": "Jackpot",
        "light": "Light",
        "dark": "Dark",
        "english": "English",
        "zh_tw": "Traditional Chinese",
        "api_keys": "API Keys",
        "connected_env": "Connected (env)",
        "connected_session": "Connected (session)",
        "missing": "Missing",
        "enter_key": "Enter API key",
        "clear_session_keys": "Clear session keys",
        "stage": "Stage",
        "docs_loaded": "Docs loaded",
        "toc_ready": "ToC ready",
        "run_pipeline": "Run Pipeline",
        "cancel": "Cancel",
        "upload_pdfs": "Upload PDFs",
        "upload_zip": "Upload ZIP Folder",
        "scan_path": "Scan Path (if accessible)",
        "scan": "Scan",
        "trim_first_page": "Trim first page (cover/metadata)",
        "summary_prompt": "Summary prompt",
        "model": "Model",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "toc_editor": "Master ToC (editable)",
        "toc_preview": "ToC Preview",
        "refresh_toc": "Refresh ToC from summaries",
        "download_toc": "Download ToC",
        "single_agent": "Single Agent Run",
        "pipeline_agents": "Agent Pipeline",
        "select_agent": "Select agent",
        "add_step": "Add step",
        "remove_step": "Remove last step",
        "reset": "Reset",
        "run_step": "Run current step",
        "run_all": "Run all steps",
        "input_source": "Input source",
        "from_prev": "Previous output",
        "from_toc": "Master ToC",
        "custom": "Custom",
        "output": "Output",
        "note_input": "Paste note (text or markdown)",
        "transform": "Transform to organized markdown",
        "keywords": "Keywords",
        "keyword_color": "Keyword color",
        "highlight_preview": "Highlighted preview",
        "raw_markdown": "Raw markdown",
        "ai_magics": "AI Magics",
        "apply_magic": "Apply magic",
        "replace_note": "Replace note",
        "append_section": "Append section",
    },
    "zh-TW": {
        "workspace": "工作區",
        "agents": "代理工作室",
        "notes": "AI 筆記管家",
        "dashboard": "儀表板",
        "control_center": "WOW 控制中心",
        "theme": "主題",
        "language": "語言",
        "style": "畫家風格",
        "style_arcade": "風格遊戲機",
        "spin": "旋轉",
        "jackpot": "幸運抽獎",
        "light": "亮色",
        "dark": "暗色",
        "english": "英文",
        "zh_tw": "繁體中文",
        "api_keys": "API 金鑰",
        "connected_env": "已連線（環境變數）",
        "connected_session": "已連線（工作階段）",
        "missing": "未提供",
        "enter_key": "輸入 API 金鑰",
        "clear_session_keys": "清除工作階段金鑰",
        "stage": "階段",
        "docs_loaded": "已載入文件",
        "toc_ready": "ToC 已就緒",
        "run_pipeline": "執行流程",
        "cancel": "取消",
        "upload_pdfs": "上傳 PDFs",
        "upload_zip": "上傳 ZIP 資料夾",
        "scan_path": "掃描路徑（若可存取）",
        "scan": "掃描",
        "trim_first_page": "裁切第一頁（封面/中繼資料）",
        "summary_prompt": "摘要提示詞",
        "model": "模型",
        "max_tokens": "最大 tokens",
        "temperature": "溫度",
        "toc_editor": "主 ToC（可編輯）",
        "toc_preview": "ToC 預覽",
        "refresh_toc": "由摘要更新 ToC",
        "download_toc": "下載 ToC",
        "single_agent": "單一代理執行",
        "pipeline_agents": "代理流程",
        "select_agent": "選擇代理",
        "add_step": "新增步驟",
        "remove_step": "移除最後步驟",
        "reset": "重設",
        "run_step": "執行目前步驟",
        "run_all": "執行所有步驟",
        "input_source": "輸入來源",
        "from_prev": "前一步輸出",
        "from_toc": "主 ToC",
        "custom": "自訂",
        "output": "輸出",
        "note_input": "貼上筆記（文字或 Markdown）",
        "transform": "轉換為有組織的 Markdown",
        "keywords": "關鍵字",
        "keyword_color": "關鍵字顏色",
        "highlight_preview": "高亮預覽",
        "raw_markdown": "原始 Markdown",
        "ai_magics": "AI 魔法",
        "apply_magic": "套用魔法",
        "replace_note": "取代筆記",
        "append_section": "附加章節",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return I18N.get(lang, I18N["en"]).get(key, key)


def _pick_style(style_id: str) -> Dict[str, str]:
    for s in PAINTER_STYLES:
        if s["id"] == style_id:
            return s
    return PAINTER_STYLES[0]


def inject_wow_css():
    theme = st.session_state.get("ui_theme", "dark")
    style = _pick_style(st.session_state.get("ui_style", "monet"))

    if theme == "dark":
        base_bg = "#0B1220"
        base_fg = "#E5E7EB"
        surface = "#0F172A"
        surface2 = "#111827"
        border = "rgba(255,255,255,0.10)"
        muted = "rgba(229,231,235,0.70)"
    else:
        base_bg = style["bg"]
        base_fg = style["fg"]
        surface = "#FFFFFF"
        surface2 = "rgba(255,255,255,0.85)"
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
        --wow-surface2: {surface2};
        --wow-border: {border};
        --wow-muted: {muted};
        --wow-accent: {accent};
        --wow-accent2: {accent2};
        --wow-radius: 16px;
      }}

      .stApp {{
        background:
          radial-gradient(1200px 700px at 10% 0%,
            color-mix(in srgb, var(--wow-accent) 18%, transparent),
            transparent 60%),
          radial-gradient(1200px 700px at 90% 10%,
            color-mix(in srgb, var(--wow-accent2) 16%, transparent),
            transparent 60%),
          var(--wow-bg);
        color: var(--wow-fg);
      }}

      .block-container {{ padding-top: 1.0rem; }}

      .wow-hero {{
        background: var(--wow-surface2);
        border: 1px solid var(--wow-border);
        border-radius: calc(var(--wow-radius) + 6px);
        padding: 16px 16px;
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
      .wow-dot.ok   {{ background: #22C55E; box-shadow: 0 0 0 3px rgba(34,197,94,0.2); }}
      .wow-dot.bad  {{ background: #EF4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.2); }}
      .wow-dot.warn {{ background: #F59E0B; box-shadow: 0 0 0 3px rgba(245,158,11,0.2); }}

      .wow-muted {{ color: var(--wow-muted); }}

      a {{ color: var(--wow-accent) !important; }}

      .stButton > button {{
        border-radius: 12px !important;
        border: 1px solid var(--wow-border) !important;
      }}
      .stButton > button:hover {{
        border-color: color-mix(in srgb, var(--wow-accent) 55%, var(--wow-border)) !important;
      }}

      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
        border-radius: 12px !important;
      }}

      pre {{
        border-radius: 12px;
        border: 1px solid var(--wow-border);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------
# 2) Session state / metrics / logging
# ----------------------------

def ss_init():
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("ui_lang", "en")
    st.session_state.setdefault("ui_style", "monet")

    # API keys (session-only)
    st.session_state.setdefault("openai_key", "")
    st.session_state.setdefault("gemini_key", "")
    st.session_state.setdefault("anthropic_key", "")
    st.session_state.setdefault("grok_key", "")

    st.session_state.setdefault("pipeline_stage", "Idle")
    st.session_state.setdefault("cancel_requested", False)
    st.session_state.setdefault("processing_log", [])

    st.session_state.setdefault("pdf_items", [])   # dicts: {id, name, source, bytes, path, text, meta}
    st.session_state.setdefault("summaries", {})   # id -> summary md
    st.session_state.setdefault("toc_markdown", "")

    st.session_state.setdefault("trim_first_page", True)
    st.session_state.setdefault("summary_model", "gemini-2.5-flash")
    st.session_state.setdefault("summary_max_tokens", 1500)
    st.session_state.setdefault("summary_temperature", 0.2)
    st.session_state.setdefault(
        "summary_prompt",
        "Summarize the following regulatory/medical device document in 5–8 concise bullet points. "
        "Focus on: device name, indications for use, intended users, regulatory classification/product code, "
        "key performance/clinical results, PCCP/change control plan (if any), and constraints."
    )

    # Agents
    st.session_state.setdefault("agents_catalog", [])
    st.session_state.setdefault("agent_single_id", "")
    st.session_state.setdefault("agent_single_model", "gpt-4o-mini")
    st.session_state.setdefault("agent_single_max_tokens", DEFAULT_MAX_TOKENS)
    st.session_state.setdefault("agent_single_temperature", DEFAULT_TEMPERATURE)
    st.session_state.setdefault("agent_single_prompt", "")

    st.session_state.setdefault("agent_steps", [])
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
        "Extract 8–15 keywords and return them in a '## Keywords' list. Preserve meaning; do not invent facts."
    )

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
    st.session_state.processing_log.append({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "msg": msg,
        "data": data or {},
    })


def set_stage(stage: str):
    st.session_state.pipeline_stage = stage


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip()
    return name[:120] if len(name) > 120 else name


# ----------------------------
# 3) API Keys: env-first, sidebar fallback
# ----------------------------

def env_key(provider: str) -> str:
    return os.environ.get(ENV_KEYS[provider], "").strip()


def get_effective_key(provider: str) -> Tuple[str, str]:
    """
    Returns: (key, source) where source is env|session|missing
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
# 4) agents.yaml / SKILL.md
# ----------------------------

def load_skill_md() -> str:
    try:
        with open("SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_agents_yaml() -> List[Dict[str, Any]]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}

    agents = []
    if isinstance(raw, dict) and isinstance(raw.get("agents"), list):
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

    if not agents:
        agents = [
            {
                "id": "regulatory_extractor",
                "name": "Regulatory Extractor",
                "category": "Extraction",
                "system_prompt": "",
                "user_prompt_template": "From the Master ToC, extract for each document: device name, regulation number, class, product code, intended use/indications. Return a markdown table.",
                "default_model": "gemini-2.5-flash",
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "default_temperature": 0.2,
            },
            {
                "id": "compare_contrast",
                "name": "Compare & Contrast",
                "category": "Synthesis",
                "system_prompt": "",
                "user_prompt_template": "Compare documents in the Master ToC: highlight similarities, differences, and key evidence/metrics. Provide structured headings and cite the document section titles/filenames.",
                "default_model": "gpt-4o-mini",
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "default_temperature": 0.2,
            },
        ]
    return agents


# ----------------------------
# 5) LLM Gateway
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
    return "openai"


def llm_call(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
    provider = provider_for_model(model)
    key, src = get_effective_key(provider)
    if not key:
        raise RuntimeError(f"Missing API key for provider '{provider}' required by model '{model}'.")

    started = time.time()
    st.session_state.metrics["llm_runs"] += 1
    st.session_state.metrics["last_model"] = model
    st.session_state.metrics["last_run_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. Add 'openai' to requirements.txt.")
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
            raise RuntimeError("openai SDK required for Grok (OpenAI-compatible).")
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
            raise RuntimeError("google-generativeai SDK not installed.")
        genai.configure(api_key=key)
        prompt = user_prompt
        if system_prompt.strip():
            prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}"
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(
            prompt,
            generation_config={"max_output_tokens": int(max_tokens), "temperature": float(temperature)},
        )
        out = (getattr(resp, "text", None) or "").strip()

    elif provider == "anthropic":
        if Anthropic is None:
            raise RuntimeError("anthropic SDK not installed.")
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for b in getattr(resp, "content", []) or []:
            if getattr(b, "type", "") == "text":
                parts.append(getattr(b, "text", ""))
            else:
                parts.append(str(b))
        out = "".join(parts).strip()
    else:
        raise RuntimeError(f"Unknown provider '{provider}' for model '{model}'.")

    elapsed = time.time() - started
    log_event("LLM call completed", data={
        "provider": provider,
        "model": model,
        "key_source": src,
        "elapsed_sec": round(elapsed, 3),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "prompt_tokens_est": approx_tokens(system_prompt + "\n" + user_prompt),
        "output_tokens_est": approx_tokens(out),
    })
    return out


# ----------------------------
# 6) PDF discovery / extraction
# ----------------------------

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


def read_pdf_text(pdf_bytes: bytes, trim_first_page: bool) -> Tuple[str, Dict[str, Any]]:
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


# ----------------------------
# 7) Summaries + ToC
# ----------------------------

def summarize_document(doc_name: str, doc_text: str) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill.strip() else "") + "You are a precise regulatory document summarizer."
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
        md = (st.session_state.summaries.get(sid) or "").strip() or "_(No summary yet)_"
        entries.append(f"## {idx}. {name}\n\n{md}\n")
    return "# Master Table of Contents\n\n" + "\n".join(entries)


# ----------------------------
# 8) Agents: single + pipeline
# ----------------------------

def normalize_agent_prompt(agent: Dict[str, Any], user_edited_prompt: str, context: str) -> Tuple[str, str]:
    skill = load_skill_md()
    system_prompt = ""
    if skill.strip():
        system_prompt += skill.strip() + "\n\n"
    if (agent.get("system_prompt") or "").strip():
        system_prompt += agent["system_prompt"].strip()

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
    out = llm_call(model, system_prompt, user_prompt, max_tokens, temperature)
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
    agent_id = st.session_state.agents_catalog[0]["id"] if st.session_state.agents_catalog else ""
    agent = next((a for a in st.session_state.agents_catalog if a["id"] == agent_id), None)
    return {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "model": (agent.get("default_model") if agent else "") or st.session_state.agent_single_model,
        "max_tokens": int((agent.get("default_max_tokens") if agent else None) or DEFAULT_MAX_TOKENS),
        "temperature": float((agent.get("default_temperature") if agent else None) or DEFAULT_TEMPERATURE),
        "prompt": (agent.get("user_prompt_template") if agent else "") or "",
        "input_mode": "prev",   # prev|toc|custom
        "custom_input": "",
        "output_text": "",
        "output_md": "",
        "last_run_at": None,
    }


# ----------------------------
# 9) Notes + keyword highlight + magics
# ----------------------------

def extract_keywords_from_markdown(md: str) -> List[str]:
    kw = []
    m = re.search(r"(?is)^\s*##\s*Keywords\s*(.*?)(?:\n##\s|\Z)", md)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = re.sub(r"^[\-\*\d\.\)]\s*", "", line.strip())
            if line:
                kw.extend([p.strip() for p in re.split(r"[,\u3001]", line) if p.strip()])

    if not kw:
        m2 = re.search(r"(?is)\bKeywords?\s*:\s*(.+)", md)
        if m2:
            kw.extend([p.strip() for p in re.split(r"[,\u3001]", m2.group(1)) if p.strip()])

    seen, out = set(), []
    for k in kw:
        kn = k.lower()
        if kn not in seen:
            seen.add(kn)
            out.append(k)
    return out[:25]


def highlight_keywords_html(md: str, keywords: List[str], color: str) -> str:
    if not md.strip() or not keywords:
        return md
    kws = sorted([k for k in keywords if k.strip()], key=len, reverse=True)

    def repl(text: str) -> str:
        for k in kws:
            pat = re.escape(k)
            if re.search(r"\s", k):
                regex = re.compile(pat, re.IGNORECASE)
            else:
                regex = re.compile(rf"\b{pat}\b", re.IGNORECASE)
            text = regex.sub(lambda m: f"<span style='color:{color}; font-weight:700;'>{m.group(0)}</span>", text)
        return text

    return repl(md)


def note_transform(raw_note: str, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill.strip() else "") + "You are an expert note organizer. Preserve factual meaning."
    user_prompt = f"{prompt}\n\n[Note]\n{raw_note}"
    return llm_call(model, system_prompt, user_prompt, max_tokens, temperature)


def magic_apply(kind: str, note_md: str, model: str, max_tokens: int, temperature: float, extra: Dict[str, Any]) -> str:
    skill = load_skill_md()
    system_prompt = (skill + "\n\n" if skill.strip() else "") + "You are a careful editor. Output valid Markdown only."

    if kind == "ai_keywords":
        kws = [k.strip() for k in (extra.get("keywords") or []) if k.strip()]
        if not kws:
            return note_md
        if re.search(r"(?im)^\s*##\s*Keywords\b", note_md):
            return re.sub(
                r"(?is)^\s*##\s*Keywords\s*.*?(?=\n##\s|\Z)",
                "## Keywords\n" + "\n".join([f"- {k}" for k in kws]) + "\n",
                note_md
            )
        return note_md.rstrip() + "\n\n## Keywords\n" + "\n".join([f"- {k}" for k in kws]) + "\n"

    if kind == "action_items":
        user_prompt = (
            "Extract actionable tasks from the note. Create/replace a section:\n"
            "## Action Items\n- [ ] Task (Owner: ..., Due: ...)\n\n"
            "If owner/due date not present, leave them blank.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "meeting_minutes":
        user_prompt = (
            "Rewrite as structured meeting minutes:\n"
            "## Attendees\n## Agenda\n## Decisions\n## Risks\n## Next Steps\n\n"
            "Do not invent facts.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "flashcards":
        user_prompt = (
            "Generate 8–15 flashcards as Markdown:\n"
            "## Flashcards\n**Q:** ...\n**A:** ...\n\n"
            "Focus on definitions, constraints, and key numbers. Do not invent.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "exec_summary":
        length = int(extra.get("length", 100))
        user_prompt = (
            f"Write an executive summary of about {length} words as:\n"
            "## Executive Summary\n...\n\n"
            "Be faithful; do not invent.\n\n"
            f"[Note]\n{note_md}"
        )
    elif kind == "clarity_improve":
        user_prompt = (
            "Improve clarity and structure while preserving meaning. Output:\n"
            "1) Rewritten note in clean Markdown\n"
            "2) End with:\n## Clarity Improvements\n- What changed and why\n\n"
            "Do not invent facts.\n\n"
            f"[Note]\n{note_md}"
        )
    else:
        return note_md

    out = llm_call(model, system_prompt, user_prompt, max_tokens, temperature)
    st.session_state.metrics["magics_applied"] += 1
    return out



# ----------------------------
# 10) Sidebar WOW Control Center (KEY CHANGE)
# ----------------------------

def sidebar_control_center():
    with st.sidebar:
        st.markdown(f"### {t('control_center')}")
        st.caption("Theme • Language • Painter Style • API Keys")

        # Theme toggle
        theme_choice = st.radio(
            t("theme"),
            options=["dark", "light"],
            horizontal=True,
            format_func=lambda x: t("dark") if x == "dark" else t("light"),
            index=0 if st.session_state.ui_theme == "dark" else 1,
        )
        st.session_state.ui_theme = theme_choice

        # Language
        lang_choice = st.selectbox(
            t("language"),
            options=["en", "zh-TW"],
            index=0 if st.session_state.ui_lang == "en" else 1,
            format_func=lambda x: t("english") if x == "en" else t("zh_tw"),
        )
        st.session_state.ui_lang = lang_choice

        # Painter Style + Game
        st.markdown("---")
        st.markdown(f"**{t('style')}**")

        style_ids = [s["id"] for s in PAINTER_STYLES]
        style_names = [s["name"] for s in PAINTER_STYLES]
        current_idx = style_ids.index(st.session_state.ui_style) if st.session_state.ui_style in style_ids else 0

        picked_name = st.selectbox(" ", options=style_names, index=current_idx, label_visibility="collapsed")
        st.session_state.ui_style = style_ids[style_names.index(picked_name)]

        st.markdown(f"**{t('style_arcade')}**")
        arcade = st.empty()

        def spin_style(mode: str):
            # "Spin" cycles quickly; "Jackpot" cycles longer and ends with a dramatic final pick
            cycles = 14 if mode == "spin" else 30
            delay = 0.06 if mode == "spin" else 0.045
            start = int(time.time() * 1000) % 10_000
            idx = start % len(PAINTER_STYLES)

            for i in range(cycles):
                idx = (idx + 1) % len(PAINTER_STYLES)
                arcade.markdown(
                    f"<div class='wow-card'><b>{'🎰' if mode=='jackpot' else '🌀'}</b> "
                    f"<span class='wow-muted'>Selecting…</span><br><b>{PAINTER_STYLES[idx]['name']}</b></div>",
                    unsafe_allow_html=True
                )
                time.sleep(delay)

            final = (idx + (start % 7)) % len(PAINTER_STYLES)
            st.session_state.ui_style = PAINTER_STYLES[final]["id"]
            arcade.markdown(
                f"<div class='wow-card'><span class='wow-dot ok'></span> "
                f"<b>Selected:</b> {PAINTER_STYLES[final]['name']}</div>",
                unsafe_allow_html=True
            )
            st.toast(f"{t('style')}: {PAINTER_STYLES[final]['name']}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("spin"), use_container_width=True):
                spin_style("spin")
        with c2:
            if st.button(t("jackpot"), use_container_width=True):
                spin_style("jackpot")

        # API Keys (env-first; show inputs only if env not set)
        st.markdown("---")
        st.markdown(f"### {t('api_keys')}")

        def key_line(provider: str, label: str):
            k, src = get_effective_key(provider)
            if src == "env":
                st.markdown(
                    f"<div class='wow-pill'><span class='wow-dot ok'></span><b>{label}</b>"
                    f"<span class='wow-muted'>{t('connected_env')}</span></div>",
                    unsafe_allow_html=True
                )
                # IMPORTANT: Do not show input, do not show key
                return

            # Missing or session: show input (password)
            st.markdown(
                f"<div class='wow-pill'><span class='wow-dot {'ok' if src=='session' else 'bad'}'></span>"
                f"<b>{label}</b><span class='wow-muted'>{t('connected_session') if src=='session' else t('missing')}</span></div>",
                unsafe_allow_html=True
            )
            st.text_input(t("enter_key"), type="password", key=f"{provider}_key")

        key_line("openai", "OpenAI")
        key_line("gemini", "Gemini")
        key_line("anthropic", "Anthropic")
        key_line("grok", "Grok")

        if st.button(t("clear_session_keys"), use_container_width=True):
            clear_session_keys()
            st.toast("Session keys cleared")


# ----------------------------
# 11) Header / status strip
# ----------------------------

def wow_header():
    m = st.session_state.metrics
    stage = st.session_state.pipeline_stage
    docs = len(st.session_state.pdf_items)
    toc_ready = bool(st.session_state.toc_markdown.strip())

    def provider_badge(provider: str, label: str) -> str:
        _, src = get_effective_key(provider)
        if src in ("env", "session"):
            dot = "ok"
            txt = t("connected_env") if src == "env" else t("connected_session")
        else:
            dot = "bad"
            txt = t("missing")
        return f"<span class='wow-pill'><span class='wow-dot {dot}'></span><b>{label}</b><span class='wow-muted'>{txt}</span></span>"

    st.markdown(
        f"""
        <div class="wow-hero">
          <div style="display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap;">
            <div>
              <div style="font-size:1.2rem; font-weight:800;">{APP_TITLE}</div>
              <div class="wow-muted">PDF → Summaries → Master ToC → Agents / Notes</div>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
              <span class='wow-pill'><span class='wow-dot'></span><b>{t('stage')}:</b><span class='wow-muted'>{stage}</span></span>
              <span class='wow-pill'><b>{t('docs_loaded')}:</b><span class='wow-muted'>{docs}</span></span>
              <span class='wow-pill'><b>{t('toc_ready')}:</b><span class='wow-muted'>{'Yes' if toc_ready else 'No'}</span></span>
            </div>
          </div>
          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;">
            {provider_badge('openai','OpenAI')}
            {provider_badge('gemini','Gemini')}
            {provider_badge('anthropic','Anthropic')}
            {provider_badge('grok','Grok')}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# 12) Pages: Workspace / Agents / Notes / Dashboard
# ----------------------------

def workspace_page():
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    st.markdown("### Step 1 — Load Documents")

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        up_pdfs = st.file_uploader(t("upload_pdfs"), type=["pdf"], accept_multiple_files=True)
        if up_pdfs:
            for f in up_pdfs:
                add_pdf_item_from_upload(f.name, f.read())
            st.toast(f"Added {len(up_pdfs)} PDF(s)")
    with c2:
        up_zip = st.file_uploader(t("upload_zip"), type=["zip"], accept_multiple_files=False)
        if up_zip is not None:
            try:
                found = discover_pdfs_from_zip(up_zip.read())
                for name, b in found:
                    add_pdf_item_from_upload(name, b)
                st.toast(f"ZIP: found {len(found)} PDF(s)")
            except Exception as e:
                log_event("ZIP extraction failed", level="error", data={"error": str(e)})
                st.error(f"ZIP extraction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown("### Step 2 — Process (Trim → Summarize → ToC)")

    path = st.text_input(t("scan_path"), value="")
    if st.button(t("scan")) and path.strip():
        try:
            set_stage("Scanning")
            pdf_paths = discover_pdfs_from_path(path.strip())
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

    st.session_state.trim_first_page = st.toggle(t("trim_first_page"), value=st.session_state.trim_first_page)

    cA, cB, cC = st.columns([1.4, 1.0, 1.0])
    with cA:
        st.selectbox(t("model"), SUPPORTED_MODELS, key="summary_model",
                     index=SUPPORTED_MODELS.index(st.session_state.summary_model) if st.session_state.summary_model in SUPPORTED_MODELS else 0)
    with cB:
        st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256, key="summary_max_tokens")
    with cC:
        st.slider(t("temperature"), 0.0, 1.0, float(st.session_state.summary_temperature), 0.05, key="summary_temperature")

    st.text_area(t("summary_prompt"), key="summary_prompt", height=120)

    run_btn = st.button(t("run_pipeline"), use_container_width=True)
    cancel_btn = st.button(t("cancel"), use_container_width=True)

    if cancel_btn:
        st.session_state.cancel_requested = True
        st.toast("Cancel requested")

    if run_btn:
        st.session_state.cancel_requested = False
        if not st.session_state.pdf_items:
            st.warning("No PDFs loaded yet.")
        else:
            ok = no_text = single_page = errors = 0
            total = len(st.session_state.pdf_items)
            prog = st.progress(0)
            status = st.status("Processing PDFs…", expanded=True)
            try:
                set_stage("Trimming / Extracting")

                for i, item in enumerate(st.session_state.pdf_items, start=1):
                    if st.session_state.cancel_requested:
                        status.update(label="Canceled by user", state="error")
                        break

                    sid, name = item["id"], item["name"]

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
                        md = summarize_document(name, item["text"])
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

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown("### Step 3 — Review & Edit Master ToC")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**{t('toc_editor')}**")
        st.text_area("", key="toc_markdown", height=420)

        cX, cY = st.columns(2)
        with cX:
            if st.button(t("refresh_toc"), use_container_width=True):
                st.session_state.toc_markdown = build_master_toc()
                st.toast("ToC refreshed")
        with cY:
            st.download_button(
                t("download_toc"),
                data=(st.session_state.toc_markdown or "").encode("utf-8"),
                file_name="ToC_Master.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with c2:
        st.markdown(f"**{t('toc_preview')}**")
        st.markdown(st.session_state.toc_markdown or "_(empty)_")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Logs"):
        for item in st.session_state.processing_log[-200:]:
            st.write(f"[{item['ts']}] {item['level'].upper()}: {item['msg']}  {json.dumps(item['data'], ensure_ascii=False)}")


def agent_studio_page():
    toc = st.session_state.toc_markdown.strip() or "(Master ToC is empty. Build it in Workspace first.)"
    agents = st.session_state.agents_catalog
    agent_names = [f"{a['name']}  ·  {a.get('category','')}" for a in agents]
    agent_ids = [a["id"] for a in agents]

    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    st.markdown(f"### {t('single_agent')}")

    if agents:
        idx = agent_ids.index(st.session_state.agent_single_id) if st.session_state.agent_single_id in agent_ids else 0
        picked = st.selectbox(t("select_agent"), agent_names, index=idx)
        st.session_state.agent_single_id = agent_ids[agent_names.index(picked)]
        agent = agents[agent_ids.index(st.session_state.agent_single_id)]
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

    st.text_area("Prompt (editable)", key="agent_single_prompt", height=160)

    if st.button("Run agent on ToC", use_container_width=True):
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
            st.markdown(f"#### {t('output')}")
            st.markdown(out)
        except Exception as e:
            set_stage("Idle")
            log_event("Agent run failed", level="error", data={"error": str(e)})
            st.error(f"Agent run failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown(f"### {t('pipeline_agents')}")

    cA, cB, cC = st.columns(3)
    with cA:
        if st.button(t("add_step"), use_container_width=True):
            st.session_state.agent_steps.append(default_step_dict())
            st.session_state.agent_active_step = len(st.session_state.agent_steps) - 1
    with cB:
        if st.button(t("remove_step"), use_container_width=True) and st.session_state.agent_steps:
            st.session_state.agent_steps.pop()
            st.session_state.agent_active_step = max(0, min(st.session_state.agent_active_step, len(st.session_state.agent_steps) - 1))
    with cC:
        if st.button(t("reset"), use_container_width=True):
            st.session_state.agent_steps = []
            st.session_state.agent_active_step = 0

    if not st.session_state.agent_steps:
        st.info("Add steps to build a pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    step_labels = [
        f"Step {i+1}: {next((a['name'] for a in agents if a['id']==s['agent_id']), 'Agent')}"
        for i, s in enumerate(st.session_state.agent_steps)
    ]
    st.session_state.agent_active_step = st.selectbox(
        "Active step",
        list(range(len(step_labels))),
        format_func=lambda i: step_labels[i],
        index=min(st.session_state.agent_active_step, len(step_labels)-1),
    )
    s = st.session_state.agent_steps[st.session_state.agent_active_step]

    # Step config
    agent_idx = agent_ids.index(s["agent_id"]) if s["agent_id"] in agent_ids else 0
    picked_agent = st.selectbox(t("select_agent"), agent_names, index=agent_idx, key=f"pipe_agent_{s['id']}")
    s["agent_id"] = agent_ids[agent_names.index(picked_agent)]

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        s["model"] = st.selectbox(t("model"), SUPPORTED_MODELS,
                                 index=SUPPORTED_MODELS.index(s["model"]) if s["model"] in SUPPORTED_MODELS else 0,
                                 key=f"pipe_model_{s['id']}")
    with c2:
        s["max_tokens"] = st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256,
                                          value=int(s["max_tokens"]), key=f"pipe_max_{s['id']}")
    with c3:
        s["temperature"] = st.slider(t("temperature"), 0.0, 1.0, float(s["temperature"]), 0.05, key=f"pipe_temp_{s['id']}")

    s["prompt"] = st.text_area("Prompt (editable)", value=s.get("prompt",""), height=140, key=f"pipe_prompt_{s['id']}")

    s["input_mode"] = st.radio(
        t("input_source"),
        options=["prev", "toc", "custom"],
        index=["prev", "toc", "custom"].index(s.get("input_mode", "prev")),
        format_func=lambda x: {"prev": t("from_prev"), "toc": t("from_toc"), "custom": t("custom")}.get(x, x),
        horizontal=True,
        key=f"pipe_inmode_{s['id']}",
    )
    if s["input_mode"] == "custom":
        s["custom_input"] = st.text_area("Custom input", value=s.get("custom_input",""), height=120, key=f"pipe_custom_{s['id']}")

    def step_input(i: int) -> str:
        step = st.session_state.agent_steps[i]
        if step["input_mode"] == "toc":
            return toc
        if step["input_mode"] == "custom":
            return step.get("custom_input", "")
        if i == 0:
            return toc
        prev = st.session_state.agent_steps[i-1]
        return (prev.get("output_text") or prev.get("output_md") or toc)

    cR1, cR2 = st.columns(2)
    with cR1:
        if st.button(t("run_step"), use_container_width=True):
            try:
                set_stage("Agent Running")
                i = st.session_state.agent_active_step
                out = run_agent_once(
                    agent_id=s["agent_id"],
                    prompt=s["prompt"],
                    model=s["model"],
                    max_tokens=int(s["max_tokens"]),
                    temperature=float(s["temperature"]),
                    context=step_input(i),
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
    with cR2:
        if st.button(t("run_all"), use_container_width=True):
            try:
                set_stage("Agent Running")
                for i in range(len(st.session_state.agent_steps)):
                    step = st.session_state.agent_steps[i]
                    out = run_agent_once(
                        agent_id=step["agent_id"],
                        prompt=step["prompt"],
                        model=step["model"],
                        max_tokens=int(step["max_tokens"]),
                        temperature=float(step["temperature"]),
                        context=step_input(i),
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

    st.markdown(f"#### {t('output')}")
    view = st.radio("View", ["text", "markdown"], index=1, horizontal=True,
                    format_func=lambda x: "Text" if x == "text" else "Markdown",
                    key=f"pipe_view_{s['id']}")
    if view == "text":
        s["output_text"] = st.text_area("", value=s.get("output_text",""), height=240, key=f"pipe_outtext_{s['id']}")
    else:
        s["output_md"] = st.text_area("Markdown (editable)", value=s.get("output_md",""), height=240, key=f"pipe_outmd_{s['id']}")
        st.markdown("Preview:")
        st.markdown(s["output_md"] or "_(empty)_")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Agent run history"):
        st.json(st.session_state.agent_run_history[-50:])


def note_keeper_page():
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    st.markdown("### AI Note Keeper — Transform & Enhance")

    st.text_area(t("note_input"), key="note_raw_input", height=220)

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        st.selectbox(t("model"), SUPPORTED_MODELS, key="note_model",
                     index=SUPPORTED_MODELS.index(st.session_state.note_model) if st.session_state.note_model in SUPPORTED_MODELS else 0)
    with c2:
        st.number_input(t("max_tokens"), min_value=256, max_value=20000, step=256, key="note_max_tokens")
    with c3:
        st.slider(t("temperature"), 0.0, 1.0, float(st.session_state.note_temperature), 0.05, key="note_temperature")

    st.text_area("Transform prompt (editable)", key="note_prompt", height=140)

    if st.button(t("transform"), use_container_width=True):
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
            st.session_state.note_keywords = extract_keywords_from_markdown(md)
            st.session_state.metrics["notes_created"] += 1
            set_stage("Idle")
            st.toast("Note transformed")
        except Exception as e:
            set_stage("Idle")
            log_event("Note transform failed", level="error", data={"error": str(e)})
            st.error(f"Note transform failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    cA, cB = st.columns([1, 1])

    with cA:
        st.markdown(f"**{t('raw_markdown')}**")
        st.text_area("", key="note_markdown", height=360)
        st.color_picker(t("keyword_color"), key="note_keyword_color")

        kw_str = ", ".join(st.session_state.note_keywords)
        st.text_input(t("keywords"), value=kw_str, key="note_keywords_input")
        st.session_state.note_keywords = [k.strip() for k in re.split(r"[,;\n\u3001]", st.session_state.note_keywords_input) if k.strip()][:50]

        st.download_button(
            "Download note.md",
            data=(st.session_state.note_markdown or "").encode("utf-8"),
            file_name="note.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with cB:
        st.markdown(f"**{t('highlight_preview')}**")
        highlighted = highlight_keywords_html(
            st.session_state.note_markdown or "",
            st.session_state.note_keywords or [],
            st.session_state.note_keyword_color or DEFAULT_KEYWORD_COLOR,
        )
        st.markdown(highlighted or "_(empty)_", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.markdown(f"### {t('ai_magics')}")

    magic = st.selectbox(
        "Magic",
        options=[
            ("ai_keywords", "AI Keywords (insert/update Keywords section)"),
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
    elif magic == "exec_summary":
        extra["length"] = st.selectbox("Length (words)", [50, 100, 200], index=1)

    if st.button(t("apply_magic"), use_container_width=True):
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

            st.session_state.note_keywords = extract_keywords_from_markdown(st.session_state.note_markdown) or st.session_state.note_keywords
            set_stage("Idle")
            st.toast("Magic applied")
        except Exception as e:
            set_stage("Idle")
            log_event("Magic failed", level="error", data={"error": str(e)})
            st.error(f"Magic failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def dashboard_page():
    m = st.session_state.metrics
    st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDFs Found", m["pdf_found"])
    c2.metric("Summarized OK", m["pdf_summarized_ok"])
    c3.metric("No Text (scanned)", m["pdf_no_text"])
    c4.metric("Errors", m["pdf_errors"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM Runs", m["llm_runs"])
    c2.metric("Agents Executed", m["agents_executed"])
    c3.metric("Notes Created", m["notes_created"])
    c4.metric("Magics Applied", m["magics_applied"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-card' style='margin-top:12px;'>", unsafe_allow_html=True)
    st.write("Last model:", m.get("last_model") or "-")
    st.write("Last run at:", m.get("last_run_at") or "-")
    st.write("Current stage:", st.session_state.pipeline_stage)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Recent logs"):
        for item in st.session_state.processing_log[-80:][::-1]:
            st.write(f"[{item['ts']}] {item['level'].upper()}: {item['msg']}  {json.dumps(item['data'], ensure_ascii=False)}")


# ----------------------------
# 13) Main
# ----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ss_init()

    # Sidebar controls first (so theme/lang/style affects CSS immediately)
    sidebar_control_center()
    inject_wow_css()

    # Load agents once per session
    if not st.session_state.agents_catalog:
        st.session_state.agents_catalog = load_agents_yaml()
        st.session_state.agent_single_id = st.session_state.agents_catalog[0]["id"]
        st.session_state.agent_single_prompt = st.session_state.agents_catalog[0].get("user_prompt_template", "")

    wow_header()

    tabs = st.tabs([t("workspace"), t("agents"), t("notes"), t("dashboard")])

    with tabs[0]:
        workspace_page()
    with tabs[1]:
        agent_studio_page()
    with tabs[2]:
        note_keeper_page()
    with tabs[3]:
        dashboard_page()


if __name__ == "__main__":
    main()
