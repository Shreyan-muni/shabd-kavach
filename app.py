import io
import os
import re
import json
import html
import base64
from PIL import Image
import gradio as gr
from groq import Groq

# =====================================================================
# CONFIG (API key comes from the GROQ_API_KEY environment variable / Space secret)
# =====================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it as a secret or environment variable on your host.")
client = Groq(api_key=GROQ_API_KEY)

MODEL_CANDIDATES = ["qwen/qwen3.8-27b"]
_state = {"model": None}

# Your logo, embedded in the code (no file or upload needed).
LOGO_B64 = (
    "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"
    "EYdknwGFDdvvGzGkY+5poH3iYcnVLqb2gn4F2G4+AV10lp3E6kFG6VKm1xy8PPhji/jjgcn0XrW1OnNztHWqJ8MLKx454L24y8ke"
    "0e5WuqaoEsOpqyXHSYtlafiCCpT0/uJdda6Pulnppm27U0VK6anmp2jhm4efIOzg9COf1sj0w3cDU2399ssTtP2iWjubZW/pgpWw"
    "tcznxB3CefTHJePthNJBuvZRE4jjmcxx8wWOyFydpf3FnfRtVcdZCeE2m3+rhw7mufA6+90+2vbCV27bqqkMtJpL9PHj3p8uJ8rd"
    "1dwGTMxqSoeGuBLXRx+9jng+78laS13GC62SkuVKQYqmFszSPJwyqr7lWAab3HuFFBGWU8z/AGqDy4H88D4O4h8lMOyeoIanbKSg"
    "qJhx2qRzXE9InZe0/Ae8Pkrjoxf3FG/q2V3Nt8ebb4x7s964+4q+l2nW1fTaOoWdNRXDOElwku3Hc8L3ng7tbh3+za3ZaNPXSSkZ"
    "TwNM4Y1p4pHHIByD4Nx9q7NB6y1dJpK/awu9TV3eloIxHFSe4wOfyL3ZDfBoIz48iVD9+uk171RcbxLniq6h8wHk0n3R8m4CtBom"
    "wU2nNrqK2V7I2AwGWs7zAbxPHE/iz0AOPktej3N1qmpVq0arUI5a4vCzwjw5ePuM9ctbTSNKoUZUoupLCfBZeOMuPPw9jINvO72t"
    "rpM90NzbbYSfdio2BuP3RyT9q8yi3F11TTiduqK+XB+rK8SNPycCpAobnslpGofFBG+9ThxJndT+0BvPwaXYbgen2rw9fat281Jp"
    "4x2G0T0l0je0xTCkbC0tz7zXFp5jHhy8Qqu6p3MISrVL9OouO1SfwTXDPhgtLSpbTnGhT05qm+G5wXxaazjxzkz/AG13SdqytdZL"
    "xDHDcmsL45Yhhk7R48ujh448CM+SyLXWu7fom0skmZ7RXT5FPSh2OLHi5x6NHn8gq87ezupdz7E+MkOdWMZkeTuRH2Er7NzbpNdt"
    "0LrJK8llNJ7LE0n6rWcv5XcR+asaHSm4hpLnJ5qbtqfuzl+K+2SruOiVrPWFTgsUtu5rxzjC7k+fxxg7bpulri61JkF5koY88oaJ"
    "ojaPnzJ+ZXZad19bWyra+S6GvhafehrGh4cPxhhw+1ZXtRt1ZL5p06hv1Oaxskro6enc4hgDTguODzJOfTksov2y2mrlUU8lp47Q"
    "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"
)

# Telemetry counters (global: shared by all users of this app, reset on restart)
stats = {"total_calls": 0, "profane_detected": 0}


# =====================================================================
# LOGO
# =====================================================================
def get_hero_html():
    # The logo is inlined as a data URI, so no file or path is needed.
    return HERO_TEMPLATE.replace("__LOGO__", LOGO_B64)


# =====================================================================
# PROMPT + UI HTML
# =====================================================================
SYSTEM_PROMPT = """You are an advanced bilingual (English/Hindi/Hinglish) content moderation, NLP evaluation, and de-escalation engine.
Your task is to detect vulgarities, abusive slurs (including phonetic Hinglish variants like laude, chutiya, bc, mc, gand, harami, bkl, etc.), threats, targeted harassment, and toxic tone.

Follow this exact internal multi-task procedure:
1. Dialect & Token Analysis: Identify language ("English", "Hindi", or "Code-Mixed Hinglish") and list violating tokens in 'flagged_words'.
2. Intent Disentanglement: Extract the functional core message or business grievance into 'functional_intent', completely stripped of abusive emotion.
3. Multi-Vector Safety Scoring: Score continuous risks (0 to 100) for Harassment, Hate/Bias, Vulgarity, and Physical Threat.
4. Censorship: Mask the text using the requested style (Asterisks, Partial, or Redact).
5. Diplomatic Rewrite: Restructure into a polite, professional, workplace-ready version in 'diplomatic_rewrite' preserving the functional intent.
6. Self-Critique & Metrics:
   - Estimate 'semantic_preservation' (0-100%): How much functional intent was retained?
   - Estimate 'toxicity_reduction' (0-100%): Net percentage of toxicity eliminated.

You MUST return strictly a JSON object matching this schema:
{
  "flagged_words": ["word1"],
  "censored_text": "censored string",
  "severity_score": 0-100,
  "severity_label": "Clean | Low | Moderate | Severe | Critical",
  "meter_emoji": "emoji",
  "spectrum_emojis": ["e1", "e2", "e3", "e4", "e5"],
  "dialect": "English | Hindi | Code-Mixed Hinglish",
  "functional_intent": "Core grievance",
  "diplomatic_rewrite": "Polite version",
  "vectors": {"Harassment": 0, "Hate / Bias": 0, "Vulgarity": 0, "Threat": 0},
  "sentiment": {"label": "Neutral", "tone": "Professional", "confidence": 100},
  "metrics": {"semantic_preservation": 100, "toxicity_reduction": 100},
  "audit_reasoning": "explanation"
}
"""

CUSTOM_CSS = """
:root { --sk-navy: #0b2a5b; --sk-blue: #1d5fd1; --sk-saffron: #f97316; --sk-green: #16a34a; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- hero ---------- */
.sk-hero { position: relative; overflow: hidden; border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(135deg, #0b2a5b 0%, #12306b 55%, #0a1a3a 100%);
  box-shadow: 0 12px 32px rgba(11, 42, 91, 0.28); }
.sk-stripe { height: 5px; background: linear-gradient(90deg, #f97316 0 33.33%, #ffffff 33.33% 66.66%, #16a34a 66.66% 100%); }
.sk-hero-inner { display: flex; align-items: center; gap: 26px; padding: 26px 32px; flex-wrap: wrap; }
.sk-logo { width: 112px; height: 112px; flex: 0 0 auto; border-radius: 18px; background: #ffffff; padding: 6px;
  display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35); }
.sk-logo img { width: 100%; height: 100%; object-fit: contain; border-radius: 12px; }
.sk-hero-text { flex: 1 1 360px; min-width: 0; }
.sk-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.sk-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.18); background: rgba(255, 255, 255, 0.08);
  font-size: 11px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase; color: #dbe6fb; }
.sk-chip i { width: 7px; height: 7px; border-radius: 50%; background: #34d399; display: inline-block; }
.sk-hero h1 { margin: 12px 0 6px 0 !important; font-size: 34px !important; font-weight: 800 !important;
  letter-spacing: -0.5px; line-height: 1.1 !important; color: #ffffff !important; }
.sk-hero p { margin: 0 !important; max-width: 680px; font-size: 14px; line-height: 1.6; color: #c7d4ec !important; }

/* ---------- section labels + cards ---------- */
.sk-section { font-size: 11px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;
  color: var(--body-text-color-subdued); margin: 2px 0 4px 2px; }
.sk-card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }
.sk-card-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.sk-card-title { font-size: 11px; font-weight: 800; letter-spacing: 0.9px; text-transform: uppercase; color: var(--body-text-color-subdued); }
.sk-pill { padding: 3px 12px; border-radius: 999px; border: 1px solid; font-size: 12px; font-weight: 700; color: var(--body-text-color); }
.sk-spectrum { display: flex; justify-content: space-between; font-size: 20px; margin-bottom: 10px; }
.sk-track { position: relative; height: 10px; border-radius: 999px; background: linear-gradient(90deg, #10b981, #facc15, #fb923c, #ef4444); }
.sk-thumb { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; background: #ffffff;
  border: 3px solid; transform: translate(-50%, -50%); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3); }
.sk-foot { margin-top: 14px; font-size: 13px; color: var(--body-text-color-subdued); }
.sk-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }
.sk-tile { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 12px; background: var(--background-fill-secondary); }
.sk-tile b { display: block; font-size: 22px; font-weight: 800; color: var(--body-text-color); }
.sk-tile span { font-size: 11px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; color: var(--body-text-color-subdued); }
.sk-bar { height: 8px; border-radius: 999px; background: #10b981; overflow: hidden; }
.sk-bar > div { height: 100%; background: #ef4444; }
.sk-disclaimer { text-align: center; font-size: 12px; color: var(--body-text-color-subdued); margin: 14px 0 4px 0; }

/* ---------- controls ---------- */
#analyze-btn { background: linear-gradient(135deg, #0b2a5b, #1d5fd1) !important; border: none !important;
  color: #ffffff !important; font-weight: 700 !important; box-shadow: 0 6px 16px rgba(29, 95, 209, 0.35) !important; }
#analyze-btn:hover { filter: brightness(1.1); }
"""

HERO_TEMPLATE = """
<div class="sk-hero">
  <div class="sk-stripe"></div>
  <div class="sk-hero-inner">
    <div class="sk-logo"><img src="data:image/jpeg;base64,__LOGO__" alt="Shabd Kavach logo"></div>
    <div class="sk-hero-text">
      <div class="sk-chips">
        <span class="sk-chip"><i></i>NLP · Toxicity Analysis</span>
        <span class="sk-chip">English · Hindi · Hinglish</span>
        <span class="sk-chip">Made in India</span>
      </div>
      <h1>Shabd Kavach</h1>
      <p>Analyze text, estimate multi-vector toxicity, and generate context-aware diplomatic rewrites powered by Qwen.</p>
    </div>
  </div>
</div>
"""


# =====================================================================
# HELPERS
# =====================================================================
def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_model_json(raw):
    """Strip <think> blocks / markdown fences and parse the JSON object."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def call_model(messages):
    """Try the remembered model first, then the others; retry without JSON mode if needed."""
    order = ([_state["model"]] if _state["model"] else []) + [m for m in MODEL_CANDIDATES if m != _state["model"]]
    last_err = None
    for model in order:
        for use_json in (True, False):
            try:
                kwargs = dict(model=model, messages=messages, temperature=0.0)
                if use_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                _state["model"] = model
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                print(f"[{model}, json_mode={use_json}] failed: {e!r}")
    raise last_err


def mask_word(word, style):
    if style == "Redact":
        return "[REDACTED]"
    if style == "Asterisks" or len(word) <= 2:
        return "*" * len(word)
    return word[0] + "*" * (len(word) - 1)  # Partial


def censor_text(text, flagged, style):
    """Mask flagged words in code (more reliable than asking the LLM to do it)."""
    words = sorted({w.strip() for w in flagged if isinstance(w, str) and w.strip()}, key=len, reverse=True)
    for w in words:
        pattern = r"(?<![\w\u0900-\u097F])" + re.escape(w) + r"(?![\w\u0900-\u097F])"
        text = re.sub(pattern, lambda m: mask_word(m.group(0), style), text, flags=re.I)
    return text


# =====================================================================
# RENDERERS
# =====================================================================
def render_dynamic_emoji_meter(score, label, active_emoji="🛡️", spectrum=None):
    clamped_score = max(0, min(100, to_int(score)))
    label = html.escape(str(label))
    active_emoji = html.escape(str(active_emoji))
    if not spectrum or not isinstance(spectrum, list) or len(spectrum) != 5:
        spectrum = ["🌿", "🕊️", "⚠️", "🔥", "💥"]
    glow_color = "#10b981" if clamped_score < 25 else "#facc15" if clamped_score < 50 else "#fb923c" if clamped_score < 75 else "#ef4444"
    spans = "".join("<span>" + html.escape(str(x)) + "</span>" for x in spectrum)
    return f"""
    <div class="sk-card">
      <div class="sk-card-head">
        <span class="sk-card-title">Toxicity gauge</span>
        <span class="sk-pill" style="border-color: {glow_color};">{clamped_score}/100 · {label}</span>
      </div>
      <div class="sk-spectrum">{spans}</div>
      <div class="sk-track"><div class="sk-thumb" style="left: {clamped_score}%; border-color: {glow_color};"></div></div>
      <div class="sk-foot">Emotion / tone: <b style="color: {glow_color};">{active_emoji} {label}</b></div>
    </div>
    """


def render_chart_html(total, detected):
    clean = max(0, total - detected)
    rate = round((detected / max(1, total)) * 100, 1)
    return f"""
    <div class="sk-card">
      <div class="sk-card-head">
        <span class="sk-card-title">API traffic</span>
        <span class="sk-pill" style="border-color: #94a3b8;">Detection rate {rate}%</span>
      </div>
      <div class="sk-tiles">
        <div class="sk-tile"><b>{total}</b><span>Requests</span></div>
        <div class="sk-tile"><b style="color: #ef4444;">{detected}</b><span>Flagged</span></div>
        <div class="sk-tile"><b style="color: #10b981;">{clean}</b><span>Clean</span></div>
      </div>
      <div class="sk-bar"><div style="width: {rate}%;"></div></div>
    </div>
    """


# =====================================================================
# CORE LOGIC
# =====================================================================
def moderate_and_analyze(text, mask_style):
    if not text or not text.strip():
        return "", render_dynamic_emoji_meter(0, "Clean"), "Neutral", "", "", "English", "", "", {}, "", "", render_chart_html(stats["total_calls"], stats["profane_detected"])

    stats["total_calls"] += 1
    try:
        raw = call_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Style: {mask_style}\nInput: {text}"},
        ])
        data = parse_model_json(raw)

        flagged_list = [w for w in (data.get("flagged_words") or []) if isinstance(w, str)]
        censored = censor_text(text, flagged_list, mask_style)
        severity = max(0, min(100, to_int(data.get("severity_score"))))
        label = data.get("severity_label", "Clean")
        dialect = data.get("dialect", "English")

        if flagged_list or severity >= 40:
            stats["profane_detected"] += 1

        vectors_raw = data.get("vectors") or {}
        vectors_norm = {str(k): max(0.0, min(1.0, to_float(v) / 100.0)) for k, v in vectors_raw.items()}

        m = data.get("metrics") or {}
        metrics = (f"Preservation: {to_int(m.get('semantic_preservation'), 100)}% | "
                   f"Toxicity reduction: {to_int(m.get('toxicity_reduction'), 100)}%")

        sent = data.get("sentiment") or {}

        return (
            censored,
            render_dynamic_emoji_meter(severity, label, data.get("meter_emoji", "🛡️"), data.get("spectrum_emojis")),
            sent.get("label", "Neutral"),
            f"{sent.get('tone', '')} ({to_int(sent.get('confidence'))}%)",
            ", ".join(flagged_list) or "None",
            dialect,
            data.get("functional_intent", ""),
            data.get("diplomatic_rewrite", ""),
            vectors_norm,
            metrics,
            data.get("audit_reasoning", ""),
            render_chart_html(stats["total_calls"], stats["profane_detected"]),
        )
    except Exception as e:
        print("moderate_and_analyze error:", repr(e))  # real error goes to the console only
        return ("Something went wrong while analyzing this text. Check the console output for details.",
                render_dynamic_emoji_meter(0, "Error"), "Error", "", "", "", "", "", {}, "", "",
                render_chart_html(stats["total_calls"], stats["profane_detected"]))


# =====================================================================
# GRADIO UI
# =====================================================================
def build_theme():
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    try:
        return base.set(
            body_background_fill="#f4f6fb",
            body_background_fill_dark="#0b1220",
            block_background_fill_dark="#111a2e",
            block_border_color_dark="#22304d",
            block_shadow="0 1px 2px rgba(15, 23, 42, 0.06)",
            block_title_text_weight="600",
            block_label_text_weight="600",
            input_background_fill_dark="#0d1627",
            button_primary_background_fill="*primary_700",
            button_primary_background_fill_hover="*primary_800",
            button_primary_text_color="white",
            button_large_text_weight="600",
        )
    except Exception as e:
        print("Theme refinement skipped:", e)
        return base


def clear_all():
    return ("",) + moderate_and_analyze("", "Partial")


theme = build_theme()
blocks_kwargs = {"title": "Shabd Kavach Pro"}
launch_kwargs = {}
try:
    _gr_major = int(gr.__version__.split(".")[0])
except Exception:
    _gr_major = 5
if _gr_major >= 6:   # Gradio 6 moved theme/css from Blocks() to launch()
    launch_kwargs.update(theme=theme, css=CUSTOM_CSS)
else:
    blocks_kwargs.update(theme=theme, css=CUSTOM_CSS)

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(get_hero_html())

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=340):
            gr.HTML('<div class="sk-section">Input</div>')
            inp = gr.Textbox(label="Message", lines=6, placeholder="Type or paste English / Hindi / Hinglish text here...")
            style = gr.Radio(["Partial", "Asterisks", "Redact"], value="Partial", label="Censorship style")
            with gr.Row():
                btn = gr.Button("Scan & De-Escalate", variant="primary", elem_id="analyze-btn")
                clear_btn = gr.Button("Clear", variant="secondary")
            gr.Examples(
                examples=[
                    ["This is a really slow and useless process. Fix it today."],
                    ["Tumhari service bilkul bekaar hai, kab tak wait karwaoge?"],
                    ["I will make sure you regret this if my refund is not processed."],
                ],
                inputs=[inp],
                label="Try an example",
            )
            gr.HTML('<div class="sk-section" style="margin-top: 14px;">Usage (all visitors)</div>')
            chart_box = gr.HTML(value=render_chart_html(0, 0))

        with gr.Column(scale=7, min_width=420):
            gr.HTML('<div class="sk-section">Results</div>')
            with gr.Tabs():
                with gr.Tab("Overview"):
                    out = gr.Textbox(label="Sanitized output", lines=3, interactive=False)
                    rewrite_box = gr.Textbox(label="Diplomatic rewrite (workplace-ready)", lines=3, interactive=False)
                    meter_box = gr.HTML(value=render_dynamic_emoji_meter(0, "Clean"))
                with gr.Tab("Detailed analysis"):
                    intent_box = gr.Textbox(label="Disentangled functional intent", interactive=False)
                    with gr.Row():
                        dialect_box = gr.Textbox(label="Dialect / code-mixing", interactive=False)
                        flagged_box = gr.Textbox(label="Flagged violations", interactive=False)
                    with gr.Row():
                        sent_box = gr.Textbox(label="Sentiment", interactive=False)
                        tone_box = gr.Textbox(label="Tone & confidence", interactive=False)
                    vector_meter = gr.Label(label="Multi-vector threat profile")
                    metrics_box = gr.Textbox(label="Model reflection metrics", interactive=False)
                    reason_box = gr.Textbox(label="Model audit reasoning", lines=3, interactive=False)

    gr.HTML('<div class="sk-disclaimer">Shabd Kavach · Made in India · Automated moderation can make mistakes, so review flagged content before acting on it.</div>')

    all_outputs = [out, meter_box, sent_box, tone_box, flagged_box, dialect_box, intent_box, rewrite_box, vector_meter, metrics_box, reason_box, chart_box]
    btn.click(moderate_and_analyze, inputs=[inp, style], outputs=all_outputs)
    clear_btn.click(clear_all, inputs=None, outputs=[inp] + all_outputs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)  # Render sets PORT
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=port, **launch_kwargs)
