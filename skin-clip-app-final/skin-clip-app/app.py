"""DermaCLIP Review - fine-tuned CLIP skin-lesion classifier with a physician-in-the-loop feedback workflow."""
from __future__ import annotations

import hmac
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from core import SkinCLIP
from feedback_store import CERTAINTY, LABEL_BASIS, FeedbackStore
from i18n import CLASS_TH, tr

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "200"))
LOW_CONF = 0.55
SKIN_GATE_MIN = float(os.environ.get("SKIN_GATE_MIN", "0.5"))   # below this: "not a skin-lesion image"
SITES = ["", "Head/neck", "Trunk", "Upper limb", "Lower limb", "Palms/soles", "Genital", "Other/unknown"]
EN_NAMES = {"nv": "Melanocytic nevus", "mel": "Melanoma", "bkl": "Benign keratosis", "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratosis", "vasc": "Vascular lesion", "df": "Dermatofibroma"}
RISK = {"mel": "Malignant", "bcc": "Malignant", "akiec": "Pre-malignant", "nv": "Benign", "bkl": "Benign",
        "vasc": "Benign", "df": "Benign"}
TIER_COLOR = {"Malignant": "#B42318", "Pre-malignant": "#B54708", "Benign": "#067647"}

THEMES = {
    "light": dict(bg="#F6F9F9", surface="#FFFFFF", text="#10222B", muted="#5B6F78", border="#E3ECEC",
                  note_bg="#F1F5F9", warn_bg="#FFFAEB", warn_bd="#FEDF89", warn_tx="#93370D",
                  bar="#B8CCCC", top="#0F766E", amber="#D97706", scale=[[0, "#EAF4F3"], [1, "#0F766E"]]),
    "dark": dict(bg="#0B1417", surface="#121F24", text="#E6F0F0", muted="#9DB2B8", border="#26393F",
                 note_bg="#16262C", warn_bg="#3A2A0A", warn_bd="#7A5A12", warn_tx="#FDE68A",
                 bar="#3B5560", top="#2DD4BF", amber="#FBBF24", scale=[[0, "#16262C"], [1, "#2DD4BF"]]),
}

st.set_page_config(page_title="DermaCLIP Review", page_icon="🩺", layout="wide")


# ----------------------------------------------------------------------------- language / theme state
def _qp(name: str, allowed: tuple, default: str) -> str:
    v = st.query_params.get(name, default)
    return v if v in allowed else default


if "lang" not in st.session_state:
    st.session_state["lang"] = _qp("lang", ("en", "th"), "en")
if "theme" not in st.session_state:
    st.session_state["theme"] = _qp("theme", ("light", "dark"), "light")


def L(key: str, **kw) -> str:
    return tr(st.session_state["lang"], key, **kw)


def cname(code: str) -> str:
    return CLASS_TH[code] if st.session_state["lang"] == "th" else EN_NAMES[code]


def P() -> dict:
    return THEMES[st.session_state["theme"]]


def inject_css() -> str:
    c = P()
    return f"""<style>
:root {{--bg:{c['bg']}; --surface:{c['surface']}; --text:{c['text']}; --muted:{c['muted']}; --border:{c['border']};}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{background: var(--bg) !important;}}
[data-testid="stSidebar"] {{background: var(--surface) !important;}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {{color: var(--text);}}
[data-testid="stCaptionContainer"] p {{color: var(--muted) !important;}}
.block-container {{padding-top: 4.2rem; max-width: 1180px;}}
h1, h2, h3 {{letter-spacing: -0.02em;}}
.hero {{background: linear-gradient(120deg,#0F766E 0%,#115E59 55%,#134E4A 100%); padding: 26px 30px; border-radius: 18px;
       margin-bottom: 18px;}}
.hero h1 {{color:#fff !important; margin:0; font-size: 1.9rem;}}
.hero p {{margin:6px 0 0; color:#CCFBF1 !important; font-size: .98rem;}}
.card {{background: var(--surface); border:1px solid var(--border); border-radius:16px; padding:20px 22px;
       box-shadow: 0 1px 2px rgba(16,34,43,.06);}}
.card .muted, .muted {{color: var(--muted) !important; font-size:.88rem;}}
.pred-name {{font-size:1.7rem; font-weight:700; margin:2px 0 6px; color: var(--text);}}
.pill {{display:inline-block; padding:3px 12px; border-radius:999px; font-size:.8rem; font-weight:600; color:#fff !important;}}
.warn {{background:{c['warn_bg']}; border:1px solid {c['warn_bd']}; color:{c['warn_tx']} !important; padding:10px 14px;
       border-radius:12px; font-size:.9rem; margin-top:10px;}}
.disclaimer {{background:{c['note_bg']}; border-left:4px solid #0F766E; padding:10px 14px; border-radius:8px; font-size:.85rem;
             color: var(--text);}}
.adj {{border:2px solid {c['amber']};}}
.old {{text-decoration: line-through; opacity:.7;}}
div[data-testid="stMetric"] {{background: var(--surface); border:1px solid var(--border); padding:14px 16px; border-radius:14px;}}
.stTabs [data-baseweb="tab"] {{font-weight:600;}}
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="textarea"] {{background-color: var(--surface) !important; color: var(--text) !important;
                            border-color: var(--border) !important;}}
[data-testid="stSelectbox"] [data-rac], [data-testid="stSelectbox"] input, [data-testid="stSelectbox"] button
 {{background-color: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;}}
[data-baseweb="popover"] *, [role="listbox"], [role="listbox"] *, [data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] * {{background-color: var(--surface); color: var(--text);}}
[role="option"][data-focused="true"], [role="option"]:hover {{background-color: var(--border) !important;}}
[data-testid="stFileUploaderDropzone"] {{background: var(--surface) !important; border-color: var(--border) !important;}}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderFile"] * {{color: var(--text) !important;}}
.stButton > button[kind="secondary"], .stDownloadButton > button {{background: var(--surface); color: var(--text);
                                                                   border:1px solid var(--border);}}
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * {{color:#fff !important;}}
[data-baseweb="tab-list"] {{border-bottom-color: var(--border);}}
[data-testid="stSegmentedControl"] button {{color: var(--text);}}
table {{color: var(--text); background: var(--surface);}} th, td {{border-color: var(--border) !important;}}
code {{color: inherit;}}
[data-testid="stAlert"], [data-testid="stAlertContainer"] {{background: var(--surface) !important; border-radius:12px;}}
[data-testid="stAlertContainer"] {{border:1px solid var(--border);}}
.stApp button p {{color: inherit;}}   /* button labels keep the button's own colour (e.g. primary) */
[data-testid="stButtonGroup"] button[role="radio"] {{background: var(--surface) !important; border-color: var(--border) !important;}}
[data-testid="stButtonGroup"] button[role="radio"] p {{color: var(--text) !important;}}
[data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] {{background: {c['top']}33 !important;
                                                                          border-color: {c['top']} !important;}}
[data-testid="stAlert"], [data-testid="stAlert"] * {{color: var(--text) !important;}}
[data-testid="stFileUploaderDropzone"] button {{background: var(--surface) !important; color: var(--text) !important;
                                               border:1px solid var(--border) !important;}}
.pbar {{height:10px; border-radius:999px; background: var(--border); overflow:hidden; margin:4px 0 14px;}}
.pbar > div {{height:100%; background:{c['top']}; border-radius:999px;}}
</style>"""


def hero() -> None:
    st.markdown(inject_css() + f'<div class="hero"><h1>DermaCLIP Review</h1><p>{L("hero_sub")}</p></div>',
                unsafe_allow_html=True)


def topbar() -> None:
    _, c1, c2 = st.columns([6, 1.5, 2.2])
    lang = c1.segmented_control(L("lang_label"), ["en", "th"], default=st.session_state["lang"], key="ctl_lang",
                                format_func=lambda v: "EN" if v == "en" else "ไทย", label_visibility="collapsed")
    theme = c2.segmented_control(L("theme_label"), ["light", "dark"], default=st.session_state["theme"], key="ctl_theme",
                                 format_func=lambda v: L("theme_light") if v == "light" else L("theme_dark"),
                                 label_visibility="collapsed")
    changed = False
    if lang and lang != st.session_state["lang"]:
        st.session_state["lang"], changed = lang, True
    if theme and theme != st.session_state["theme"]:
        st.session_state["theme"], changed = theme, True
    if changed:
        st.query_params.update(lang=st.session_state["lang"], theme=st.session_state["theme"])
        st.rerun()


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading model...")
def get_model() -> SkinCLIP:
    return SkinCLIP()


@st.cache_resource
def get_store() -> FeedbackStore:
    return FeedbackStore(DATA_DIR)


def _secret(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            val = str(st.secrets.get(name, ""))
        except Exception:
            val = ""
    return val


def passcode_gate() -> None:
    expected = _secret("REVIEWER_PASSCODE")
    if not expected:
        st.sidebar.warning(L("open_warn"))
        return
    if st.session_state.get("authed"):
        return
    hero()
    st.info(L("login_only"))
    with st.form("login"):
        pw = st.text_input(L("passcode"), type="password")
        if st.form_submit_button(L("sign_in"), type="primary"):
            if hmac.compare_digest(pw.encode(), expected.encode()):
                st.session_state["authed"] = True
                st.rerun()
            st.error(L("wrong_pass"))
    st.stop()


def prob_chart(probs: dict, top: str, physician: str | None) -> go.Figure:
    c = P()
    items = sorted(probs.items(), key=lambda kv: kv[1])

    def color(code):
        return c["amber"] if code == physician else (c["top"] if code == top else c["bar"])
    fig = go.Figure(go.Bar(
        x=[v * 100 for _, v in items], y=[cname(k) for k, _ in items], orientation="h",
        marker_color=[color(k) for k, _ in items], text=[f"{v*100:.1f}%" for _, v in items],
        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=0, r=40, t=10, b=0), xaxis=dict(range=[0, 112], title=L("prob_axis"),
                      gridcolor=c["border"]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13, color=c["text"]))
    return fig


def result_card(pred: dict, physician: str | None, saved: bool = False) -> str:
    """HTML for the result card. If `physician` is set the card shows the physician-adjusted diagnosis."""
    conf = f"{pred['confidence']*100:.1f}"
    if physician and physician != pred["top_code"]:
        tier = RISK[physician]
        meta = L("card_meta_adj", pred=cname(pred["top_code"]), c=conf, p=f"{pred['probs'][physician]*100:.1f}",
                 v=pred["model_version"])
        return (f'<div class="card adj"><span class="muted">{L("card_adj")}</span>'
                f'<div class="pred-name">{cname(physician)}</div>'
                f'<span class="pill" style="background:{TIER_COLOR[tier]}">{L(tier)}</span> '
                f'<span class="muted">&nbsp; {meta}</span></div>')
    tier = pred["risk_tier"]
    meta = L("card_meta", c=conf, m=f"{pred['margin']*100:.1f}", v=pred["model_version"])
    return (f'<div class="card"><span class="muted">{L("card_top")}</span>'
            f'<div class="pred-name">{cname(pred["top_code"])}</div>'
            f'<span class="pill" style="background:{TIER_COLOR.get(tier, "#475467")}">{L(tier)}</span> '
            f'<span class="muted">&nbsp; {meta}</span></div>')


# ----------------------------------------------------------------------------- pages
def page_analyze(model: SkinCLIP, store: FeedbackStore) -> None:
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.subheader(L("step1"))
        up = st.file_uploader(L("uploader"), type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.caption(L("deid_caption"))
        deid = st.checkbox(L("deid_check"))
        if up is not None:
            img = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
            st.image(img, use_container_width=True)
            key = (up.file_id, up.name, up.size)
            if st.session_state.get("case_key") != key:
                for k in ("rv_choice", "rv_label", "case_final"):
                    st.session_state.pop(k, None)
                st.session_state.update(case_key=key, case_img=img, case_pred=None, case_saved=False,
                                        case_score=None, case_override=False)

    with right:
        st.subheader(L("step2"))
        if up is None:
            st.info(L("upload_prompt"))
            return
        if not deid:
            st.warning(L("confirm_prompt"))
            return

        # ---- input gate: is this a skin image at all?
        if st.session_state.get("case_score") is None:
            with st.spinner(L("checking")):
                s = model.skin_score(st.session_state["case_img"])
                st.session_state["case_score"] = -1.0 if s is None else s     # -1 = gate disabled
        score = st.session_state["case_score"]
        if 0 <= score < SKIN_GATE_MIN and not st.session_state.get("case_override"):
            st.markdown(f'<div class="card"><span class="muted">{L("gate_label")}</span>'
                        f'<div class="pred-name">{L("gate_title")}</div>'
                        f'<span class="muted">{L("gate_body", s=f"{score*100:.0f}", m=f"{SKIN_GATE_MIN*100:.0f}")}</span></div>',
                        unsafe_allow_html=True)
            if st.button(L("gate_btn")):
                st.session_state["case_override"] = True
                st.rerun()
            return

        if st.session_state.get("case_pred") is None:
            with st.spinner(L("analysing")):
                st.session_state["case_pred"] = model.predict(st.session_state["case_img"]).to_dict()
        pred = st.session_state["case_pred"]
        saved = st.session_state.get("case_saved")
        final = st.session_state.get("case_final")          # {"decision","label","basis","certainty",...} after save

        # ---- what label should the card show right now?  (live preview of the physician's selection)
        if saved:
            physician = final["label"] if final and final["decision"] == "corrected" else None
        else:
            physician = st.session_state.get("rv_label") if st.session_state.get("rv_choice") == "correct" else None
            if physician not in pred["probs"]:
                physician = None

        st.markdown(result_card(pred, physician), unsafe_allow_html=True)
        if 0 <= score < SKIN_GATE_MIN:
            st.markdown(f'<div class="warn">{L("gate_override")}</div>', unsafe_allow_html=True)
        if not model.is_finetuned:
            st.markdown(f'<div class="warn">{L("demo")}</div>', unsafe_allow_html=True)
        if pred["confidence"] < LOW_CONF and not physician:
            st.markdown(f'<div class="warn">{L("low_conf")}</div>', unsafe_allow_html=True)
        st.plotly_chart(prob_chart(pred["probs"], pred["top_code"], physician), use_container_width=True,
                        config={"displayModeBar": False})
        if physician:
            st.caption(L("chart_legend"))

        st.subheader(L("step3"))
        if saved:
            if final and final["decision"] == "unusable":
                st.success(L("saved_unusable"))
            else:
                st.success(L("saved_ok"))
            b1, b2 = st.columns(2)
            if b1.button(L("change_label"), use_container_width=True):
                # re-open the form pre-filled with the saved values; the newer review supersedes the old one
                st.session_state["rv_choice"] = {"approved": "approve", "corrected": "correct",
                                                 "unusable": "unusable"}[final["decision"]]
                if final["decision"] == "corrected":
                    st.session_state["rv_label"] = final["label"]
                st.session_state["case_saved"] = False
                st.rerun()
            if b2.button(L("another"), use_container_width=True):
                st.session_state.update(case_key=None, case_saved=False)
                st.rerun()
            return

        reviewer = st.text_input(L("reviewer"), value=st.session_state.get("reviewer", ""), placeholder=L("reviewer_ph"))
        choice = st.radio(L("q_correct"), ["approve", "correct", "unusable"], key="rv_choice", horizontal=True,
                          format_func=lambda k: L({"approve": "opt_approve", "correct": "opt_correct",
                                                   "unusable": "opt_unusable"}[k]))
        final_label = basis = certainty = None
        if choice == "correct":
            options = [c for c in model.codes if c != pred["top_code"]]
            final_label = st.selectbox(L("pick_label"), options, key="rv_label", index=None,
                                       placeholder=L("pick_ph"), format_func=lambda c: f"{cname(c)} ({c})")
            st.caption(L("live_hint"))
        if choice != "unusable":
            c1, c2 = st.columns(2)
            basis = c1.selectbox(L("basis"), LABEL_BASIS, index=1, format_func=L)
            certainty = c2.select_slider(L("certainty"), CERTAINTY, value="Medium", format_func=L)
        site = st.selectbox(L("site"), SITES, format_func=lambda s: L(s) if s else L("choose"))
        notes = st.text_area(L("notes"), height=80)

        if st.button(L("submit"), type="primary", use_container_width=True):
            if not reviewer.strip():
                st.error(L("err_reviewer"))
                return
            if choice == "correct" and not final_label:
                st.error(L("err_label"))
                return
            decision = {"approve": "approved", "correct": "corrected", "unusable": "unusable"}[choice]
            try:
                store.add_review(img=st.session_state["case_img"], prediction=pred, reviewer_id=reviewer,
                                 decision=decision, final_label=final_label, label_basis=basis,
                                 reviewer_certainty=certainty, lesion_site=site, notes=notes)
            except ValueError as e:
                st.error(str(e))
                return
            st.session_state.update(case_saved=True, reviewer=reviewer.strip(),
                                    case_final={"decision": decision, "label": final_label or pred["top_code"]})
            st.rerun()


def page_feedback(model: SkinCLIP, store: FeedbackStore) -> None:
    c = P()
    st.subheader(L("an_title"))
    df = store.reviews()
    if df.empty:
        st.info(L("an_none"))
        return
    usable = df[df["decision"] != "unusable"]
    agree = (usable["decision"] == "approved").mean() if len(usable) else float("nan")
    ready = len(store.training_manifest())
    k = st.columns(4)
    k[0].metric(L("m_reviews"), len(df))
    k[1].metric(L("m_agree"), f"{agree*100:.1f}%" if len(usable) else "–")
    k[2].metric(L("m_corr"), int((df["decision"] == "corrected").sum()))
    k[3].metric(L("m_ready"), f"{ready} / {RETRAIN_THRESHOLD}")
    pct = min(ready / RETRAIN_THRESHOLD, 1.0) * 100
    st.markdown(f'<div class="muted">{L("progress")}</div><div class="pbar"><div style="width:{pct:.1f}%"></div></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**{L('ch_agree')}**")
        g = usable.groupby("model_pred")["decision"].agg(n="count", agree=lambda s: (s == "approved").mean())
        g = g.reindex([x for x in model.codes if x in g.index])
        fig = go.Figure(go.Bar(x=g.index, y=g["agree"] * 100, marker_color=c["top"],
                               text=[f"n={n}" for n in g["n"]], textposition="outside"))
        fig.update_layout(height=300, yaxis=dict(range=[0, 115], title=L("ch_agree_y"), gridcolor=c["border"]),
                          margin=dict(t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=c["text"]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown(f"**{L('ch_conf')}**")
        cm = pd.crosstab(usable["model_pred"], usable["final_label"]).reindex(
            index=model.codes, columns=model.codes, fill_value=0)
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale=c["scale"],
                                   text=[["" if v == 0 else str(v) for v in row] for row in cm.values],
                                   texttemplate="%{text}", showscale=False))
        fig.update_layout(height=300, xaxis_title=L("ax_phys"), yaxis_title=L("ax_model"),
                          yaxis=dict(autorange="reversed"), margin=dict(t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color=c["text"]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"**{L('recent')}**")
    show = df.sort_values("id", ascending=False).head(200)[
        ["created_at", "reviewer_id", "model_pred", "model_conf", "decision", "final_label", "label_basis",
         "reviewer_certainty", "model_version"]].copy()
    show["decision"] = show["decision"].map(L)
    show["created_at"] = show["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    show.columns = [L(x) for x in ("col_time", "col_reviewer", "col_model", "col_conf", "col_decision", "col_final",
                                   "col_basis", "col_cert", "col_ver")]
    if st.session_state["theme"] == "dark":      # st.dataframe is canvas-drawn and ignores the CSS theme
        show[L("col_conf")] = show[L("col_conf")].map("{:.2f}".format)
        st.table(show.head(25).reset_index(drop=True))
    else:
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={L("col_conf"): st.column_config.ProgressColumn(L("col_conf"), min_value=0,
                                                                                    max_value=1, format="%.2f")})
    d1, d2 = st.columns(2)
    d1.download_button(L("dl_csv"), df.to_csv(index=False), "reviews.csv", "text/csv")
    d2.download_button(L("dl_zip"), store.export_zip(), "feedback_bundle.zip", "application/zip")
    st.caption(L("retrain_cap"))


def page_model_card(model: SkinCLIP) -> None:
    st.subheader(L("mc_title"))
    meta = model.metadata
    st.write(L("mc_head", v=model.version, b=meta.get("backbone", "ViT-B/32"),
               c=", ".join(f"{cname(c)} ({c})" for c in model.codes)))
    m = meta.get("metrics")
    if m:
        st.markdown(L("mc_metrics"))
        st.dataframe(pd.DataFrame(m).T.style.format("{:.3f}"), use_container_width=True)
        if "beats_baseline" in meta:
            st.caption(L("mc_beats", x=meta["beats_baseline"]))
    else:
        st.info(L("mc_nometrics"))
    st.markdown(L("mc_body"))


def main() -> None:
    topbar()
    passcode_gate()
    model, store = get_model(), get_store()
    hero()
    st.markdown(f'<div class="disclaimer">{L("disclaimer")}</div>', unsafe_allow_html=True)
    st.write("")
    t1, t2, t3 = st.tabs([L("tab_analyze"), L("tab_feedback"), L("tab_card")])
    with t1:
        page_analyze(model, store)
    with t2:
        page_feedback(model, store)
    with t3:
        page_model_card(model)


main()
