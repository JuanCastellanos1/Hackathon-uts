"""Estilo visual Windows Vista / Aero Glass para la demo Streamlit.

Solo cambia la apariencia: no toca datos, lógica ni textos de la app.
Todo el CSS se inyecta en un único bloque <style> desde aplicar_estilo().
"""
from __future__ import annotations

import html
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NOMBRE_PLANTILLA = "vista"
COLORWAY = ["#3C7FB1", "#61BE37", "#2E77C4", "#E07A1F", "#8CC4EE", "#0D3D78", "#D3614C", "#7FB8DE"]

_FUENTE = '"Segoe UI",Tahoma,Geneva,Verdana,sans-serif'
_CRISTAL_OSCURO = "linear-gradient(180deg,rgba(30,50,80,.55),rgba(8,18,34,.72))"

_CSS = r"""
:root{--vista-azul:#3C7FB1;--vista-celeste:#EAF6FD;--vista-borde:#A0A0A0;--vista-texto:#1E3D5C}
html,body,.stApp,.stApp p,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp button,.stApp input,.stApp td,.stApp th,[data-testid="stSidebar"]{font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif !important}
[data-testid="stApp"],.stApp{background:radial-gradient(58% 68% at 12% 88%,rgba(0,168,150,.55),transparent 70%),radial-gradient(46% 60% at 46% 16%,rgba(120,190,255,.34),transparent 65%),radial-gradient(40% 52% at 88% 30%,rgba(90,120,220,.3),transparent 65%),radial-gradient(50% 46% at 68% 86%,rgba(10,80,120,.5),transparent 70%),linear-gradient(160deg,#0E2C46,#123A5C 38%,#0C2C4A 68%,#081C30) !important;background-attachment:fixed !important}
[data-testid="stApp"]::before{content:"";position:fixed;inset:-10%;pointer-events:none;z-index:0;background:linear-gradient(115deg,transparent 28%,rgba(255,255,255,.13) 36%,transparent 44%),linear-gradient(115deg,transparent 58%,rgba(255,255,255,.10) 64%,transparent 72%);filter:blur(16px)}
[data-testid="stHeader"]{background:transparent !important;box-shadow:none !important}
[data-testid="stHeader"] *{color:#EAF4FF !important}
[data-testid="stDecoration"]{display:none !important}
[data-testid="stMain"]{position:relative !important;z-index:1 !important;padding-top:52px !important}
[data-testid="stMainBlockContainer"],.block-container{position:relative !important;background:transparent !important;max-width:1400px !important;width:100% !important;margin:0 auto !important;padding:38px 8px 70px 8px !important;border:none !important}
[data-testid="stMainBlockContainer"]::before,.block-container::before{content:"";position:absolute;inset:0;z-index:-1;border-radius:8px 8px 5px 5px;background:rgba(22,42,72,.45);-webkit-backdrop-filter:blur(16px) saturate(1.5);backdrop-filter:blur(16px) saturate(1.5);border:1px solid rgba(255,255,255,.45);box-shadow:inset 0 0 0 1px rgba(255,255,255,.18),inset 0 0 22px rgba(255,255,255,.09),0 6px 26px rgba(0,10,25,.65);pointer-events:none}
[data-testid="stMainBlockContainer"]>div:first-child,.block-container>div:first-child{background:#FFFFFF !important;border:1px solid #6E6E6E !important;padding:18px 20px 20px 20px !important;min-height:300px;border-radius:0}
[data-testid="stElementContainer"]:has(style),[data-testid="stElementContainer"]:has(#vista-taskbar),[data-testid="stElementContainer"]:has(.vista-tb){position:absolute !important;top:0;left:0;right:0;height:0;overflow:visible;margin:0 !important;padding:0 !important}
[data-testid="stElementContainer"]:has(.vista-tb){top:1px;left:1px;right:1px;z-index:3}
.vista-tb{position:relative;height:30px;display:flex;align-items:center;padding:0 2px 0 6px;font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;font-size:13px;font-weight:400;color:#FFF;text-shadow:0 0 8px rgba(255,255,255,.85),0 0 3px rgba(255,255,255,.9);background:linear-gradient(180deg,rgba(255,255,255,.16),rgba(255,255,255,0));border-radius:7px 7px 0 0;box-sizing:border-box}
.vista-tb-ico{margin-right:6px;font-size:13px;text-shadow:none}
.vista-tb-txt{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vista-tb-btns{display:flex;gap:1px;margin-left:auto;flex:none}
.vb{display:inline-flex;align-items:center;justify-content:center;height:19px;width:29px;box-sizing:border-box;border-radius:3px;border:1px solid rgba(255,255,255,.6);background:linear-gradient(180deg,rgba(255,255,255,.52),rgba(255,255,255,.2) 45%,rgba(255,255,255,.1) 55%,rgba(255,255,255,.22));box-shadow:inset 0 0 0 1px rgba(255,255,255,.25),inset 0 1px 3px rgba(255,255,255,.5);color:#FFF;font-size:11px;line-height:1;text-shadow:0 0 3px rgba(0,20,50,.5)}
.vb-close{width:44px;border-color:rgba(120,30,20,.6);background:linear-gradient(180deg,#EDA796,#D3614C 42%,#B23925 48%,#A8321F)}
.vb:hover{background:radial-gradient(circle at 50% 40%,#FFF,#CDE9FF 45%,#8CC4EE);border-color:#CDE9FF;box-shadow:0 0 6px rgba(150,210,255,.85)}
.vb-close:hover{background:radial-gradient(circle at 50% 40%,#FFD9CE,#FF9C81 40%,#E45B3C 80%,#C9402A);border-color:#FFB8A6;box-shadow:0 0 6px rgba(255,130,90,.9)}
[data-testid="stMainBlockContainer"] h1,[data-testid="stMainBlockContainer"] h2,[data-testid="stMainBlockContainer"] h3,[data-testid="stMainBlockContainer"] h4,.block-container h1,.block-container h2,.block-container h3,.block-container h4{color:#1E3D5C !important;font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif !important}
[data-testid="stMainBlockContainer"] h1,.block-container h1{font-weight:600 !important;font-size:1.9rem}
[data-testid="stMainBlockContainer"] h2,.block-container h2{font-weight:600 !important;font-size:1.5rem;padding-bottom:4px;background:linear-gradient(90deg,#3C7FB1,#A8D4F0 55%,rgba(168,212,240,0)) no-repeat 0 100%/100% 2px}
[data-testid="stMainBlockContainer"] h3,.block-container h3{font-weight:600 !important;font-size:1.2rem;padding-bottom:4px;background:linear-gradient(90deg,#3C7FB1,#A8D4F0 55%,rgba(168,212,240,0)) no-repeat 0 100%/100% 2px}
[data-testid="stMainBlockContainer"] h4,.block-container h4{font-weight:600 !important}
[data-testid="stMainBlockContainer"] hr,.block-container hr{border:none !important;height:1px !important;background:linear-gradient(90deg,#3C7FB1,#B8D4EC 60%,rgba(184,212,236,0)) !important;margin:1.2rem 0 !important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#4A5F75 !important}
[data-baseweb="tab-list"],[data-testid="stTabs"] [role="tablist"]{gap:2px !important;border-bottom:1px solid #A0A0A0 !important;background:transparent !important}
[data-baseweb="tab"],[data-testid="stTab"]{background:linear-gradient(180deg,#FDFDFD,#E9EEF3) !important;border:1px solid #A0A0A0 !important;border-bottom:none !important;border-radius:3px 3px 0 0 !important;padding:6px 14px !important;margin-bottom:0 !important;height:auto !important;color:#1E3D5C !important}
[data-baseweb="tab"] *,[data-testid="stTab"] *{color:#1E3D5C !important}
[data-baseweb="tab"]:hover,[data-testid="stTab"]:hover{background:#EAF6FD !important}
[data-baseweb="tab"][aria-selected="true"],[data-testid="stTab"][aria-selected="true"]{background:#FFFFFF !important;border-top:1px solid #3C7FB1 !important;margin-bottom:-1px !important;position:relative;font-weight:600 !important}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"],[data-testid="stTabs"] [role="tablist"] > div:not([data-testid="stTab"]){display:none !important}
[data-baseweb="tab-panel"]{padding:14px 2px 0 2px !important}
[data-testid="stButtonGroup"]{gap:2px !important;border-bottom:1px solid #A0A0A0 !important;margin-bottom:12px !important}
[data-testid="stButtonGroup"] button{background:linear-gradient(180deg,#FDFDFD,#E9EEF3) !important;border:1px solid #A0A0A0 !important;border-bottom:none !important;border-radius:3px 3px 0 0 !important;color:#1E3D5C !important;box-shadow:none !important;margin-bottom:-1px !important}
[data-testid="stButtonGroup"] button:hover{background:#EAF6FD !important;border-color:#3C7FB1 !important}
[data-testid="stButtonGroup"] button[kind$="Active"],[data-testid="stButtonGroup"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[aria-pressed="true"]{background:#FFFFFF !important;border-top:2px solid #3C7FB1 !important;font-weight:600 !important}
[data-testid="stMetric"]{background:linear-gradient(180deg,#FFFFFF,#F2F7FC) !important;border:1px solid #B8D4EC !important;border-radius:4px !important;padding:10px 12px !important}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#3A5A7A !important}
[data-testid="stMetricValue"]{color:#1B3B5F !important}
.stButton button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{background:linear-gradient(180deg,#F2F2F2,#E4E4E4 45%,#DCDCDC) !important;height:25px !important;min-height:25px !important;border:1px solid #707070 !important;border-radius:2px !important;color:#1E1E1E !important;box-shadow:none !important;padding:0 12px !important}
.stButton button:hover,[data-testid="stBaseButton-secondary"]:hover,[data-testid="stBaseButton-secondaryFormSubmit"]:hover{background:linear-gradient(180deg,#EAF6FD,#D9F0FC 50%,#D2ECFB) !important;border-color:#3C7FB1 !important;box-shadow:0 0 4px rgba(60,127,177,.45) !important}
.stButton button:active,[data-testid="stBaseButton-secondary"]:active{background:linear-gradient(180deg,#C4E5F6,#98D1EF) !important;border-color:#2C628B !important;box-shadow:inset 0 2px 3px rgba(30,80,130,.35) !important}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{background:linear-gradient(180deg,#F5FBFF,#DDEEF9 50%,#D2E9F8) !important;border-color:#3C7FB1 !important;box-shadow:0 0 5px rgba(60,127,177,.35) !important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"]{background:#FFFFFF !important;border:1px solid #A0A0A0 !important;border-radius:2px !important}
[data-baseweb="select"]>div:focus-within,[data-baseweb="input"]:focus-within{border-color:#3C7FB1 !important;box-shadow:0 0 5px rgba(60,127,177,.55) !important}
[data-testid="stSlider"] [role="slider"]{background:#3C7FB1 !important;border:2px solid #FFFFFF !important;box-shadow:0 0 0 1px #2C628B,0 0 4px rgba(60,127,177,.6) !important}
[data-testid="stThumbValue"]{color:#3C7FB1 !important}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #A8A8A8 !important;border-radius:2px !important;overflow:hidden}
[data-testid="stAlert"],[data-testid="stAlertContainer"]{background:#F5FAFE !important;border:1px solid #B8DCF5 !important;border-radius:3px !important;color:#1E3D5C !important}
[data-testid="stAlert"] p,[data-testid="stAlertContainer"] p{color:#1E3D5C !important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(30,50,80,.55),rgba(8,18,34,.72)) !important;-webkit-backdrop-filter:blur(10px) !important;backdrop-filter:blur(10px) !important;border-right:1px solid rgba(255,255,255,.35) !important}
[data-testid="stSidebar"]>div:first-child{background:transparent !important}
[data-testid="stSidebarUserContent"]{padding-bottom:60px !important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] li,[data-testid="stSidebar"] label,[data-testid="stSidebar"] button,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] span:not([class*="material"]){color:#EAF4FF !important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{text-shadow:0 0 8px rgba(255,255,255,.85),0 0 3px rgba(255,255,255,.9)}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:rgba(234,244,255,.85) !important}
[data-testid="stSidebar"] a{color:#9FD3FF !important}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]>div{background:linear-gradient(180deg,rgba(30,50,80,.55),rgba(8,18,34,.72));border:1px solid rgba(255,255,255,.35);border-radius:18px;padding:10px 14px;margin-bottom:10px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.12),inset 0 0 22px rgba(255,255,255,.07),0 6px 18px rgba(0,10,25,.45)}
::-webkit-scrollbar{width:17px;height:17px}
::-webkit-scrollbar-track{background:#F0F0F0}
::-webkit-scrollbar-thumb{background:linear-gradient(90deg,#FBFBFB,#C8C8C8);border:1px solid #9E9E9E}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(90deg,#EAF6FD,#8CC4EE);border-color:#3C7FB1}
#vista-taskbar{position:fixed;left:0;right:0;bottom:0;height:40px;box-sizing:border-box;z-index:999;display:flex;align-items:center;border-top:1px solid rgba(255,255,255,.3);background:linear-gradient(180deg,rgba(32,38,48,.88),rgba(2,4,8,.96));-webkit-backdrop-filter:blur(14px) saturate(1.3);backdrop-filter:blur(14px) saturate(1.3);box-shadow:inset 0 12px 24px -12px rgba(120,180,255,.4);font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;color:#FFF}
.vt-orb{position:relative;width:46px;height:36px;flex:none;margin-left:2px}
.vt-sphere{position:absolute;left:4px;top:1px;width:36px;height:34px;border-radius:50%;background:radial-gradient(circle at 50% 26%,#D8F0FF,#8CC8F2 22%,#2E77C4 52%,#0D3D78 78%,#062348);box-shadow:0 0 12px rgba(140,205,255,.75),0 0 26px rgba(90,160,255,.4)}
.vt-sphere::before{content:"";position:absolute;left:7px;top:3px;width:22px;height:12px;border-radius:50%;background:linear-gradient(180deg,rgba(255,255,255,.95),rgba(255,255,255,.15))}
.vt-globe{position:absolute;left:4px;top:1px;width:36px;height:34px;line-height:34px;text-align:center;font-size:15px}
.vt-task{margin-left:6px;height:28px;padding:0 14px;box-sizing:border-box;display:flex;align-items:center;border-radius:4px;border:1px solid rgba(255,255,255,.28);background:linear-gradient(180deg,rgba(255,255,255,.16),rgba(255,255,255,.02));font-size:12px;text-shadow:0 0 6px rgba(255,255,255,.6);white-space:nowrap}
.vt-activo{background:linear-gradient(180deg,rgba(255,255,255,.26),rgba(120,180,245,.18));border-color:rgba(160,210,255,.6);box-shadow:0 0 8px rgba(120,180,245,.5),inset 0 0 8px rgba(255,255,255,.12)}
.vt-tray{margin-left:auto;display:flex;align-items:center;gap:12px;padding-right:8px;font-size:12px;color:#EAF4FF;white-space:nowrap}
.vt-clock{font-size:13px;color:#FFF;text-shadow:0 0 6px rgba(255,255,255,.6)}
.vt-show{flex:none;width:9px;height:100%;border-left:1px solid rgba(255,255,255,.35);background:rgba(255,255,255,.08)}
@media (max-width:640px){.vt-txt{display:none}[data-testid="stMainBlockContainer"],.block-container{padding-left:4px !important;padding-right:4px !important}}
"""


def _barra_tareas(titulo: str) -> str:
    """Barra de tareas fija inferior. Una sola línea: el parser de markdown no la rompe."""
    hora = datetime.now().strftime("%H:%M")
    return (
        '<div id="vista-taskbar" title="' + html.escape(titulo, quote=True) + '">'
        '<div class="vt-orb"><span class="vt-sphere"></span><span class="vt-globe">🌎</span></div>'
        '<div class="vt-task vt-activo">Priorización sísmica · Santander</div>'
        '<div class="vt-tray"><span class="vt-txt">SGC · DANE · UNGRD</span>'
        '<span class="vt-clock">' + hora + '</span></div>'
        '<div class="vt-show"></div>'
        '</div>'
    )


def barra_titulo(titulo: str) -> str:
    """HTML de la barra de título Vista (30 px) con los tres botones de ventana."""
    texto = html.escape(titulo)
    return (
        '<div class="vista-tb"><span class="vista-tb-ico">🌎</span>'
        '<span class="vista-tb-txt">' + texto + '</span>'
        '<span class="vista-tb-btns"><span class="vb vb-min">─</span>'
        '<span class="vb vb-max">□</span><span class="vb vb-close">✕</span></span></div>'
    )


def aplicar_estilo(titulo: str) -> None:
    """Inyecta el CSS (un solo bloque <style>) y la barra de tareas fija."""
    st.markdown("<style>" + _CSS + "</style>", unsafe_allow_html=True)
    st.markdown(_barra_tareas(titulo), unsafe_allow_html=True)


def plantilla_plotly() -> None:
    """Registra y activa un template de Plotly en tema claro Vista."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Segoe UI, Tahoma, sans-serif", size=12, color="#1E1E1E"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=COLORWAY,
        title=dict(font=dict(color="#1E3D5C")),
        xaxis=dict(gridcolor="#E3ECF4", zerolinecolor="#E3ECF4", linecolor="#A0A0A0"),
        yaxis=dict(gridcolor="#E3ECF4", zerolinecolor="#E3ECF4", linecolor="#A0A0A0"),
        legend=dict(font=dict(color="#1E1E1E")),
        hoverlabel=dict(bgcolor="#FFFFE1", bordercolor="#767676", font=dict(color="#1E1E1E")),
    )
    pio.templates[NOMBRE_PLANTILLA] = tpl
    pio.templates.default = "plotly_white+" + NOMBRE_PLANTILLA
