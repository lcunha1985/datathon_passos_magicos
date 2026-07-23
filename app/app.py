"""Passos Mágicos — Risco de Defasagem Educacional.

Modelo: RandomForest calibrado · alvo = defasagem no ano SEGUINTE (T+1).
Base PEDE 2022/2023/2024 · 1.365 transições aluno-ano.
"""
import base64
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import core_negocio as neg  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402


@st.cache_resource(show_spinner=False)
def _pdf_mod():
    """Importa o gerador de PDF só quando for preciso.

    Ele puxa matplotlib e reportlab (~0,4 s), que não fazem falta a quem só usa a tela.
    """
    try:
        import pdf_report
        return pdf_report
    except Exception:
        return None

st.set_page_config(page_title="Passos Mágicos — Risco de Defasagem",
                   page_icon="🎓", layout="wide")

MODEL, FEATURES, THRESHOLD = core.MODEL, core.FEATURES, core.THRESHOLD
INDICADORES, INDS_6, PEDRAS = core.INDICADORES, core.INDS_6, core.PEDRAS
NOME_FEATURE, NOME_CURTO, MEDIANAS = core.NOME_FEATURE, core.NOME_CURTO, core.MEDIANAS
METRICAS, AUC_PR_CV = core.METRICAS, core.AUC_PR_CV

# Tema dos gráficos (reatribuído se o modo escuro estiver ligado)
_PAPER, _FONTC, _GRID, _AZUL, _AZUL_MED = "white", "#333", "#eee", "#2D82C4", "#AECBE6"


def _tema(fig):
    fig.update_layout(paper_bgcolor=_PAPER, plot_bgcolor=_PAPER, font=dict(color=_FONTC))
    return fig


def titulo_info(titulo, ajuda):
    """Título à esquerda + tooltip (?) nativo do Streamlit à direita."""
    c1, c2 = st.columns([0.9, 0.1])
    c1.markdown(f"**{titulo}**")
    c2.markdown("", help=ajuda)


def calcular_proba(linha):
    """Alias fino de core.prever() — mantido porque a UI o passa como callback."""
    return core.prever(linha)


@st.cache_data(show_spinner="Classificando alunos…")
def _classificar_lote(validos):
    """Classifica o lote. Em cache: reprocessa só quando o arquivo muda."""
    out = validos.copy()
    out["prob_risco"] = MODEL.predict_proba(out[FEATURES])[:, 1]
    out["faixa"] = out["prob_risco"].apply(lambda p: core.classificar(p)[0])
    out["sinalizado"] = np.where(out["prob_risco"] >= THRESHOLD, "Sim", "Não")
    return out


# ─────────────────────────── SHAP ────────────────────────────────────────────
@st.cache_resource
def _explainer():
    import shap
    rf = MODEL.named_steps["mdl"].calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(rf)


def shap_contribs(linha):
    """Contribuições SHAP por feature original (agrega o one-hot da Pedra).

    Positivo = empurra para o RISCO. Retorna (Series, valor base).
    """
    pre = MODEL.named_steps["pre"]
    Xt = pre.transform(core.montar_X(linha))
    ex = _explainer()
    sv = np.array(ex.shap_values(Xt))
    sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]        # classe 1 = risco
    base = ex.expected_value
    base = float(base[1]) if hasattr(base, "__len__") else float(base)
    nomes = list(pre.get_feature_names_out())
    agg = {}
    for n, v in zip(nomes, sv):
        chave = n.split("__", 1)[1]
        if chave.startswith("Pedra"):
            chave = "Pedra"
        agg[chave] = agg.get(chave, 0.0) + float(v)
    return pd.Series(agg), base


def semaforo(contribs, com_valores=False):
    c = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    linhas = []
    for f, v in c.items():
        emoji = "🔴" if v > 0 else ("🟢" if v < 0 else "⚪")
        extra = f" `{v:+.3f}`" if com_valores else ""
        linhas.append(f"{emoji} {NOME_FEATURE.get(f, f)}{extra}")
    return "\n\n".join(linhas)


# ─────────────────────────── gráficos ────────────────────────────────────────
def gauge(proba):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e74c3c"},
               "steps": [{"range": [0, core.FAIXA_MOD * 100], "color": "#d5f5e3"},
                         {"range": [core.FAIXA_MOD * 100, core.FAIXA_ALTO * 100],
                          "color": "#fdebd0"},
                         {"range": [core.FAIXA_ALTO * 100, 100], "color": "#fadbd8"}],
               "threshold": {"line": {"color": "black", "width": 4},
                             "thickness": 0.75, "value": THRESHOLD * 100}}))
    fig.update_layout(height=340, margin=dict(t=30, b=10))
    return _tema(fig)


def barras_pares(linha):
    rot = [NOME_CURTO[i] for i in INDS_6]
    y_al = [core.to_num(linha[i]) for i in INDS_6]
    y_md = [MEDIANAS[i] for i in INDS_6]
    diffs = [a - m for a, m in zip(y_al, y_md)]
    # customdata com tipos mistos (float+string) faz o plotly perder a formatação
    # numérica no hover; pré-formatamos tudo como string.
    cdata = [[f"{a:.2f}", f"{d:+.2f}",
              f"{(d / m * 100 if m else 0.0):+.2f}%"]
             for a, d, m in zip(y_al, diffs, y_md)]
    cor = ["#e74c3c" if d < 0 else _AZUL for d in diffs]
    fig = go.Figure()
    fig.add_bar(name="Aluno", x=rot, y=y_al, marker_color=_AZUL, customdata=cdata,
                hoverlabel=dict(bgcolor="white", bordercolor=cor, font=dict(color=cor)),
                hovertemplate=("<b>%{x}</b><br>Aluno: %{customdata[0]}<br>"
                               "Diferença: %{customdata[1]} pts "
                               "(%{customdata[2]})<extra></extra>"))
    fig.add_bar(name="Média da base", x=rot, y=y_md, marker_color=_AZUL_MED,
                hoverlabel=dict(bgcolor="white", bordercolor=_AZUL_MED,
                                font=dict(color="#4A4A4A")),
                hovertemplate="<b>%{x}</b><br>Média da base: %{y:.2f}<extra></extra>")
    fig.update_layout(barmode="group", height=340, margin=dict(t=30),
                      yaxis=dict(range=[0, 10], gridcolor=_GRID, tickformat=".2f"),
                      legend=dict(orientation="h", y=1.14))
    return _tema(fig)



def grafico_waterfall(contribs, base):
    c = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(c),
        y=["Ponto de partida"] + [NOME_FEATURE.get(f, f) for f in c.index],
        x=[base] + list(c.values),
        connector={"line": {"color": "#ccc"}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#2ecc71"}}))
    fig.update_layout(height=420, margin=dict(t=30, l=10))
    fig.update_xaxes(showticklabels=False, title="")
    fig.update_yaxes(autorange="reversed")
    return _tema(fig)


def grafico_importancia():
    itens = sorted(core.IMPORTANCIA.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in itens],
                           y=[NOME_FEATURE.get(k, k) for k, _ in itens],
                           orientation="h", marker_color=_AZUL,
                           text=[f"{v:.3f}" for _, v in itens], textposition="outside"))
    fig.update_layout(height=400, margin=dict(t=30, l=10, r=40),
                      xaxis_title="Δ AUC ao embaralhar a variável")
    return _tema(fig)


def grafico_calibracao():
    d = core.FAIXA_REALIDADE
    fig = go.Figure(go.Bar(x=[x["faixa"].replace("\n", "<br>") for x in d],
                           y=[x["taxa_real"] for x in d],
                           marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
                           text=[f"{x['taxa_real']:.1f}%<br>n={x['n']}" for x in d],
                           textposition="outside"))
    fig.update_layout(height=350, margin=dict(t=30),
                      yaxis=dict(title="defasagem real no ano seguinte (%)",
                                 range=[0, 115], gridcolor=_GRID),
                      xaxis_title="faixa de risco prevista")
    return _tema(fig)


def curva_sensibilidade(linha, ind):
    xs = [round(v, 1) for v in np.arange(0, 10.01, 0.5)]
    cenarios = [{**linha, ind: v} for v in xs]
    ys = core.prever_lote(cenarios) * 100          # uma única chamada ao modelo
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers",
                               line=dict(color=_AZUL, width=3)))
    fig.add_vline(x=core.to_num(linha[ind]), line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"atual ({core.to_num(linha[ind]):.1f})")
    fig.add_hline(y=THRESHOLD * 100, line_dash="dot", line_color="#888",
                  annotation_text=f"limiar ({THRESHOLD*100:.0f}%)")
    fig.update_layout(height=360, margin=dict(t=30),
                      xaxis_title=NOME_FEATURE.get(ind, ind),
                      yaxis=dict(title="Risco previsto (%)", range=[0, 100], gridcolor=_GRID))
    return _tema(fig)


def heatmap_sensibilidade(linha, ix, iy, passo=1.0):
    vals = [round(v, 1) for v in np.arange(0, 10.01, passo)]
    # monta a grade inteira e prevê de uma vez (era 121 chamadas separadas)
    cenarios = [{**linha, ix: vx, iy: vy} for vy in vals for vx in vals]
    Z = (core.prever_lote(cenarios) * 100).reshape(len(vals), len(vals))
    fig = go.Figure(go.Heatmap(z=Z, x=vals, y=vals, colorscale="RdYlGn_r",
                               zmin=0, zmax=100, colorbar=dict(title="Risco %")))
    fig.add_scatter(x=[core.to_num(linha[ix])], y=[core.to_num(linha[iy])], mode="markers",
                    marker=dict(color="#2D82C4", size=13, line=dict(color="white", width=2)),
                    name="aluno atual")
    fig.update_layout(height=420, margin=dict(t=30),
                      xaxis_title=NOME_FEATURE.get(ix, ix),
                      yaxis_title=NOME_FEATURE.get(iy, iy))
    return _tema(fig)


def boxplot_base(linha):
    fig = go.Figure()
    for i in INDS_6:
        d = core.DIST_BASE[i]
        fig.add_box(y=[NOME_CURTO[i]], q1=[d["q1"]], median=[d["med"]], q3=[d["q3"]],
                    lowerfence=[d["min"]], upperfence=[d["max"]], orientation="h",
                    showlegend=False, marker_color=_AZUL_MED, line=dict(color=_AZUL))
    fig.add_scatter(x=[core.to_num(linha[i]) for i in INDS_6],
                    y=[NOME_CURTO[i] for i in INDS_6], mode="markers",
                    marker=dict(color="#e74c3c", size=12, symbol="diamond",
                                line=dict(color="white", width=1)), name="este aluno")
    fig.update_layout(height=380, margin=dict(t=30),
                      xaxis=dict(range=[0, 10], gridcolor=_GRID),
                      legend=dict(orientation="h", y=1.1))
    return _tema(fig)


def diagrama_desenho():
    fig = go.Figure()
    caixas = [(0.5, "Ano T<br>(indicadores)", _AZUL),
              (2.0, "Modelo<br>aprende", "#6b7f92"),
              (3.5, "Ano T+1<br>(defasado?)", "#e74c3c")]
    for x, txt, cor in caixas:
        fig.add_shape(type="rect", x0=x - 0.5, x1=x + 0.5, y0=0.35, y1=0.75,
                      fillcolor=cor, line=dict(color=cor), opacity=0.9)
        fig.add_annotation(x=x, y=0.55, text=txt, showarrow=False,
                           font=dict(color="white", size=12))
    for x0, x1 in [(1.05, 1.45), (2.55, 2.95)]:
        fig.add_annotation(x=x1, y=0.55, ax=x0, ay=0.55, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3,
                           arrowsize=1.4, arrowcolor=_FONTC)
    fig.add_annotation(x=2.0, y=0.15,
                       text="treina só com o passado → prevê o ano seguinte (sem vazamento)",
                       showarrow=False, font=dict(color=_FONTC, size=12))
    fig.update_xaxes(visible=False, range=[-0.2, 4.2])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=190, margin=dict(t=6, b=6))
    return _tema(fig)


def render_laudo(linha, proba, chave=""):
    """Laudo pedagógico na tela: cards + narrativa + gráficos + alavancas + atividades.

    Usado na Predição Individual e na linha selecionada da Predição em Lote.
    """
    label, bg = core.classificar(proba)
    st.markdown(f"<div class='risk-banner' style='background:{bg}; padding:16px; "
                f"border-radius:8px; text-align:center; font-size:24px; "
                f"font-weight:bold; color:#1a1a1a;'>{label}</div>",
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chance de defasagem (ano seguinte)", f"{proba*100:.1f}%")
    k2.metric("Defasagem hoje", f"{int(core.to_num(linha['Defasagem']))} ano(s)")
    k3.metric("Estágio", str(linha["Pedra"]))
    k4.metric("Acompanhamento prioritário", "Sim ✋" if proba >= THRESHOLD else "Não")
    st.caption("**Como ler:** 🟢 0–36% baixo · 🟡 36–67% moderado · 🔴 67–100% alto · "
               f"▎traço preto = limiar de triagem ({THRESHOLD*100:.0f}%)")

    # ── narrativa em linguagem natural (mesma do PDF) ──
    st.markdown("### 📋 Leitura pedagógica")
    abertura, paragrafos = core.narrativa_pedagogica(linha, proba)
    st.markdown(abertura)
    for p in paragrafos:
        st.markdown(p)

    alerta, _ = core.risco_evasao(linha)
    if alerta:
        st.warning("🚪 **Atenção à permanência** — sinal heurístico, não é saída do modelo.")

    g, r = st.columns(2)
    with g:
        titulo_info("Chance de defasagem no ano seguinte",
                    "Probabilidade estimada de o aluno estar em defasagem no próximo ano. "
                    "O traço preto marca o limiar de triagem.")
        st.plotly_chart(gauge(proba), width='stretch', key=f"g_{chave}")
    with r:
        titulo_info("Aluno × média da base",
                    "Compara cada indicador do aluno (barra escura) com a média da base "
                    "(barra clara). Passe o mouse para ver a diferença em pontos e %.")
        st.plotly_chart(barras_pares(linha), width='stretch', key=f"b_{chave}")

    # ── alavancas ──
    try:
        base_p, cenarios = core.detectar_alavancas(linha, calcular_proba)
    except Exception:
        base_p, cenarios = None, []

    st.markdown("#### 🎯 Onde focar — maiores alavancas para este aluno")
    st.caption("O app testa melhorar cada indicador (e combinações) e mede quanto o risco "
               "cairia. É análise de sensibilidade do modelo (associação), não promessa causal.")
    if not cenarios:
        st.success("Indicadores já em bom nível — sem alavancas relevantes. 👏")
    else:
        m = cenarios[0]
        nomes = ", ".join(NOME_FEATURE.get(v, v).split(" (")[0] for v in m["vars"])
        st.markdown(f"💪 **Maior alavanca:** trabalhar **{nomes}** em conjunto levaria o risco "
                    f"de **{base_p*100:.0f}%** para **{m['novo_risco']*100:.0f}%** "
                    f"(queda de **{m['ganho']:.0f} pontos**).")
        _linhas = "".join(
            f"<tr><td>{' + '.join(NOME_FEATURE.get(v, v).split(' (')[0] for v in c['vars'])}</td>"
            f"<td>{c['novo_risco']*100:.0f}%</td>"
            f"<td>{round(c['ganho'], 1)}</td></tr>"
            for c in cenarios)
        st.markdown(
            "<table class='lev-table'>"
            "<thead><tr><th>Trabalhar em conjunto</th><th>Risco depois</th>"
            "<th>Queda (p.p.)</th></tr></thead>"
            f"<tbody>{_linhas}</tbody></table>",
            unsafe_allow_html=True)

    # ── atividades e encaminhamentos ──
    ativ = core.atividades_para(linha, cenarios)
    if ativ:
        st.markdown("#### 🧭 O que fazer — atividades sugeridas")
        for rotulo, lista in ativ:
            st.markdown(f"**{rotulo}**")
            for a in lista:
                st.markdown(f"- {a}")

    enc = core.encaminhamentos(linha)
    if enc:
        st.markdown("#### 📨 Encaminhamento sugerido")
        for e in enc:
            st.markdown(f"- {e}")

    return cenarios


@st.cache_data(show_spinner=False, max_entries=20)
def _laudo_pdf_bytes(linha, proba, label):
    """PDF em cache: o download_button precisa dos bytes ANTES do clique, então sem
    cache o PDF seria remontado a cada renderização da página."""
    mod = _pdf_mod()
    return None if mod is None else mod.laudo_pedagogico_pdf(linha, proba, label,
                                                             calcular_proba)


@st.cache_data(show_spinner=False, max_entries=20)
def _tecnico_pdf_bytes(linha):
    mod = _pdf_mod()
    return None if mod is None else mod.relatorio_tecnico_pdf(linha, calcular_proba)


def botao_laudo_pdf(linha, proba, chave=""):
    """Botão único: gera o laudo e já dispara o download (padrão do app)."""
    botao_gerar_e_baixar(
        "📄 Gerar laudo pedagógico (PDF)",
        lambda: _laudo_pdf_bytes(linha, proba, core.classificar(proba)[0]),
        "laudo_passos_magicos.pdf", key=f"laudo_{chave}")


# ─────────────────────────── header ──────────────────────────────────────────
@st.cache_data
def _asset_b64(nome):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", nome)
    return base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else None


_logo = _asset_b64("logo.png")
_fiap = _asset_b64("fiap.png")
_img_logo = (f"<img src='data:image/png;base64,{_logo}' style='height:58px;background:#fff;"
             f"padding:6px 10px;border-radius:8px;margin-right:18px'>") if _logo else ""
_img_fiap = (f"<img src='data:image/png;base64,{_fiap}' "
             f"style='height:44px;margin-left:auto;opacity:0.95'>") if _fiap else ""

# ─────────────────────────── sidebar / seletor de página ─────────────────────
MODO = st.sidebar.radio("🖥️ Modo de exibição",
                        ["🙂 Análise Pedagógica", "📊 Análise de Negócio",
                         "🔬 Análise Técnica"], index=0,
                        help="Pedagógica: uso da equipe. Negócio: as 10 perguntas analíticas "
                             "do case. Técnica: dashboard do modelo.")
TECNICO = MODO.startswith("🔬")
NEGOCIO = MODO.startswith("📊")
DARK = st.sidebar.toggle("🌙 Modo escuro", value=False)
st.sidebar.caption("A **Análise Pedagógica** é o uso diário da equipe. A **Análise Técnica** "
                   "troca a página para o dashboard do modelo (usa o último aluno calculado).")
st.sidebar.divider()

# O banner "Risco de Defasagem Educacional" é o cabeçalho da Análise Pedagógica/Técnica —
# a Análise de Negócio tem o seu próprio banner dentro de render_negocio(), então pula este
# aqui pra não duplicar cabeçalho na mesma página.
if not NEGOCIO:
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #2D82C4 0%, #1F5C8F 100%);
                    padding: 20px 26px; border-radius: 12px; margin-bottom: 8px;
                    display:flex; align-items:center;'>
            {_img_logo}
            <div style='flex:1;'>
                <h1 style='color: white; margin: 0; font-size: 26px;'>
                    Risco de Defasagem Educacional</h1>
                <p style='color: #D6E8F7; margin: 6px 0 0 0; font-size: 14px;'>
                    Estima a chance de o aluno estar em defasagem no <b>ano seguinte</b>,
                    a partir dos indicadores de hoje — para o apoio chegar antes.</p>
            </div>
            {_img_fiap}
        </div>
        """, unsafe_allow_html=True)
    st.info("ℹ️ Ferramenta de **apoio à decisão**, não decisão automática. "
            "🔒 Nenhum dado inserido é armazenado.")
_dt = core.DATA_TREINO
try:
    _dt = pd.to_datetime(core.DATA_TREINO).strftime("%d/%m/%Y")
except Exception:
    pass
st.sidebar.caption(f"🧠 **Modelo {core.VERSAO}** · última atualização: **{_dt}**  \n"
                   f"Base PEDE 2022–2024 · 1.365 transições  \n"
                   f"Limiar de triagem: {THRESHOLD:.2f}")

if DARK:
    _PAPER, _FONTC, _GRID, _AZUL, _AZUL_MED = "#18232f", "#e6edf3", "#263543", "#4AA3E0", "#3a536b"
    st.markdown("""
    <style>
      .stApp, [data-testid="stAppViewContainer"] { background-color: #0f1720; }
      [data-testid="stSidebar"] { background-color: #141d27; }
      /* Barra superior do Streamlit (Deploy / menu) — escura em dark mode */
      [data-testid="stHeader"], header[data-testid="stHeader"] {
        background-color: #0f1720 !important;
      }
      [data-testid="stHeader"] * { color: #9fb0c0 !important; }

      h1,h2,h3,h4,h5,h6,p,span,label,li,div,
      [data-testid="stMarkdownContainer"] { color: #e6edf3 !important; }
      [data-testid="stMetric"] { background-color: #18232f; border: 1px solid #263543;
                                 border-radius: 8px; padding: 8px; }
      [data-testid="stExpander"] { background-color: #18232f; border-color: #263543; }
      [data-testid="stCaptionContainer"] { color: #9fb0c0 !important; }

      /* Banner de risco: força fonte ESCURA nas classes que aplicamos */
      .risk-banner, .risk-banner * { color: #1a1a1a !important; }

      /* Tabela de alavancas em HTML puro: fonte BRANCA em fundo escuro */
      .lev-table, .lev-table * { color: #e6edf3 !important; }
      .lev-table { width:100%; border-collapse:collapse; margin: 4px 0 14px 0; }
      .lev-table th, .lev-table td {
        background-color: #18232f !important;
        border: 1px solid #263543 !important;
        padding: 8px 12px !important;
        text-align: left !important;
      }
      .lev-table th { background-color: #141d27 !important; font-weight: 600 !important; }

      /* Widgets de entrada: fundo CLARO + texto ESCURO (legível em dark mode) */
      .stButton button, .stDownloadButton button,
      [data-testid="stPopover"] button,
      [data-baseweb="select"] > div,
      [data-testid="stNumberInput"] > div > div {
        background-color: #f0f4f8 !important;
      }
      .stButton button, .stButton button *,
      .stDownloadButton button, .stDownloadButton button *,
      [data-testid="stPopover"] button, [data-testid="stPopover"] button *,
      [data-baseweb="select"] div, [data-baseweb="select"] span,
      [data-baseweb="popover"] li, [role="option"], [role="listbox"] *,
      [data-testid="stNumberInput"] input, [data-testid="stNumberInput"] button {
        color: #1a1a1a !important;
      }
      [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
        color: #e6edf3 !important;
      }

      /* Botões de PDF/CSV: mesmo estilo do expander (cinza escuro + fonte branca) */
      [class*="st-key-btn_pdf_gerar"] button,
      [class*="st-key-btn_pdf_baixar"] button,
      [class*="st-key-btn_pdf_gerar"] button *,
      [class*="st-key-btn_pdf_baixar"] button * {
        background-color: #2b3a4b !important;
        color: #e6edf3 !important;
        border-color: #2b3a4b !important;
      }
    </style>
    """, unsafe_allow_html=True)

DEFAULTS = {"defas": 0, "fase": 3, "idade": 13, "tempo": 3, "pedra": "Ametista",
            "ida": 7, "ieg": 9, "ipv": 8, "iaa": 9, "ips": 7, "ipp": 8}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Persistência entre páginas: quando o seletor troca para a Análise Técnica, os widgets
# da Análise Pedagógica não são renderizados (st.stop()) e o Streamlit descartaria o
# estado deles — o que resetaria tudo o que o usuário digitou. Reatribuir as chaves a
# cada execução mantém o estado vivo.
for _k in list(DEFAULTS) + [f"sim_{f}" for f in INDS_6]:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


# ─────────────────────────── página técnica ──────────────────────────────────
AG = neg.AGREGADOS
PEDRA_ORDEM = neg.PEDRA_ORDEM
_VERDE, _LARANJA, _VERMELHO = "#2ecc71", "#f39c12", "#e74c3c"
_DICA_LEGENDA = "Clique na legenda para ocultar uma série; duplo clique isola só ela."


def _fmt_p(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}"


def _cat(fig):
    """Força o eixo X a ser categórico (nunca desenha anos fracionados como
    '2020.5' entre os rótulos) e mantém o eixo Y sem faixa fixa, para que
    isolar uma série pela legenda reajuste a escala automaticamente."""
    fig.update_xaxes(type="category")
    fig.update_yaxes(autorange=True)
    return fig


# ─────────────────────────── botão padrão de relatório (1 clique) ────────────
def _auto_download(dados, filename, mime="application/pdf"):
    """Dispara o download assim que os bytes existem, sem precisar de um
    segundo botão "Baixar". st.markdown não executa <script>, por isso usa
    components.html (roda dentro de um iframe, onde o script é executado)."""
    b64 = base64.b64encode(dados).decode()
    components.html(
        f"""<a id="dl" href="data:{mime};base64,{b64}" download="{filename}"></a>
        <script>document.getElementById('dl').click();</script>""",
        height=0, width=0,
    )


def botao_gerar_e_baixar(label, gerar_fn, filename, key):
    """Botão único e padrão para PDFs do app: ao clicar, gera o arquivo e já
    dispara o download — não existe um segundo botão só para baixar."""
    if st.button(label, width='stretch', key=f"btn_{key}"):
        with st.spinner("Montando o arquivo…"):
            dados = gerar_fn()
        if dados is None:
            st.error("Não foi possível gerar o arquivo agora.")
        else:
            st.session_state[f"_dl_pronto_{key}"] = dados
    dados = st.session_state.pop(f"_dl_pronto_{key}", None)
    if dados is not None:
        _auto_download(dados, filename)
        st.success(f"✅ **{filename}** gerado. Se o download não abrir sozinho, seu "
                   "navegador pode ter bloqueado o pop-up — clique no botão de novo e permita.")


def render_negocio():
    # Paleta de Pedra em tons de azul, na mesma lógica clara/escura que _AZUL e
    # _AZUL_MED já usam no resto do app (recalculada a cada execução, então
    # acompanha o toggle de modo escuro em vez de ficar presa a um tom fixo).
    if DARK:
        CORES_PEDRA = {"Quartzo": "#3a536b", "Agata": "#5A85A8",
                       "Ametista": _AZUL, "Topázio": "#8FCBFF"}
    else:
        CORES_PEDRA = {"Quartzo": _AZUL_MED, "Agata": "#6FAADB",
                       "Ametista": _AZUL, "Topázio": "#1F5C8F"}

    _img_logo = (f"<img src='data:image/png;base64,{_asset_b64('logo.png')}' "
                 f"style='height:58px;background:#fff;padding:6px 10px;border-radius:8px;"
                 f"margin-right:18px'>") if _asset_b64("logo.png") else ""
    _img_fiap = (f"<img src='data:image/png;base64,{_asset_b64('fiap.png')}' "
                 f"style='height:44px;margin-left:auto;opacity:0.95'>") if _asset_b64("fiap.png") else ""
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, #2D82C4 0%, #1F5C8F 100%);
                    padding: 20px 26px; border-radius: 12px; margin-bottom: 8px;
                    display:flex; align-items:center;'>
            {_img_logo}
            <div style='flex:1;'>
                <h1 style='color: white; margin: 0; font-size: 26px;'>
                    Análise de Negócio — 10 Perguntas do Case</h1>
                <p style='color: #D6E8F7; margin: 6px 0 0 0; font-size: 14px;'>
                    Leitura analítica dos indicadores pedagógicos ao longo do ciclo, com
                    insight acionável em cada pergunta.</p>
            </div>
            {_img_fiap}
        </div>
        """, unsafe_allow_html=True)
    st.info(
        "ℹ️ Base: **PEDE_PASSOS_DATASET_FIAP.csv** (oficial FIAP) · anos **2020, 2021, 2022** · "
        "1.349 alunos · 2.275 linhas aluno-ano. Uso atual: *desenvolvimento/teste*. Quando a "
        "base **2024** (mesma do modelo em produção) estiver disponível, os agregados devem "
        "ser reprocessados para **homologação** fora da amostra."
    )

    anos_str = [str(a) for a in neg.ANOS_BASE]

    # ── 1. IAN ──────────────────────────────────────────────────────────────
    titulo_info("1. Adequação do nível (IAN)",
                "Perfil de defasagem dos alunos e como ele evolui ano a ano.")
    q1 = AG["q1_faixa_ian_ano"]
    st.caption(f"ℹ️ Barras empilhadas somando 100% dos alunos por ano, divididos pela "
               f"situação de defasagem (IAN). {_DICA_LEGENDA}")
    fig = go.Figure()
    for faixa, cor in [("Sem defasagem", _VERDE), ("Moderada", _LARANJA), ("Severa", _VERMELHO)]:
        fig.add_bar(name=faixa, x=anos_str, y=[q1[faixa][a] for a in anos_str], marker_color=cor)
    fig.update_layout(barmode="stack", height=380, margin=dict(t=20),
                       yaxis=dict(title="% de alunos", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q1_{DARK}")
    st.caption(
        f"Em 2022, **{AG['q1_severa_moderada_2022_pct']}%** dos alunos estavam com defasagem "
        f"moderada ou severa (IAN < 10) — contra "
        f"{q1['Moderada']['2020'] + q1['Severa']['2020']:.1f}% em 2020. A faixa **Severa** "
        "também cresceu (1,5% → 3,2%)."
    )
    st.info(
        "💡 **Insight acionável:** a proporção de alunos sem defasagem caiu de forma "
        "consistente ano a ano (49,4% → 38,8% → 30,0%). Vale cruzar com `tempo_prog` para "
        "saber se é entrada de alunos já defasados ou perda de ritmo dos que já estavam no "
        "programa."
    )

    # ── 2. IDA ──────────────────────────────────────────────────────────────
    titulo_info("2. Desempenho acadêmico (IDA)",
                "Se o desempenho médio está melhorando, estagnado ou caindo por fase e ano.")
    q2 = AG["q2_ida_pedra_ano"]
    st.caption(f"ℹ️ Uma linha por fase (Pedra) — a nota média (IDA, 0 a 10) daquela fase em "
               f"cada ano. {_DICA_LEGENDA}")
    fig = go.Figure()
    for pedra in PEDRA_ORDEM:
        if pedra not in q2:
            continue
        fig.add_trace(go.Scatter(x=anos_str, y=[q2[pedra][a] for a in anos_str],
                                  mode="lines+markers", name=pedra,
                                  line=dict(color=CORES_PEDRA.get(pedra), width=3)))
    fig.update_layout(height=380, margin=dict(t=20),
                       yaxis=dict(title="IDA médio (0–10)", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q2_{DARK}")
    kw = AG["q2_kruskal_ano"]
    st.caption(
        f"IDA médio geral: {AG['q2_ida_medio_ano']['2020']} (2020) → "
        f"{AG['q2_ida_medio_ano']['2021']} (2021) → {AG['q2_ida_medio_ano']['2022']} (2022) — "
        f"queda em 2021, recuperação parcial em 2022 (Kruskal-Wallis H={kw['H']}, "
        f"{_fmt_p(kw['p'])}: diferença entre anos é estatisticamente significativa)."
    )
    st.info(
        "💡 **Insight acionável:** a queda de 2021 aparece em **todas** as Pedras (inclusive "
        "Topázio, o grupo mais maduro) — padrão consistente com um efeito de período (ex.: "
        "pandemia/ensino remoto) mais do que um problema pontual de fase. A recuperação de "
        "2022 não repõe o nível de 2020 em nenhuma Pedra."
    )

    # ── 3. IEG × IDA × IPV ──────────────────────────────────────────────────
    titulo_info("3. Engajamento (IEG) × Desempenho (IDA) × Ponto de Virada (IPV)",
                "Se o engajamento tem relação direta com desempenho e ponto de virada.")
    corr = AG["q3_corr"]
    labels = ["IEG", "IDA", "IPV"]
    z = [[corr[a][b] for b in labels] for a in labels]
    st.caption("ℹ️ Correlação entre cada par de indicadores (−1 a 1): quanto mais escuro, "
               "mais forte a relação. Calculado com os 3 anos juntos (a pergunta é sobre "
               "relação entre indicadores, não sobre evolução no tempo).")
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="Blues",
                                text=[[f"{v:.2f}" for v in row] for row in z],
                                texttemplate="%{text}"))
    fig.update_layout(height=340, margin=dict(t=20))
    st.plotly_chart(_tema(fig), width='stretch', key=f"neg_q3_{DARK}")
    st.caption(
        f"IEG correlaciona **{corr['IEG']['IDA']:.2f}** com IDA e **{corr['IEG']['IPV']:.2f}** "
        "com IPV (Spearman) — relação moderada e positiva em ambos, mais forte com desempenho "
        "do que com protagonismo."
    )
    q3p = AG["q3_corr_ieg_ida_por_pedra"]
    st.caption(
        "Por Pedra, a correlação IEG↔IDA é bem mais fraca dentro de cada grupo "
        f"({', '.join(f'{p}: {v:.2f}' for p, v in q3p.items())}) — parte da correlação geral "
        "vem da diferença **entre** fases, não de uma relação individual forte dentro de cada "
        "nível de maturidade."
    )
    with st.expander("📅 Checagem de estabilidade por ano"):
        st.caption(f"ℹ️ A mesma correlação IEG↔IDA e IEG↔IPV, calculada separadamente em cada "
                   f"ano — mostra se a força da relação é estável ou muda com o tempo. "
                   f"{_DICA_LEGENDA}")
        q3ida = AG["q3_corr_ieg_ida_por_ano"]
        q3ipv = AG["q3_corr_ieg_ipv_por_ano"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anos_str, y=[q3ida[a] for a in anos_str],
                                  mode="lines+markers", name="IEG↔IDA", line=dict(color=_AZUL, width=3)))
        fig.add_trace(go.Scatter(x=anos_str, y=[q3ipv[a] for a in anos_str],
                                  mode="lines+markers", name="IEG↔IPV", line=dict(color=_LARANJA, width=3)))
        fig.update_layout(height=300, margin=dict(t=20),
                           yaxis=dict(title="Correlação (Spearman)", gridcolor=_GRID))
        st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q3_ano_{DARK}")
        st.caption(
            "A correlação IEG↔IPV varia bastante entre anos (0,27 em 2020 → 0,71 em 2021 → "
            "0,55 em 2022) — não é uma relação estável, então generalizar um único número "
            "agregado para os três anos exige cautela. IEG↔IDA é mais estável (0,51–0,63)."
        )
    st.info(
        "💡 **Insight acionável:** engajamento importa mais como diferenciador entre fases do "
        "que como preditor fino dentro de uma fase — programas de estímulo ao engajamento "
        "tendem a ajudar mais a avançar de Pedra do que a subir nota dentro da mesma Pedra."
    )

    # ── 4. IAA ──────────────────────────────────────────────────────────────
    titulo_info("4. Autoavaliação (IAA) × Desempenho real",
                "Se a percepção do aluno sobre si mesmo é coerente com IDA e IEG, e se isso "
                "está mudando ao longo do tempo.")
    st.caption(f"ℹ️ Autoavaliação (IAA) e nota real (IDA) médias, ano a ano — mostra se o "
               f"\"otimismo\" dos alunos sobre o próprio desempenho está aumentando, diminuindo "
               f"ou estável. {_DICA_LEGENDA}")
    iaa_ano, ida_ano = AG["q4_iaa_medio_por_ano"], AG["q4_ida_medio_por_ano"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos_str, y=[iaa_ano[a] for a in anos_str], mode="lines+markers",
                              name="Autoavaliação (IAA)", line=dict(color=_LARANJA, width=3)))
    fig.add_trace(go.Scatter(x=anos_str, y=[ida_ano[a] for a in anos_str], mode="lines+markers",
                              name="Nota real (IDA)", line=dict(color=_AZUL, width=3)))
    fig.update_layout(height=360, margin=dict(t=20), yaxis=dict(title="Média (0–10)", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q4_evo_{DARK}")
    mism_ano = AG["q4_mismatch_medio_por_ano"]
    kw4 = AG["q4_kruskal_mismatch_ano"]
    st.caption(
        f"O desvio (IAA − IDA) foi de {mism_ano['2020']} em 2020, {mism_ano['2021']} em 2021 e "
        f"{mism_ano['2022']} em 2022 — variação estatisticamente significativa entre anos "
        f"(Kruskal-Wallis H={kw4['H']}, {_fmt_p(kw4['p'])}), mas **sem tendência clara de "
        "melhora ou piora**: o desvio aumenta em 2021 (junto com a queda de nota real) e "
        "recua parcialmente em 2022, acompanhando o mesmo padrão do IDA."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Superestimadores", f"{AG['q4_pct_superestimadores']}%", help="IAA muito acima do IDA")
    c2.metric("Coerentes", f"{AG['q4_pct_coerentes']}%")
    c3.metric("Subestimadores", f"{AG['q4_pct_subestimadores']}%", help="IAA muito abaixo do IDA")
    with st.expander("📅 Checagem de estabilidade por ano"):
        st.caption(f"ℹ️ Correlação IAA↔IDA e % de superestimadores, calculados separadamente "
                   f"em cada ano.")
        q4c = AG["q4_corr_iaa_ida_por_ano"]
        q4s = AG["q4_pct_superestimadores_por_ano"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anos_str, y=[q4c[a] for a in anos_str], mode="lines+markers",
                                  name="Correlação IAA↔IDA", line=dict(color=_AZUL, width=3)))
        fig.update_layout(height=280, margin=dict(t=20),
                           yaxis=dict(title="Correlação (Spearman)", gridcolor=_GRID))
        st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q4_ano_{DARK}")
        st.caption(
            f"Correlação fraca em todos os anos (0,19–0,25) — resultado estável. Já o % de "
            f"superestimadores oscila mais: {q4s['2020']}% em 2020, {q4s['2021']}% em 2021, "
            f"{q4s['2022']}% em 2022."
        )
    st.info(
        "💡 **Insight acionável:** a autoavaliação sozinha não é um bom proxy de desempenho, e "
        "isso **não mudou** nos três anos — seu maior valor está no **desvio**: alunos "
        "superestimadores (IAA alto, IDA baixo) são um grupo prático de atenção para calibrar "
        "expectativa antes de uma frustração maior adiante."
    )

    # ── 5. IPS ──────────────────────────────────────────────────────────────
    titulo_info("5. Aspectos psicossociais (IPS)",
                "Se há padrões psicossociais que antecedem queda de desempenho ou engajamento, "
                "e se esse padrão é o mesmo nos dois períodos disponíveis.")
    p5 = AG["q5_por_par_ano"]
    st.caption(f"ℹ️ Nível psicossocial (IPS) médio no ano anterior, separado por período de "
               f"transição (2020→2021 e 2021→2022) e por grupo (quem caiu de nota no ano "
               f"seguinte vs. quem não caiu). {_DICA_LEGENDA}")
    fig = go.Figure()
    fig.add_bar(name="Caiu no IDA (ano seguinte)", x=["2020→2021", "2021→2022"],
                y=[p5["2020->2021"]["ips_medio_caiu"], p5["2021->2022"]["ips_medio_caiu"]],
                marker_color=_VERMELHO)
    fig.add_bar(name="Não caiu", x=["2020→2021", "2021→2022"],
                y=[p5["2020->2021"]["ips_medio_nao_caiu"], p5["2021->2022"]["ips_medio_nao_caiu"]],
                marker_color=_VERDE)
    fig.update_layout(barmode="group", height=360, margin=dict(t=20),
                       yaxis=dict(title="IPS médio no ano anterior (0–10)", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q5_{DARK}")
    p_2021 = p5["2020->2021"].get("mannwhitney_p")
    p_2022 = p5["2021->2022"].get("mannwhitney_p")
    st.caption(
        f"**2020→2021:** IPS médio de {p5['2020->2021']['ips_medio_caiu']} (caiu) vs "
        f"{p5['2020->2021']['ips_medio_nao_caiu']} (não caiu) — {_fmt_p(p_2021) if p_2021 else 'sem teste'}. "
        f"**2021→2022:** {p5['2021->2022']['ips_medio_caiu']} (caiu) vs "
        f"{p5['2021->2022']['ips_medio_nao_caiu']} (não caiu) — {_fmt_p(p_2022) if p_2022 else 'sem teste'}. "
        "Em **nenhum** dos dois períodos a diferença é estatisticamente significativa — o "
        "padrão se repete (ausência de efeito) nos dois pares de anos disponíveis."
    )
    st.warning(
        "⚠️ **Insight honesto:** ao contrário da hipótese inicial, o IPS isolado não se mostrou "
        "um antecedente estatístico de queda de desempenho em **nenhum** dos dois períodos "
        "testados (2020→2021 e 2021→2022) — não é um resultado isolado de um ano específico. "
        "Isso não descarta o efeito — pode estar diluído porque o IPS já reflete o ano "
        "corrente, não uma tendência. Vale testar com a base 2024 e/ou variação de IPS (Δ IPS) "
        "em vez do nível absoluto."
    )

    # ── 6. IPP × IAN ────────────────────────────────────────────────────────
    titulo_info("6. Aspectos psicopedagógicos (IPP) × Defasagem (IAN)",
                "Se a avaliação psicopedagógica confirma ou contradiz a defasagem do IAN.")
    ct = AG["q6_contingencia_ian_ipp"]
    terc = ["Baixo", "Médio", "Alto"]
    faixas = ["Sem defasagem", "Moderada", "Severa"]
    st.caption(f"ℹ️ Para cada faixa de defasagem (IAN), a distribuição do terço de IPP em que "
               f"os alunos daquela faixa se encontram. {_DICA_LEGENDA}")
    fig = go.Figure()
    for t, cor in zip(terc, [_VERMELHO, _LARANJA, _VERDE]):
        fig.add_bar(name=f"IPP {t}", x=faixas, y=[ct[t][f] for f in faixas], marker_color=cor)
    fig.update_layout(barmode="group", height=380, margin=dict(t=20),
                       yaxis=dict(title="% dentro da faixa de IAN", gridcolor=_GRID))
    st.plotly_chart(_tema(fig), width='stretch', key=f"neg_q6_{DARK}")
    st.caption(
        f"Correlação IPP↔IAN = **{AG['q6_corr_ipp_ian']:.2f}** (Spearman) — fraca, mas no "
        f"sentido esperado. **{AG['q6_pct_divergencia']}%** dos alunos estão no grupo "
        "divergente clássico: sem defasagem pelo IAN, mas com IPP no terço mais baixo."
    )
    with st.expander("📅 Checagem de estabilidade por ano"):
        st.caption("ℹ️ Correlação IPP↔IAN calculada separadamente em cada ano.")
        q6a = AG["q6_corr_ipp_ian_por_ano"]
        fig = go.Figure(go.Bar(x=anos_str, y=[q6a[a] for a in anos_str], marker_color=_AZUL))
        fig.update_layout(height=280, margin=dict(t=20),
                           yaxis=dict(title="Correlação IPP↔IAN (Spearman)", gridcolor=_GRID))
        st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q6_ano_{DARK}")
        st.caption(
            "A correlação é fraca em todos os anos e varia bastante em termos relativos "
            "(0,06 em 2020, 0,02 em 2021, 0,11 em 2022) — reforça que IPP e IAN não devem ser "
            "tratados como substitutos um do outro em nenhum ano isolado."
        )
    st.info(
        "💡 **Insight acionável:** o IPP confirma a direção do IAN, mas fracamente — medem "
        "coisas relacionadas, não a mesma coisa. O grupo divergente (~10%) é o caso mais útil "
        "na prática: aluno aparentemente bem pelo indicador de nível, mas sinalizado pela "
        "avaliação psicopedagógica — candidato a acompanhamento preventivo."
    )

    # ── 7. IPV ──────────────────────────────────────────────────────────────
    titulo_info("7. Ponto de Virada (IPV)",
                "Quais comportamentos mais influenciam o Ponto de Virada ao longo do tempo.")
    coef_ano = AG["q7_coef_por_ano"]
    st.caption(f"ℹ️ Peso de cada indicador (regressão logística) na chance de atingir o Ponto "
               f"de Virada, calculado separadamente em cada ano. {_DICA_LEGENDA}")
    fig = go.Figure()
    for ind, cor in zip(["IDA", "IEG", "IPS", "IPP"],
                         [_AZUL, _VERDE, _LARANJA, _AZUL_MED]):
        ys = [coef_ano[a][ind] for a in anos_str if a in coef_ano]
        xs = [a for a in anos_str if a in coef_ano]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=ind,
                                  line=dict(color=cor, width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color=_GRID)
    fig.update_layout(height=380, margin=dict(t=20),
                       yaxis=dict(title="Coeficiente padronizado (regressão logística)", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q7_{DARK}")
    taxa = AG["q7_taxa_pv_por_ano"]
    st.caption(
        f"Taxa de alunos que atingiram o Ponto de Virada: {taxa['2020']}% (2020), "
        f"{taxa['2021']}% (2021), {taxa['2022']}% (2022) — estável em torno de 13-16%."
    )
    st.warning(
        "⚠️ **Insight honesto:** o comportamento que mais influencia o Ponto de Virada **muda "
        "de ano para ano** — IPP lidera em 2020, IEG em 2021, IDA em 2022. Não há um único "
        "preditor dominante e estável nos três anos; IPP e IEG aparecem entre os dois maiores "
        "pesos com mais frequência, mas nenhum indicador domina de forma consistente. Isso é "
        "diferente de tratar a relação como fixa — o peso de cada comportamento parece "
        "sensível ao contexto de cada ano (ex.: o mesmo efeito de período visto nas perguntas "
        "2 e 10)."
    )

    # ── 8. INDE ─────────────────────────────────────────────────────────────
    titulo_info("8. Multidimensionalidade dos indicadores (INDE)",
                "Quais combinações de indicadores elevam mais a nota global do aluno.")
    corr8 = AG["q8_corr_inde"]
    st.caption("ℹ️ Correlação de cada indicador com a nota final (INDE), calculada com os 3 "
               "anos juntos (a pergunta é sobre peso relativo, não sobre evolução no tempo).")
    fig = go.Figure(go.Bar(x=list(corr8.keys()), y=list(corr8.values()), marker_color=_AZUL))
    fig.update_layout(height=340, margin=dict(t=20),
                       yaxis=dict(title="Correlação com INDE (Pearson)", gridcolor=_GRID))
    st.plotly_chart(_tema(fig), width='stretch', key=f"neg_q8_{DARK}")
    st.caption(
        f"Um modelo linear com IDA+IEG+IPS+IPP explica **{AG['q8_r2']*100:.1f}%** da variação "
        f"do INDE (R²). IDA e IEG dominam (correlações de {corr8['IDA']:.2f} e "
        f"{corr8['IEG']:.2f}), enquanto IPS ({corr8['IPS']:.2f}) e IPP ({corr8['IPP']:.2f}) "
        "contribuem bem menos."
    )
    with st.expander("📅 Checagem de estabilidade por ano"):
        st.caption(f"ℹ️ Correlação de cada indicador com o INDE, calculada separadamente em "
                   f"cada ano. {_DICA_LEGENDA}")
        q8a = AG["q8_corr_inde_por_ano"]
        q8r = AG["q8_r2_por_ano"]
        fig = go.Figure()
        for ind, cor in zip(["IDA", "IEG", "IPS", "IPP"], [_AZUL, _VERDE, _LARANJA, _AZUL_MED]):
            fig.add_trace(go.Scatter(x=anos_str, y=[q8a[a][ind] for a in anos_str],
                                      mode="lines+markers", name=ind, line=dict(color=cor, width=3)))
        fig.update_layout(height=320, margin=dict(t=20),
                           yaxis=dict(title="Correlação com INDE (Pearson)", gridcolor=_GRID))
        st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q8_ano_{DARK}")
        st.caption(
            f"IDA e IEG lideram em todos os anos — resultado estável. O R² do modelo também "
            f"se mantém alto e estável: {q8r['2020']} (2020), {q8r['2021']} (2021), "
            f"{q8r['2022']} (2022). IPP tem o maior salto entre anos (0,13 → 0,55 → 0,29)."
        )
    st.info(
        "💡 **Insight acionável:** a combinação que mais eleva o INDE é **IDA alto + IEG alto** "
        f"simultaneamente — pesam de forma parecida no modelo (coeficientes "
        f"{AG['q8_coef_inde']['IDA']:.2f} e {AG['q8_coef_inde']['IEG']:.2f}). IPS e IPP "
        "funcionam mais como suporte do que como alavancas diretas da nota global."
    )

    # ── 9. Modelo preditivo ─────────────────────────────────────────────────
    titulo_info("9. Previsão de risco com Machine Learning",
                "O modelo já em produção no app, reaproveitado aqui sem retreinar nada.")
    st.markdown(
        f"O app já tem em produção um modelo (**{core.VERSAO}**, RandomForest calibrado) que "
        "estima a probabilidade de um aluno entrar em risco de defasagem no ano seguinte. Veja "
        "a aba **🔬 Análise Técnica** para o dashboard completo, ou a **🙂 Análise Pedagógica** "
        "para simular um aluno específico."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC-PR (teste)", f"{core.METRICAS['AUC-PR']:.3f}")
    c2.metric("Recall (risco)", f"{core.METRICAS['Recall_risco']:.1%}")
    c3.metric("Precisão (risco)", f"{core.METRICAS['Precision_risco']:.1%}")
    st.caption("ℹ️ Para cada faixa de risco prevista pelo modelo, a % de alunos que realmente "
               "ficaram defasados no ano seguinte — quanto mais a barra sobe da esquerda para "
               "a direita, melhor o modelo ordena o risco.")
    st.plotly_chart(grafico_calibracao(), width='stretch', key=f"neg_q9_{DARK}")
    st.info(
        "💡 **Insight acionável:** o modelo separa bem os extremos — alunos em **Alto Risco** "
        f"têm {core.FAIXA_REALIDADE[2]['taxa_real']}% de chance real de defasagem, contra "
        f"{core.FAIXA_REALIDADE[0]['taxa_real']}% em **Baixo Risco**. Isso permite priorizar o "
        "acompanhamento pedagógico onde ele tem mais retorno esperado."
    )

    # ── 10. Efetividade do programa ─────────────────────────────────────────
    titulo_info("10. Efetividade do programa",
                "Se os indicadores mostram melhora consistente ao longo do ciclo nas fases.")
    q10 = AG["q10_inde_pedra_ano"]
    st.caption(f"ℹ️ Uma linha por fase (Pedra) — a nota final média (INDE) daquela fase em "
               f"cada ano. {_DICA_LEGENDA}")
    fig = go.Figure()
    for pedra in PEDRA_ORDEM:
        if pedra not in q10:
            continue
        fig.add_trace(go.Scatter(x=anos_str, y=[q10[pedra][a] for a in anos_str],
                                  mode="lines+markers", name=pedra,
                                  line=dict(color=CORES_PEDRA.get(pedra), width=3)))
    fig.update_layout(height=380, margin=dict(t=20), yaxis=dict(title="INDE médio (0–10)", gridcolor=_GRID))
    st.plotly_chart(_cat(_tema(fig)), width='stretch', key=f"neg_q10_{DARK}")
    kw10 = AG["q10_kruskal_ano"]
    st.caption(
        f"INDE médio geral: {AG['q10_inde_medio_ano']['2020']} (2020) → "
        f"{AG['q10_inde_medio_ano']['2021']} (2021) → {AG['q10_inde_medio_ano']['2022']} "
        f"(2022) — diferença estatisticamente significativa entre anos (H={kw10['H']}, "
        f"{_fmt_p(kw10['p'])}), mas na direção de **queda**, não de melhora consistente."
    )
    st.warning(
        "⚠️ **Insight honesto:** nesta janela (2020–2022) os dados não confirmam melhora "
        "consistente do INDE ao longo do ciclo — todas as Pedras caem de 2020 para 2021 e "
        "recuperam só parcialmente em 2022. A hierarquia entre Pedras (Topázio > Ametista > "
        "Agata > Quartzo) se mantém estável, o que valida o desenho das fases, mas a queda de "
        "2021 é o achado que mais chama atenção — provavelmente ligada a um fator externo ao "
        "programa (o mesmo padrão aparece no IDA, pergunta 2). A leitura definitiva de "
        "efetividade do ciclo completo só fica robusta com 2023/2024 na amostra."
    )

    # ── Relatório completo ──────────────────────────────────────────────────
    st.divider()
    titulo_info("Relatório completo (PDF)",
                "As 10 perguntas respondidas em linguagem simples, para quem não tem "
                "background técnico — com os mesmos gráficos desta página.")
    botao_gerar_e_baixar(
        "📄 Gerar relatório de Análise de Negócio (PDF)",
        lambda: _pdf_mod().relatorio_negocio_pdf() if _pdf_mod() else None,
        "analise_negocio_passos_magicos.pdf",
        key="negocio",
    )


def render_tecnica(linha):
    st.subheader("🔬 Análise Técnica do modelo")

    st.warning("**Achado desta versão:** a base traz a coluna `Fase Ideal`, e verificamos que "
               "`Defasagem = Fase − Fase Ideal` em **100%** dos casos — com `Fase Ideal` "
               "determinada pela `Idade`. Por isso **IAN e INDE foram excluídos** das features "
               "(são derivados da defasagem) e o alvo passou a ser a defasagem do **ano seguinte**, "
               "genuinamente desconhecida no momento da previsão.")

    titulo_info("Desempenho no teste out-of-time (2023→2024)",
                "Métricas medidas em transições de um período posterior ao treino. AUC-PR mede o "
                "acerto na classe de risco; recall, quantos alunos em risco foram capturados.")
    cols = st.columns(5)
    dados = [("AUC-PR", METRICAS["AUC-PR"], "Qualidade do ranqueamento na classe de risco."),
             ("AUC-ROC", METRICAS["AUC-ROC"], "Separação entre alunos com e sem risco."),
             ("Precisão", METRICAS["Precision_risco"], "Dos sinalizados, quantos eram risco."),
             ("Recall", METRICAS["Recall_risco"], "Dos alunos em risco, quantos capturamos."),
             ("F1", METRICAS["F1"], "Equilíbrio entre precisão e recall.")]
    for c, (n, v, h) in zip(cols, dados):
        c.metric(n, f"{v:.3f}", help=h)
    st.caption(f"Robustez — StratifiedGroupKFold por aluno: AUC-PR ≈ {AUC_PR_CV:.3f}. "
               f"Limiar {THRESHOLD:.2f} escolhido para recall ≥ 0,85.")

    titulo_info("Calibração — o modelo acerta a ordem do risco?",
                "Para cada faixa de risco prevista, a taxa real de defasagem observada no ano "
                "seguinte. Subir de forma monotônica confirma que o modelo ordena bem o risco.")
    st.plotly_chart(grafico_calibracao(), width='stretch', key=f"cal_{DARK}")
    st.caption("✅ **Resultado positivo:** 4,1% → 46,0% → 91,8%. A taxa real sobe de forma "
               "consistente com a faixa prevista.")

    titulo_info("Importância das variáveis (permutation importance)",
                "Quanto o AUC cai ao embaralhar cada variável. Mede a influência real de cada "
                "uma na previsão — não se é bom ou ruim.")
    st.plotly_chart(grafico_importancia(), width='stretch', key=f"imp_{DARK}")
    st.caption("A **defasagem atual** e a **fase** dominam — a defasagem é persistente, e o "
               "modelo é transparente quanto a isso. Entre os índices pedagógicos, **IPP** e "
               "**IPV** lideram.")

    titulo_info("Desenho do estudo (anti-vazamento)",
                "O modelo treina com os indicadores do ano T para prever a defasagem do ano T+1, "
                "sem usar informação do próprio ano que está prevendo.")
    st.plotly_chart(diagrama_desenho(), width='stretch', key=f"dia_{DARK}")

    if linha is None:
        st.info("ℹ️ Calcule um aluno na **Análise Pedagógica** para ver as análises específicas "
                "deste aluno (SHAP, sensibilidade, mapa de calor e posição na base).")
        _botao_pdf_tecnico(None)
        return

    proba = calcular_proba(linha)
    st.divider()
    st.markdown(f"### 👤 Último aluno calculado — risco previsto: **{proba*100:.1f}%**")

    titulo_info("Como cada variável puxou o risco (SHAP)",
                "Contribuição de cada variável para a previsão deste aluno: barras vermelhas "
                "aumentam o risco; verdes diminuem.")
    try:
        contribs, base = shap_contribs(linha)
        st.plotly_chart(grafico_waterfall(contribs, base), width='stretch')

        # valores numéricos — auditabilidade de cada decisão
        c = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
        def _fmt(v):
            return f"{v:g}" if isinstance(v, (int, float)) else str(v)
        tab_shap = pd.DataFrame({
            "Variável": [NOME_FEATURE.get(f, f) for f in c.index],
            "Valor do aluno": [_fmt(linha.get(f, "—")) for f in c.index],
            "Contribuição SHAP": [round(float(v), 4) for v in c.values],
            "Efeito": ["🔴 aumenta o risco" if v > 0 else
                       ("🟢 reduz o risco" if v < 0 else "⚪ neutro") for v in c.values]})
        st.dataframe(tab_shap, width='stretch', hide_index=True)
        soma = float(c.sum())
        st.caption(f"Valor base do modelo: **{base:.4f}** · soma das contribuições: "
                   f"**{soma:+.4f}** · saída (log-odds do RandomForest base): "
                   f"**{base + soma:.4f}**. As contribuições vêm do RandomForest não calibrado "
                   f"que sustenta o ensemble — por isso somam à saída dele, não diretamente à "
                   f"probabilidade calibrada exibida acima.")
    except Exception:
        st.caption("Explicação SHAP indisponível neste ambiente.")

    titulo_info("Sensibilidade — risco ao variar um indicador",
                "Varia um indicador de 0 a 10 mantendo o resto deste aluno fixo. Curva que desce "
                "indica que melhorar aquele indicador tende a reduzir o risco.")
    ind = st.selectbox("Indicador a analisar", INDS_6,
                       format_func=lambda k: NOME_FEATURE.get(k, k), key="sens_ind")
    st.plotly_chart(curva_sensibilidade(linha, ind), width='stretch')

    titulo_info("Mapa de calor — dois indicadores juntos",
                "Risco ao variar dois indicadores ao mesmo tempo. Revela o efeito combinado; o "
                "ponto azul é a posição atual do aluno.")
    h1, h2 = st.columns(2)
    ix = h1.selectbox("Eixo horizontal", INDS_6, index=0,
                      format_func=lambda k: NOME_FEATURE.get(k, k), key="hx")
    iy = h2.selectbox("Eixo vertical", INDS_6, index=2,
                      format_func=lambda k: NOME_FEATURE.get(k, k), key="hy")
    if ix != iy:
        st.plotly_chart(heatmap_sensibilidade(linha, ix, iy), width='stretch')
    else:
        st.info("Escolha dois indicadores diferentes para o mapa de calor.")

    titulo_info("Posição do aluno frente à base",
                "Caixas = faixa típica (quartis) de cada indicador na base; losango vermelho = "
                "este aluno.")
    st.plotly_chart(boxplot_base(linha), width='stretch')

    with st.expander("📋 Ficha do modelo"):
        st.markdown(f"""
- **Versão:** {core.VERSAO} · **última atualização do modelo:** {_dt}
- **Probabilidade calibrada (este aluno):** {proba:.3f}
- **Limiar de triagem:** {THRESHOLD:.2f} (recall ≥ 0,85)
- **Modelo:** RandomForest calibrado (sigmoid) · alvo: `{core.ALVO}`
- **Treino:** transições 2022→2023 e 2023→2024 (1.365 pares aluno-ano)
- **Excluídos:** `IAN` e `INDE` (derivados da defasagem) · `Genero`
  (piorava o modelo: PR-AUC 0,830 → 0,816, além da questão de equidade)
- **Validação:** out-of-time (AUC-PR {METRICAS['AUC-PR']:.3f}) +
  StratifiedGroupKFold por aluno ({AUC_PR_CV:.3f})
""")
    _botao_pdf_tecnico(linha)


def _botao_pdf_tecnico(linha):
    st.divider()
    botao_gerar_e_baixar(
        "📄 Gerar relatório técnico (PDF)",
        lambda: _tecnico_pdf_bytes(linha),
        "relatorio_tecnico_passos_magicos.pdf", key="tecnico")


# ═══ o seletor da lateral troca a PÁGINA inteira ═══
if NEGOCIO:
    render_negocio()
    st.stop()

if TECNICO:
    render_tecnica(st.session_state.get("linha_calc"))
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Predição Individual", "📂 Predição em Lote", "ℹ️ Sobre"])

# ══════════════════════════ TAB 1 — INDIVIDUAL ══════════════════════════════
with tab1:
    st.subheader("Informe os dados do aluno (ano corrente)")
    with st.popover("📖 Glossário rápido dos indicadores"):
        for k, v in INDICADORES.items():
            st.markdown(f"**{NOME_FEATURE.get(k, k)}** — {v}")
        st.markdown("**Estágio (PEDRA)** — maturidade no programa: Quartzo → Ágata → "
                    "Ametista → Topázio.")
        st.markdown("**Defasagem** — anos de atraso escolar hoje (0 = no nível, "
                    "−1 = um ano atrás, e assim por diante).")

    def _limpar():
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    b1, _ = st.columns([1, 3])
    b1.button("♻️ Limpar", width='stretch', on_click=_limpar)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Acadêmico**")
        ida = st.number_input("Desempenho Acadêmico (IDA)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="ida", help=INDICADORES["IDA"])
        ieg = st.number_input("Engajamento (IEG)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="ieg", help=INDICADORES["IEG"])
        defas = st.number_input("Defasagem atual (anos)", -5, 3, step=1, key="defas",
                                help="0 = no nível esperado · valores negativos = anos de atraso")
        fase = st.number_input("Fase / Nível (0 = Alfa)", 0, 8, step=1, key="fase",
                               help="Nível pedagógico atual do aluno no programa. "
                                    "0 = Alfa (alfabetização) · 1 a 8 = fases seguintes de progressão.")
    with c2:
        st.markdown("**💜 Socioemocional**")
        ips = st.number_input("Psicossocial (IPS)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="ips", help=INDICADORES["IPS"])
        ipp = st.number_input("Psicopedagógico (IPP)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="ipp", help=INDICADORES["IPP"])
        iaa = st.number_input("Autoavaliação (IAA)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="iaa", help=INDICADORES["IAA"])
    with c3:
        st.markdown("**🚀 Protagonismo & Contexto**")
        ipv = st.number_input("Protagonismo (IPV)", 0.0, 10.0, step=0.1,
                              format="%.1f", key="ipv", help=INDICADORES["IPV"])
        pedra = st.selectbox("Estágio (PEDRA)", PEDRAS, key="pedra",
                             help="Estágio de maturidade no programa (classificação PEDRA). "
                                  "Ordem crescente: Quartzo → Ágata → Ametista → Topázio.")
        idade = st.number_input("Idade", 7, 25, step=1, key="idade",
                                help="Idade do aluno em anos completos.")
        tempo = st.number_input("Tempo no programa (anos)", 0, 15, step=1, key="tempo",
                                help="Quantos anos o aluno já está no programa Passos Mágicos. "
                                     "0 = ingressou este ano.")

    if st.button("🔮 Calcular Risco", type="primary", width='stretch'):
        st.session_state["linha_calc"] = {
            "Defasagem": float(defas), "Fase_num": float(fase), "Idade": float(idade),
            "IAA": iaa, "IEG": ieg, "IPS": ips, "IPP": ipp, "IDA": ida, "IPV": ipv,
            "tempo_prog": float(tempo), "Pedra": pedra}
        for f in INDS_6:
            st.session_state[f"sim_{f}"] = float(st.session_state["linha_calc"][f])
        st.session_state["calc"] = True

    if st.session_state.get("calc"):
        linha = st.session_state["linha_calc"]
        proba = calcular_proba(linha)
        cenarios = render_laudo(linha, proba, chave="ind")

        # ── por que este resultado (SHAP) ──
        st.markdown("#### 🔎 Por que este resultado?")
        st.caption("🔴 = está **aumentando** o risco deste aluno · 🟢 = está **diminuindo** "
                   "(ordenado por impacto)")
        try:
            contribs, _ = shap_contribs(linha)
            st.markdown(semaforo(contribs))
        except Exception:
            baixos = [NOME_FEATURE.get(i, i) for i in INDS_6
                      if core.to_num(linha[i]) < MEDIANAS[i]]
            st.markdown(("Pontos de atenção (abaixo da média): " + ", ".join(baixos))
                        if baixos else "Todos os índices na média ou acima. 👏")

        if cenarios and st.button("✨ Aplicar a maior alavanca no simulador"):
            for v in cenarios[0]["vars"]:
                st.session_state[f"sim_{v}"] = 8.0
            st.rerun()
        # ── simulador (visualmente separado do resultado real) ──
        st.divider()
        _borda = "#4AA3E0" if DARK else "#F2994A"
        _fundo = "#2a2416" if DARK else "#fffaf2"
        _texto = "#ffd08a" if DARK else "#7a4b00"
        st.markdown(
            f"<div style='background:{_fundo}; border:2px dashed {_borda}; border-radius:10px; "
            f"padding:12px 16px; margin-bottom:6px;'>"
            f"<div style='font-size:17px; font-weight:700; color:{_texto};'>"
            f"🔧 Cenário simulado — não altera o resultado acima</div>"
            f"<div style='font-size:12.5px; color:{_texto}; opacity:.85; margin-top:4px;'>"
            f"Tudo daqui para baixo é hipotético. O risco real deste aluno continua sendo "
            f"<b>{proba*100:.1f}%</b>.</div></div>", unsafe_allow_html=True)
        with st.expander("Abrir o simulador: e se… — teste o efeito de melhorar os indicadores"):
            st.caption("Ajuste os mesmos indicadores do bloco principal e veja o risco recalcular. "
                       "Estes valores **não** substituem os dados informados do aluno.")
            sim = dict(linha)
            s1, s2, s3 = st.columns(3)
            with s1:
                sim["IDA"] = st.number_input("Desempenho (IDA)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IDA")
                sim["IEG"] = st.number_input("Engajamento (IEG)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IEG")
            with s2:
                sim["IPS"] = st.number_input("Psicossocial (IPS)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IPS")
                sim["IPP"] = st.number_input("Psicopedagógico (IPP)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IPP")
            with s3:
                sim["IPV"] = st.number_input("Protagonismo (IPV)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IPV")
                sim["IAA"] = st.number_input("Autoavaliação (IAA)", 0.0, 10.0, step=0.1,
                                             format="%.1f", key="sim_IAA")
            proba_sim = calcular_proba(sim)
            delta = (proba_sim - proba) * 100
            m1, m2, m3 = st.columns([1, 1, 2])
            m1.metric("Risco REAL (hoje)", f"{proba*100:.1f}%",
                      help="Calculado com os dados que você informou. É este o número do laudo.")
            m2.metric("Risco simulado (hipotético)", f"{proba_sim*100:.1f}%",
                      f"{delta:+.1f} p.p.", delta_color="inverse",
                      help="Cenário 'e se'. Não entra no laudo nem substitui o risco real.")
            m3.caption("🟢 baixou / 🔴 subiu em relação ao risco real. A defasagem e a fase "
                       "permanecem as de hoje — o simulador mostra o efeito de trabalhar os "
                       "indicadores pedagógicos.")

        st.divider()
        botao_laudo_pdf(linha, proba, chave="ind")

# ══════════════════════════ TAB 2 — LOTE ════════════════════════════════════
with tab2:
    st.subheader("Predição em lote via CSV")
    st.markdown("Colunas necessárias: `" + "`, `".join(FEATURES) + "`. "
                "Aceita separador `;` ou `,` e decimal `.` ou `,`.")
    up = st.file_uploader("Arquivo CSV", type=["csv"])
    if up is None:
        st.session_state["lote_out"] = None          # limpa resultado ao remover o arquivo
    else:
        # arquivo novo -> descarta o resultado anterior (evita tabela velha na tela)
        _assin = f"{up.name}:{up.size}"
        if st.session_state.get("lote_assinatura") != _assin:
            st.session_state["lote_assinatura"] = _assin
            st.session_state["lote_out"] = None
        try:
            raw = up.getvalue().decode("utf-8-sig")
            sep = ";" if raw.count(";") > raw.count(",") else ","
            df = pd.read_csv(io.StringIO(raw), sep=sep)
        except Exception as e:
            st.error(f"Não foi possível ler o CSV: {e}")
            df = None

        if df is not None:
            st.markdown("**Prévia dos dados carregados**")
            st.dataframe(df.head(), width='stretch')
            validos, problemas, rejeitados = core.validar_lote(df)
            if validos is None:
                for p in problemas:
                    st.error(p)
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Linhas no arquivo", len(df))
                c2.metric("✅ Válidas", len(validos))
                c3.metric("⚠️ Descartadas", len(df) - len(validos))
                for p in problemas:
                    st.warning(p)

                # Detalhe linha a linha do que falhou — para corrigir sem tentativa e erro
                if rejeitados is not None and len(rejeitados):
                    with st.expander(f"🔍 Ver as {len(rejeitados)} linha(s) descartada(s) e o motivo",
                                     expanded=True):
                        st.caption("`_linha` é o número da linha no arquivo original "
                                   "(contando o cabeçalho).")
                        st.dataframe(rejeitados, width='stretch', hide_index=True)
                        st.download_button("⬇️ Baixar linhas descartadas (CSV)",
                                           rejeitados.to_csv(index=False).encode("utf-8-sig"),
                                           "linhas_descartadas.csv", "text/csv",
                                           key="btn_pdf_baixar_csv_desc")

                if len(validos) and st.button("🔮 Classificar alunos válidos", type="primary"):
                    st.session_state["lote_out"] = _classificar_lote(validos)

                # Renderiza fora do if do botão: clicar numa linha dispara rerun, e sem isso
                # o resultado sumiria da tela.
                if st.session_state.get("lote_out") is not None:
                    out = st.session_state["lote_out"]

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🔴 Alto", int((out["prob_risco"] >= core.FAIXA_ALTO).sum()))
                    m2.metric("🟡 Moderado", int(((out["prob_risco"] >= core.FAIXA_MOD) &
                                                  (out["prob_risco"] < core.FAIXA_ALTO)).sum()))
                    m3.metric("🟢 Baixo", int((out["prob_risco"] < core.FAIXA_MOD).sum()))
                    m4.metric(f"🔎 Triagem (≥{THRESHOLD*100:.0f}%)",
                              int((out["prob_risco"] >= THRESHOLD).sum()))

                    titulo_info("Distribuição da probabilidade de risco",
                                "Quantos alunos do arquivo caem em cada nível de risco. "
                                "Concentração à direita indica muitos alunos em risco alto.")
                    _h = go.Figure(go.Histogram(x=out["prob_risco"], nbinsx=30,
                                                marker_color=_AZUL))
                    _h.update_layout(height=300, margin=dict(t=20),
                                     xaxis_title="prob_risco", yaxis_title="alunos")
                    st.plotly_chart(_tema(_h), width='stretch')

                    st.markdown("### 👇 Escolha um aluno para ver o laudo pedagógico")
                    st.caption("Clique na linha da tabela **ou** use a lista abaixo dela.")
                    out_show = out.sort_values("prob_risco", ascending=False).reset_index(drop=True)
                    evento = st.dataframe(
                        out_show, width='stretch', hide_index=True,
                        on_select="rerun", selection_mode="single-row",
                        key="tab_lote",
                        column_config={"prob_risco": st.column_config.ProgressColumn(
                            "prob_risco", min_value=0.0, max_value=1.0)})

                    sel = evento.selection.rows if hasattr(evento, "selection") else []

                    # Seletor explícito: funciona sempre, independentemente de onde o
                    # Streamlit desenha a caixinha de seleção da tabela.
                    NENHUM = "— nenhum aluno selecionado —"
                    opcoes = [NENHUM] + [
                        f"Linha {i+1} · risco {r['prob_risco']*100:.0f}% · {r['faixa']} · "
                        f"defasagem {int(r['Defasagem'])} ano(s)"
                        for i, r in out_show.iterrows()]
                    escolha = st.selectbox("Aluno para o laudo", opcoes, key="sel_lote")

                    idx = None
                    if sel:
                        idx = sel[0]                      # clique na tabela tem prioridade
                    elif escolha != NENHUM:
                        idx = opcoes.index(escolha) - 1

                    if idx is not None:
                        aluno = out_show.iloc[idx]
                        linha_sel = {f: aluno[f] for f in FEATURES}
                        p_sel = float(aluno["prob_risco"])
                        st.divider()
                        st.markdown(f"## 📋 Laudo pedagógico — aluno da linha {idx + 1}")
                        render_laudo(linha_sel, p_sel, chave=f"lote{idx}")
                        st.divider()
                        botao_laudo_pdf(linha_sel, p_sel, chave=f"lote{idx}")
                        st.divider()
                    else:
                        st.info("ℹ️ Nenhum aluno selecionado. Clique numa linha da tabela ou "
                                "escolha na lista acima para gerar o laudo pedagógico completo "
                                "— com narrativa, gráficos, alavancas, atividades e PDF.")
                    st.download_button("⬇️ Resultado detalhado (CSV)",
                                       out.to_csv(index=False).encode("utf-8-sig"),
                                       "resultado_lote.csv", "text/csv",
                                       key="btn_pdf_baixar_csv_lote")

# ══════════════════════════ TAB 3 — SOBRE ═══════════════════════════════════
with tab3:
    st.subheader("O que esta ferramenta faz")
    st.markdown("""
Ajuda a equipe a **identificar cedo** quais alunos têm maior chance de estar em **defasagem
no próximo ano**, para que o apoio chegue antes. Usa os indicadores de **hoje** para estimar
o risco no **ano seguinte**.

**Como ler:** 🟢 baixo risco · 🟡 acompanhar de perto · 🔴 atenção prioritária.
A decisão é sempre da equipe pedagógica.
""")
    st.markdown("### ⚖️ Limitações e uso ético")
    st.markdown("""
- Estimativa de **apoio**, não um veredito.
- Use para **direcionar apoio**, nunca para rotular ou excluir.
- As análises de alavancas e do simulador são de **sensibilidade do modelo** (associação),
  não promessas causais.
- 🔒 O app **não armazena** nenhum dado inserido.
- O modelo **não usa gênero** — além da questão de equidade, incluí-lo piorava o desempenho.
""")
    st.markdown("### 📖 Indicadores")
    for k, v in INDICADORES.items():
        st.markdown(f"- **{NOME_FEATURE.get(k, k)}** — {v}")
    st.markdown("**Estágio (PEDRA):** Quartzo → Ágata → Ametista → Topázio.")

# ══════════════════════════ RODAPÉ ═══════════════════════════════════════════
_footer_img = (f"<img src='data:image/png;base64,{_fiap}' "
               f"style='height:26px;vertical-align:middle;margin-left:8px;'>") if _fiap else ""
st.markdown(
    f"""
    <div style='margin-top:32px;padding:14px 20px;border-top:1px solid #2b3a4b;
                text-align:center;font-size:13px;opacity:0.8;'>
      Projeto Datathon FIAP PosTech · Fase 5{_footer_img}
    </div>
    """, unsafe_allow_html=True)
