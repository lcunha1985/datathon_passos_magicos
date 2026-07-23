"""Geração de PDFs (laudo pedagógico e relatório técnico).

Gráficos em matplotlib e layout em reportlab — sem kaleido/Chrome, robusto no
Streamlit Community Cloud.
"""
import io
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)

import core

AZUL = "#2D82C4"
AZUL2 = "#1F5C8F"
_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")


def _limpa(txt):
    return (txt or "").replace("**", "").replace("*", "")


def _md(txt):
    """Converte o **negrito** do markdown para a tag <b> do reportlab."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt or "")


def _limpa_faixa(faixa):
    return (_limpa(faixa).replace("🔴", "").replace("🟡", "")
            .replace("🟢", "").strip().lower())


# ─────────────────────────── gráficos ────────────────────────────────────────
def _fig_to_img(fig, w_mm=170, h_mm=70):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_mm * mm, height=h_mm * mm)


def _chart_barras(linha):
    inds = core.INDS_6
    rot = [core.NOME_CURTO[i] for i in inds]
    aluno = [core.to_num(linha[i]) for i in inds]
    media = [core.MEDIANAS[i] for i in inds]
    x = np.arange(len(inds))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(x - 0.2, aluno, 0.4, label="Aluno", color=AZUL)
    ax.bar(x + 0.2, media, 0.4, label="Média da base", color="#AECBE6")
    ax.set_xticks(x)
    ax.set_xticklabels(rot, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8)
    ax.set_title("Aluno × média da base", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 66)


def _chart_calibracao():
    d = core.FAIXA_REALIDADE
    rot = [x["faixa"].replace("\n", " ") for x in d]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    barras = ax.bar(rot, [x["taxa_real"] for x in d],
                    color=["#2ecc71", "#f39c12", "#e74c3c"])
    for b, x in zip(barras, d):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{x['taxa_real']:.1f}%\n(n={x['n']})", ha="center", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("defasagem real no ano seguinte (%)", fontsize=9)
    ax.set_title("Calibração: risco previsto × realidade", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 66)


def _chart_importancia():
    itens = sorted(core.IMPORTANCIA.items(), key=lambda kv: kv[1])
    labels = [core.NOME_FEATURE.get(k, k) for k, _ in itens]
    vals = [v for _, v in itens]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(labels, vals, color=AZUL)
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("Δ AUC (permutation importance)", fontsize=9)
    ax.set_title("Importância das variáveis", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 78)


def _chart_sensibilidade(linha, indicador, predict_fn):
    xs = [round(v, 1) for v in np.arange(0, 10.01, 0.5)]
    ys = []
    for v in xs:
        d = dict(linha)
        d[indicador] = v
        ys.append(predict_fn(d) * 100)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(xs, ys, "-o", color=AZUL, markersize=3)
    ax.axvline(core.to_num(linha[indicador]), ls="--", color="#e74c3c",
               label=f"atual ({core.to_num(linha[indicador]):.1f})")
    ax.axhline(core.THRESHOLD * 100, ls=":", color="#888", label="limiar")
    ax.set_ylim(0, 100)
    ax.set_xlabel(core.NOME_FEATURE.get(indicador, indicador), fontsize=9)
    ax.set_ylabel("risco (%)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title("Sensibilidade do risco", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 62)


# ─────────────────────────── estilos ─────────────────────────────────────────
def _estilos():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("TituloPM", parent=ss["Title"],
                          textColor=colors.HexColor(AZUL2), fontSize=17, spaceAfter=2))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"],
                          textColor=colors.HexColor("#666"), fontSize=9))
    ss.add(ParagraphStyle("H2PM", parent=ss["Heading2"],
                          textColor=colors.HexColor(AZUL), fontSize=12,
                          spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Corpo", parent=ss["Normal"], fontSize=9.5, leading=13))
    ss.add(ParagraphStyle("Rodape", parent=ss["Normal"], fontSize=7.5,
                          textColor=colors.HexColor("#999")))
    return ss


def _cabecalho(story, ss, titulo):
    if os.path.exists(_LOGO):
        try:
            story.append(Image(_LOGO, width=32 * mm, height=16 * mm))
        except Exception:
            pass
    story.append(Paragraph(titulo, ss["TituloPM"]))
    try:
        _dt = datetime.fromisoformat(core.DATA_TREINO).strftime("%d/%m/%Y")
    except Exception:
        _dt = core.DATA_TREINO
    story.append(Paragraph(
        "Passos Mágicos · gerado em " + datetime.now().strftime("%d/%m/%Y %H:%M")
        + f" · modelo {core.VERSAO} (última atualização: {_dt})", ss["Sub"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor(AZUL),
                            spaceBefore=6, spaceAfter=8))


def _rodape_etico(story, ss):
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#ddd"),
                            spaceBefore=4, spaceAfter=4))
    story.append(Paragraph(
        "Ferramenta de apoio à decisão pedagógica — não substitui a avaliação da equipe. "
        "Use para direcionar apoio, nunca para rotular ou excluir. Nenhum dado é armazenado "
        "pelo aplicativo. As análises 'e se…' e de alavancas são de sensibilidade do modelo "
        "(associação), não promessas causais. O modelo não utiliza gênero.", ss["Rodape"]))
    story.append(Paragraph(
        "Projeto Datathon FIAP PosTech · Fase 5", ss["Rodape"]))


# ─────────────────────────── LAUDO PEDAGÓGICO ────────────────────────────────
def laudo_pedagogico_pdf(linha, proba, faixa, predict_fn):
    ss = _estilos()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=14 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    story = []
    _cabecalho(story, ss, "Laudo de Risco de Defasagem")

    # Narrativa em linguagem natural — a MESMA usada na tela (core.narrativa_pedagogica)
    abertura, paragrafos = core.narrativa_pedagogica(linha, proba)
    story.append(Paragraph(_md(abertura), ss["Corpo"]))
    story.append(Spacer(1, 4))

    # Dados informados
    story.append(Paragraph("Dados informados", ss["H2PM"]))
    campos = [("IDA", "Desempenho (IDA)"), ("IEG", "Engajamento (IEG)"),
              ("IPV", "Protagonismo (IPV)"), ("IAA", "Autoavaliação (IAA)"),
              ("IPS", "Psicossocial (IPS)"), ("IPP", "Psicopedagógico (IPP)")]
    tab = [["Indicador", "Valor", "Indicador", "Valor"]]
    for i in range(0, len(campos), 2):
        a, b = campos[i], campos[i + 1]
        tab.append([a[1], f"{core.to_num(linha[a[0]]):.1f}",
                    b[1], f"{core.to_num(linha[b[0]]):.1f}"])
    tab.append(["Defasagem atual", f"{int(core.to_num(linha['Defasagem']))} ano(s)",
                "Fase / Nível", f"{int(core.to_num(linha['Fase_num']))}"])
    tab.append(["Idade", f"{int(core.to_num(linha['Idade']))}",
                "Estágio (PEDRA)", str(linha["Pedra"])])
    t = Table(tab, colWidths=[45 * mm, 25 * mm, 45 * mm, 25 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(AZUL2)),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe6f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafd")]),
    ]))
    story.append(t)

    story.append(Paragraph("Leitura pedagógica", ss["H2PM"]))
    for par in paragrafos:
        story.append(Paragraph(_md(par), ss["Corpo"]))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    story.append(_chart_barras(linha))

    # Alavancas + atividades
    try:
        base_p, cen = core.detectar_alavancas(linha, predict_fn)
        if cen:
            story.append(Paragraph("Onde concentrar os esforços", ss["H2PM"]))
            m = cen[0]
            nomes = ", ".join(core.NOME_FEATURE.get(v, v).split(" (")[0] for v in m["vars"])
            story.append(Paragraph(
                f"Simulando melhorias nos indicadores, o maior retorno viria de trabalhar "
                f"<b>{nomes}</b> em conjunto: nesse cenário, a estimativa de risco cairia de "
                f"{base_p*100:.0f}% para cerca de {m['novo_risco']*100:.0f}%. A tabela abaixo "
                f"resume as combinações de maior impacto.", ss["Corpo"]))
            story.append(Spacer(1, 4))
            dados = [["Trabalhar em conjunto", "Risco estimado depois", "Redução (p.p.)"]]
            for c in cen[:4]:
                n = " + ".join(core.NOME_FEATURE.get(v, v).split(" (")[0] for v in c["vars"])
                dados.append([n, f"{c['novo_risco']*100:.0f}%", f"{c['ganho']:.1f}"])
            ta = Table(dados, colWidths=[95 * mm, 35 * mm, 25 * mm])
            ta.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(AZUL2)),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe6f0")),
            ]))
            story.append(ta)

            areas = []
            for c in cen:
                for v in c["vars"]:
                    if v not in areas:
                        areas.append(v)
            ativ = core.atividades_por_areas(areas)
            if ativ:
                story.append(Paragraph("Sugestões de atividades", ss["H2PM"]))
                story.append(Paragraph(
                    "A seguir, algumas frentes de trabalho para as áreas destacadas acima. "
                    "São pontos de partida — a equipe adapta ao contexto de cada aluno.",
                    ss["Corpo"]))
                for rotulo, lista in ativ[:4]:
                    story.append(Paragraph(f"<b>{_limpa(rotulo)}</b>", ss["Corpo"]))
                    for a in lista[:3]:
                        story.append(Paragraph("&nbsp;&nbsp;• " + _limpa(a), ss["Corpo"]))
    except Exception:
        pass

    _rodape_etico(story, ss)
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────── RELATÓRIO TÉCNICO ───────────────────────────────
def relatorio_tecnico_pdf(linha, predict_fn):
    ss = _estilos()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=14 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    story = []
    _cabecalho(story, ss, "Relatório Técnico do Modelo")

    story.append(Paragraph(
        "Este relatório resume como o modelo foi construído e avaliado. Cada seção traz o "
        "resultado e, em seguida, uma leitura em linguagem simples do que ele significa — "
        "se é um bom sinal ou um ponto de atenção.", ss["Corpo"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Cuidado com vazamento de dados (leakage)", ss["H2PM"]))
    story.append(Paragraph(
        "A base traz a coluna <b>Fase Ideal</b>, e verificamos que <b>Defasagem = Fase − Fase "
        "Ideal</b> em <b>100%</b> dos casos, com a Fase Ideal determinada pela Idade. Assim, "
        "os índices IAN e INDE (derivados da defasagem) foram <b>excluídos</b> das variáveis, e "
        "o alvo passou a ser a defasagem do <b>ano seguinte</b> — genuinamente desconhecida no "
        "momento da previsão. Sem esse cuidado, o modelo apenas reproduziria uma subtração já "
        "conhecida, com métricas artificialmente altas.", ss["Corpo"]))

    story.append(Paragraph("Desempenho (teste out-of-time 2023→2024)", ss["H2PM"]))
    m = core.METRICAS
    dados = [["AUC-PR", "AUC-ROC", "Precisão", "Recall", "F1"],
             [f"{m['AUC-PR']:.3f}", f"{m['AUC-ROC']:.3f}", f"{m['Precision_risco']:.3f}",
              f"{m['Recall_risco']:.3f}", f"{m['F1']:.3f}"]]
    tm = Table(dados, colWidths=[30 * mm] * 5)
    tm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(AZUL2)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe6f0")),
    ]))
    story.append(tm)
    story.append(Paragraph(
        f"<b>Como interpretar:</b> um AUC-PR de {m['AUC-PR']:.2f} é <b>bom</b> para este tipo de "
        f"problema. O <b>recall de {m['Recall_risco']:.0%}</b> indica que o modelo captura a "
        f"maior parte dos alunos que virão a ter defasagem — desejável aqui, já que deixar de "
        f"identificar quem precisa de apoio é o erro mais custoso. O limiar "
        f"{core.THRESHOLD:.2f} foi escolhido justamente para priorizar esse recall. Robustez "
        f"por aluno (StratifiedGroupKFold): AUC-PR ≈ {core.AUC_PR_CV:.3f}.", ss["Corpo"]))

    story.append(_chart_calibracao())
    story.append(Paragraph(
        "<b>Resultado positivo:</b> a taxa real de defasagem <b>sobe</b> de forma consistente do "
        "grupo de baixo risco para o de alto risco (6,8% · 53,9% · 92,1%). Isso confirma que o "
        "modelo <b>ordena corretamente</b> o risco.", ss["Corpo"]))

    story.append(_chart_importancia())
    story.append(Paragraph(
        "<b>Como ler:</b> mostra quanto o desempenho cai ao embaralhar cada variável — ou seja, o "
        "<b>peso</b> dela na decisão, não se é boa ou ruim. A <b>defasagem atual</b> e a "
        "<b>fase</b> dominam, o que é esperado: a defasagem é persistente. Entre os índices "
        "pedagógicos, IPP e IPV lideram.", ss["Corpo"]))

    if linha is not None:
        story.append(Paragraph("Sensibilidade (último aluno calculado)", ss["H2PM"]))
        story.append(_chart_sensibilidade(linha, "IPV", predict_fn))
        story.append(Paragraph(
            "<b>Como ler:</b> a linha mostra como o risco previsto mudaria se este indicador "
            "melhorasse, mantendo o resto igual. Uma curva que <b>desce</b> indica que melhorar "
            "aquele indicador tende a <b>reduzir</b> o risco.", ss["Corpo"]))

    story.append(Paragraph("Metodologia", ss["H2PM"]))
    story.append(Paragraph(
        "RandomForest calibrado (sigmoid). Alvo: defasagem &lt; 0 no ano seguinte (T+1). Treino "
        "com as transições 2022→2023 e 2023→2024 da base PEDE (1.365 pares aluno-ano). "
        "Validação out-of-time (treino 2022→2023, teste 2023→2024) e StratifiedGroupKFold por "
        "aluno. Variáveis excluídas: IAN e INDE (derivados da defasagem) e Genero — este último "
        "piorava o desempenho (PR-AUC 0,830 → 0,816) além de levantar questão de equidade.",
        ss["Corpo"]))

    _rodape_etico(story, ss)
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
# ═══════════════════════════════════════════════════════════════════════════
# COLE ISTO NO FINAL DE app/pdf_report.py (mesma pasta do core_negocio.py)
# Reaproveita _estilos(), _cabecalho(), _rodape_etico(), _fig_to_img(), _md()
# já definidos no início do arquivo — por isso não precisa de imports extras
# além do import core_negocio abaixo.
# ═══════════════════════════════════════════════════════════════════════════

import core_negocio as neg

_VERDE_PDF, _LARANJA_PDF, _VERMELHO_PDF = "#2ecc71", "#f39c12", "#e74c3c"
_AZUIS_PEDRA = {"Quartzo": "#AECBE6", "Agata": "#6FAADB",
                "Ametista": AZUL, "Topázio": AZUL2}


def _chart_barras_ano(dados_por_categoria, anos, titulo, ylabel, cores=None):
    """Gráfico de linhas simples (uma linha por categoria, x = ano). Usado nas
    perguntas com evolução temporal (IAN, IDA, IAA/IDA, IPV, INDE).
    `dados_por_categoria` é {nome: {"2020": valor, "2021": valor, ...}} —
    chaves de ano sempre como string (é como vem de core_negocio.AGREGADOS,
    por causa do round-trip JSON)."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for nome, serie in dados_por_categoria.items():
        cor = (cores or {}).get(nome, AZUL)
        ax.plot([str(a) for a in anos], [serie[str(a)] for a in anos], "-o",
                label=nome, color=cor, linewidth=2, markersize=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=7.5, ncol=min(4, len(dados_por_categoria)))
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 62)


def _chart_barras_simples(labels, valores, titulo, ylabel, cores=None):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(labels, valores, color=cores or AZUL)
    for i, v in enumerate(valores):
        ax.text(i, v + (max(valores) * 0.02 if max(valores) else 0.2), f"{v}",
                ha="center", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_img(fig, 170, 62)


# ─────────────────────────── RELATÓRIO DE NEGÓCIO (10 perguntas) ─────────────
def relatorio_negocio_pdf():
    """PDF com as 10 perguntas de negócio respondidas em linguagem simples,
    para quem não tem background técnico/estatístico."""
    ag = neg.AGREGADOS
    anos = neg.ANOS_BASE
    ss = _estilos()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=14 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    story = []
    _cabecalho(story, ss, "Análise de Negócio — 10 Perguntas sobre os Alunos")
    story.append(Paragraph(
        "Este relatório responde, em linguagem simples, dez perguntas sobre como os alunos "
        "da Passos Mágicos vêm evoluindo — usando os dados de 2020, 2021 e 2022. Sempre que "
        "um resultado não confirma o que se esperava, isso é dito abertamente: um relatório "
        "só é útil se for honesto sobre o que os números realmente mostram, mesmo quando não "
        "é a resposta que gostaríamos de ver.", ss["Corpo"]))
    story.append(Spacer(1, 6))

    # 1 ─ IAN
    story.append(Paragraph("1. Os alunos estão no nível esperado para a idade?", ss["H2PM"]))
    q1 = ag["q1_faixa_ian_ano"]
    story.append(Paragraph(_md(
        "Cada aluno é classificado em uma de três situações: **sem defasagem** (está no "
        "nível esperado), **defasagem moderada** ou **defasagem severa** (está bem atrás do "
        f"esperado para a idade). Em 2022, **{ag['q1_severa_moderada_2022_pct']}%** dos "
        "alunos estavam com algum grau de defasagem — e essa proporção vem **crescendo** ano "
        f"após ano: em 2020 era {q1['Moderada']['2020']+q1['Severa']['2020']:.0f}%, em 2022 "
        f"já é {ag['q1_severa_moderada_2022_pct']:.0f}%."), ss["Corpo"]))
    story.append(_chart_barras_ano(
        {"Sem defasagem": q1["Sem defasagem"], "Moderada": q1["Moderada"], "Severa": q1["Severa"]},
        anos, "% de alunos por situação de defasagem", "% de alunos",
        cores={"Sem defasagem": _VERDE_PDF, "Moderada": _LARANJA_PDF, "Severa": _VERMELHO_PDF}))
    story.append(Spacer(1, 4))

    # 2 ─ IDA
    story.append(Paragraph("2. As notas dos alunos estão melhorando?", ss["H2PM"]))
    q2m = ag["q2_ida_medio_ano"]
    story.append(Paragraph(_md(
        f"A nota média (IDA, de 0 a 10) foi de {q2m['2020']} em 2020, caiu para {q2m['2021']} "
        f"em 2021 e subiu de novo para {q2m['2022']} em 2022 — mas sem recuperar totalmente o "
        "nível de 2020. Essa queda em 2021 aconteceu em **todas** as fases do programa, o que "
        "sugere que foi algo que afetou todo mundo naquele ano (possivelmente o período de "
        "aulas remotas), e não um problema de uma turma específica."), ss["Corpo"]))
    story.append(_chart_barras_ano(ag["q2_ida_pedra_ano"], anos,
                                   "Nota média (IDA) por fase do programa", "Nota média (0–10)",
                                   cores=_AZUIS_PEDRA))
    story.append(Spacer(1, 4))

    # 3 ─ IEG
    story.append(Paragraph("3. Quem participa mais, tem notas melhores?", ss["H2PM"]))
    story.append(Paragraph(_md(
        "Sim, existe uma relação: alunos mais engajados (que participam mais das atividades) "
        "tendem a ter notas melhores e também mais chance de viver um \"momento de virada\" "
        "na trajetória escolar. A relação é moderada, não é uma garantia matemática — mas é "
        "consistente o suficiente para dizer que **investir em engajamento tende a ajudar** "
        "tanto o desempenho quanto o protagonismo do aluno. Vale notar que essa relação é mais "
        "forte quando comparamos alunos de fases diferentes do que dentro da mesma fase — ou "
        "seja, o engajamento ajuda mais a avançar de fase do que a subir a nota dentro da fase "
        "atual."), ss["Corpo"]))
    story.append(Spacer(1, 4))

    # 4 ─ IAA
    story.append(Paragraph("4. Os alunos sabem avaliar o próprio desempenho?", ss["H2PM"]))
    story.append(Paragraph(_md(
        f"Não muito bem. Mais da metade dos alunos ({ag['q4_pct_superestimadores']}%) se "
        "avalia bem melhor do que a nota real mostra — ou seja, acham que estão indo melhor "
        "do que realmente estão. **Isso se repete todos os anos**, sem melhora nem piora "
        "clara: o aluno costuma ter uma visão mais otimista de si mesmo do que o desempenho "
        "comprova, e esse padrão não mudou entre 2020 e 2022."), ss["Corpo"]))
    story.append(_chart_barras_ano(
        {"Autoavaliação (IAA)": ag["q4_iaa_medio_por_ano"], "Nota real (IDA)": ag["q4_ida_medio_por_ano"]},
        anos, "Autoavaliação × nota real, ao longo dos anos", "Média (0–10)",
        cores={"Autoavaliação (IAA)": _LARANJA_PDF, "Nota real (IDA)": AZUL}))
    story.append(Spacer(1, 4))

    # 5 ─ IPS
    story.append(Paragraph("5. Problemas emocionais/sociais avisam antes de uma queda de nota?",
                           ss["H2PM"]))
    p5 = ag["q5_por_par_ano"]
    story.append(Paragraph(_md(
        "Testamos se alunos com dificuldades emocionais/sociais em um ano tendem a cair de "
        "nota no ano seguinte, olhando separadamente para os dois períodos disponíveis. "
        "**Em 2020→2021 e também em 2021→2022, a resposta foi não** — não encontramos "
        "diferença relevante entre quem caiu de nota e quem não caiu, em nenhum dos dois "
        "períodos. Isso não quer dizer que o lado emocional não importa — só que, sozinho, "
        "ele não serviu como um aviso antecipado de queda de nota nesta base."), ss["Corpo"]))
    story.append(_chart_barras_simples(
        ["2020→2021\ncaiu", "2020→2021\nnão caiu", "2021→2022\ncaiu", "2021→2022\nnão caiu"],
        [p5["2020->2021"]["ips_medio_caiu"], p5["2020->2021"]["ips_medio_nao_caiu"],
         p5["2021->2022"]["ips_medio_caiu"], p5["2021->2022"]["ips_medio_nao_caiu"]],
        "Nível emocional/social médio, por período e grupo", "Média (0–10)",
        cores=[_VERMELHO_PDF, _VERDE_PDF, _VERMELHO_PDF, _VERDE_PDF]))
    story.append(Spacer(1, 4))

    # 6 ─ IPP
    story.append(Paragraph("6. A avaliação pedagógica confirma o nível medido do aluno?",
                           ss["H2PM"]))
    story.append(Paragraph(_md(
        "Em geral sim, mas a concordância é fraca. Cerca de "
        f"**{ag['q6_pct_divergencia']}%** dos alunos são um caso de atenção: o indicador de "
        "nível diz que estão bem, mas a avaliação pedagógica aponta dificuldades. Esse grupo "
        "vale acompanhamento extra, porque é exatamente onde um problema pode estar "
        "escondido atrás de um número que parece bom."), ss["Corpo"]))
    story.append(Spacer(1, 4))

    # 7 ─ IPV
    story.append(Paragraph("7. O que mais ajuda um aluno a viver seu \"momento de virada\"?",
                           ss["H2PM"]))
    coef_ano = ag["q7_coef_por_ano"]
    story.append(Paragraph(_md(
        "Isso **mudou de ano para ano**: em 2020, o fator mais associado foi a avaliação "
        "psicopedagógica; em 2021, foi o engajamento; em 2022, foi o desempenho acadêmico. "
        "Não existe um único fator que se destaque de forma constante nos três anos — o que "
        "sugere que apoiar o aluno de forma ampla (não só num único aspecto) é mais eficaz do "
        "que apostar sempre na mesma alavanca."), ss["Corpo"]))
    story.append(_chart_barras_ano(
        {ind: {str(a): coef_ano[str(a)][ind] for a in anos if str(a) in coef_ano} for ind in ["IDA","IEG","IPS","IPP"]},
        [a for a in anos if str(a) in coef_ano], "Peso de cada fator no \"momento de virada\", por ano",
        "Peso (quanto maior, mais influência)",
        cores={"IDA": AZUL, "IEG": _VERDE_PDF, "IPS": _LARANJA_PDF, "IPP": "#AECBE6"}))
    story.append(Spacer(1, 4))

    # 8 ─ INDE
    story.append(Paragraph("8. O que mais pesa na nota final do aluno?", ss["H2PM"]))
    story.append(Paragraph(_md(
        f"A combinação de **nota (IDA) + engajamento (IEG) alta** é o que mais eleva a nota "
        f"final do aluno (INDE) — juntos, os quatro indicadores analisados explicam "
        f"**{ag['q8_r2']*100:.0f}%** da nota final. Os aspectos emocional/social e "
        "psicopedagógico pesam bem menos: funcionam mais como suporte do que como uma "
        "alavanca direta de nota."), ss["Corpo"]))
    story.append(Spacer(1, 4))

    # 9 ─ ML
    story.append(Paragraph("9. É possível prever, com antecedência, quem vai ficar defasado?",
                           ss["H2PM"]))
    story.append(Paragraph(_md(
        f"Sim — o app já usa um modelo de inteligência artificial (versão {core.VERSAO}) que "
        "estima essa chance para cada aluno. Ele separa bem quem está em maior e menor risco: "
        f"alunos marcados como **Alto Risco** realmente ficam defasados em "
        f"{core.FAIXA_REALIDADE[2]['taxa_real']:.0f}% dos casos, contra apenas "
        f"{core.FAIXA_REALIDADE[0]['taxa_real']:.0f}% entre os marcados como **Baixo Risco**. "
        "Isso permite direcionar o acompanhamento pedagógico para quem mais precisa, antes "
        "que a defasagem apareça."), ss["Corpo"]))
    story.append(_chart_calibracao())
    story.append(Spacer(1, 4))

    # 10 ─ Efetividade
    story.append(Paragraph("10. O programa está funcionando ao longo do tempo?", ss["H2PM"]))
    q10m = ag["q10_inde_medio_ano"]
    story.append(Paragraph(_md(
        f"A nota final média (INDE) foi de {q10m['2020']} em 2020, caiu para {q10m['2021']} "
        f"em 2021 e recuperou parcialmente para {q10m['2022']} em 2022 — **os dados não "
        "confirmam uma melhora consistente** nesse período. O ponto positivo: a ordem entre "
        "as fases do programa (quanto mais avançada a fase, maior a nota) se mantém estável "
        "todos os anos, o que confirma que o desenho das fases faz sentido. A queda "
        "generalizada em 2021 é o achado que mais chama atenção, e parece ligada a um fator "
        "externo ao programa em si — o mesmo padrão aparece nas notas (pergunta 2)."),
        ss["Corpo"]))
    story.append(_chart_barras_ano(ag["q10_inde_pedra_ano"], anos,
                                   "Nota final (INDE) por fase do programa", "Nota média (0–10)",
                                   cores=_AZUIS_PEDRA))

    _rodape_etico(story, ss)
    doc.build(story)
    buf.seek(0)
    return buf.read()
