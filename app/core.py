"""Lógica pura do app (sem Streamlit) — testável isoladamente.

RandomForest calibrado. Alvo = Defasagem < 0 no ano SEGUINTE (T+1).
Base PEDE 2022/2023/2024 · 1.365 transições aluno-ano.

Nota de desenho: IAN e INDE foram EXCLUÍDOS das features por serem derivados da
Defasagem (leakage). Genero foi excluído por piorar o modelo e levantar questão de
equidade.
"""
import os

import joblib
import numpy as np
import pandas as pd

_AQUI = os.path.dirname(os.path.abspath(__file__))
_MODELO = os.path.join(_AQUI, "..", "models", "model.joblib")

_bundle = joblib.load(_MODELO)
MODEL = _bundle["model"]
FEATURES = _bundle["features"]
NUM_FEATURES = _bundle["num"]
CAT_FEATURES = _bundle["cat"]
THRESHOLD = _bundle["threshold"]
METRICAS = _bundle["metricas_teste"]
AUC_PR_CV = _bundle["auc_pr_groupkfold"]
PEDRAS = _bundle["pedras"]
ALVO = _bundle["target"]
DATA_TREINO = _bundle.get("data_treino", "não informada")
VERSAO = _bundle.get("versao", "—")

# Os 6 índices pedagógicos que a equipe pode influenciar
INDS_6 = ["IDA", "IEG", "IPV", "IAA", "IPS", "IPP"]

INDICADORES = {
    "IDA": "Desempenho Acadêmico — notas e domínio dos conteúdos.",
    "IEG": "Engajamento — participação e entrega nas atividades.",
    "IPV": "Protagonismo — autonomia e iniciativa do aluno.",
    "IAA": "Autoavaliação — como o próprio aluno avalia seu desempenho.",
    "IPS": "Psicossocial — relações, convivência e suporte social.",
    "IPP": "Psicopedagógico — processo de aprendizagem e suas dificuldades.",
}

NOME_FEATURE = {
    "Defasagem": "Defasagem atual (anos)",
    "Fase_num": "Fase / Nível",
    "Idade": "Idade",
    "tempo_prog": "Tempo no programa (anos)",
    "Pedra": "Estágio (PEDRA)",
    "IDA": "Desempenho Acadêmico (IDA)",
    "IEG": "Engajamento (IEG)",
    "IPV": "Protagonismo (IPV)",
    "IAA": "Autoavaliação (IAA)",
    "IPS": "Psicossocial (IPS)",
    "IPP": "Psicopedagógico (IPP)",
}

NOME_CURTO = {
    "IDA": "Desempenho", "IEG": "Engajamento", "IPV": "Protagonismo",
    "IAA": "Autoaval.", "IPS": "Psicossocial", "IPP": "Psicoped.",
    "Defasagem": "Defasagem", "Fase_num": "Fase", "Idade": "Idade",
    "tempo_prog": "Tempo prog.", "Pedra": "Estágio",
}

# ── Agregados reais da base (embutidos; sem dados individuais de alunos) ──────
MEDIANAS = {"IDA": 6.9, "IEG": 8.9, "IPV": 7.83, "IAA": 8.8, "IPS": 6.9, "IPP": 7.71}

DIST_BASE = {
    "IDA": {"min": 1.0, "q1": 5.7, "med": 6.9, "q3": 7.9, "max": 10.0},
    "IEG": {"min": 2.0, "q1": 8.1, "med": 8.9, "q3": 9.5, "max": 10.0},
    "IPV": {"min": 3.3, "q1": 7.25, "med": 7.83, "q3": 8.45, "max": 10.0},
    "IAA": {"min": 0.0, "q1": 7.8, "med": 8.8, "q3": 9.5, "max": 10.0},
    "IPS": {"min": 2.5, "q1": 5.0, "med": 6.9, "q3": 7.5, "max": 10.0},
    "IPP": {"min": 4.4, "q1": 7.19, "med": 7.71, "q3": 8.28, "max": 9.8},
}

# Faixa de risco prevista × taxa real observada (teste out-of-time 2023→2024).
# Vem do bundle: assim não pode divergir do modelo treinado.
FAIXA_REALIDADE = _bundle.get("faixa_realidade", [])

# Importância por permutação (Δ AUC no teste out-of-time)
IMPORTANCIA = _bundle.get("importancia", {})

# Cortes recalibrados por percentil real do modelo (não mais números redondos
# arbitrários): 0.36 e 0.67 caem exatamente no "salto" natural da taxa real de
# defasagem observada. Antes, com o corte de 0.70, alunos sem defasagem hoje
# (Defasagem >= 0) nunca conseguiam chegar em Alto Risco mesmo no pior cenário
# possível dos 6 indicadores pedagógicos — o teto real para esse grupo é ~0.68-0.75,
# abaixo do corte antigo.
FAIXA_MOD, FAIXA_ALTO = 0.36, 0.67


# ─────────────────────────── utilidades ──────────────────────────────────────
def to_num(v):
    """Converte para float aceitando vírgula decimal. Devolve np.nan se não der."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(str(v).strip().replace(",", "."))
    except (ValueError, AttributeError):
        return np.nan


def montar_X(linha):
    """dict -> DataFrame de 1 linha na ordem exata das features do modelo."""
    return pd.DataFrame([linha])[FEATURES]


def prever(linha):
    """Probabilidade de defasagem no ano seguinte (0 a 1)."""
    return float(MODEL.predict_proba(montar_X(linha))[0, 1])


def prever_lote(linhas):
    """Probabilidades de vários cenários numa ÚNICA chamada ao modelo.

    O custo do pipeline (imputação, escala, one-hot) é pago por chamada, não por
    linha — agrupar é ~100x mais rápido que prever um a um.
    """
    if not linhas:
        return np.array([])
    return MODEL.predict_proba(pd.DataFrame(list(linhas))[FEATURES])[:, 1]


def classificar(proba):
    """Devolve (rótulo, cor de fundo) para a faixa de risco."""
    if proba >= FAIXA_ALTO:
        return "🔴 Alto Risco", "#fadbd8"
    if proba >= FAIXA_MOD:
        return "🟡 Risco Moderado", "#fdebd0"
    return "🟢 Baixo Risco", "#d5f5e3"


# ─────────────────────────── camada pedagógica ───────────────────────────────
def defasagem_texto(defasagem):
    """Descreve em palavras a defasagem informada (valor negativo = atrasado)."""
    d = to_num(defasagem)
    if np.isnan(d):
        return "sem informação de defasagem"
    if d >= 1:
        return "adiantado em relação ao nível esperado"
    if d == 0:
        return "no nível esperado para a idade"
    if d == -1:
        return "cerca de 1 ano abaixo do nível esperado"
    return f"cerca de {int(abs(d))} anos abaixo do nível esperado"


def comparacao_pares(linha):
    """Compara os 6 índices do aluno com a mediana da base."""
    abaixo = [i for i in INDS_6 if to_num(linha.get(i)) < MEDIANAS[i]]
    acima = [i for i in INDS_6 if to_num(linha.get(i)) >= MEDIANAS[i]]
    return abaixo, acima


def gap_academico_socioemocional(linha):
    """Compara o bloco acadêmico (IDA, IEG) com o socioemocional (IPS, IPP, IAA)."""
    acad = np.mean([to_num(linha["IDA"]), to_num(linha["IEG"])])
    socio = np.mean([to_num(linha["IPS"]), to_num(linha["IPP"]), to_num(linha["IAA"])])
    diff = acad - socio
    if abs(diff) < 1.0:
        return "Blocos acadêmico e socioemocional equilibrados.", "equilibrado", acad, socio
    if diff < 0:
        return ("O bloco acadêmico está mais frágil — priorizar reforço de aprendizagem.",
                "academico", acad, socio)
    return ("O bloco socioemocional está mais frágil — priorizar acolhimento e suporte.",
            "socioemocional", acad, socio)


def risco_evasao(linha):
    """Heurístico: 3+ sinais combinados sugerem atenção à permanência."""
    fatores = []
    if to_num(linha["IEG"]) < 6:
        fatores.append("engajamento baixo")
    if to_num(linha["IPV"]) < 6:
        fatores.append("protagonismo baixo")
    if to_num(linha["IDA"]) < 5:
        fatores.append("desempenho baixo")
    if to_num(linha["IPS"]) < 5:
        fatores.append("suporte psicossocial frágil")
    if to_num(linha["Defasagem"]) <= -2:
        fatores.append("defasagem de 2+ anos")
    return (len(fatores) >= 3), fatores


def perfil_autopercepcao(linha):
    """Compara a autoavaliação (IAA) com o desempenho real (IDA)."""
    d = to_num(linha["IAA"]) - to_num(linha["IDA"])
    if d >= 2.5:
        return "superestima", ("autoavaliação bem acima do desempenho — alinhar percepção "
                               "e realidade com feedback concreto.")
    if d <= -2.5:
        return "subestima", ("autoavaliação abaixo do desempenho real — reforçar conquistas "
                             "para elevar a autoconfiança.")
    return None, ""


# ─────────────────────────── atividades ──────────────────────────────────────
BANCO_ATIVIDADES = {
    "IEG": ("Engajamento (IEG)", [
        "Busca ativa e contato com a família",
        "Projeto ligado a um interesse do aluno",
        "Tutoria entre pares com metas curtas semanais",
        "Rodas de pertencimento e acolhimento no grupo"]),
    "IDA": ("Desempenho Acadêmico (IDA)", [
        "Reforço / tutoria em contraturno",
        "Plano de estudo individualizado",
        "Trilha de recuperação por níveis",
        "Monitoria com aluno mais avançado"]),
    "IPS": ("Psicossocial (IPS)", [
        "Roda de conversa e escuta ativa",
        "Atividades de expressão (arte, teatro, escrita)",
        "Mediação de conflitos",
        "Encaminhamento à equipe psicológica"]),
    "IPP": ("Psicopedagógico (IPP)", [
        "Avaliação psicopedagógica",
        "Material adaptado e estratégias multissensoriais",
        "Acompanhamento individualizado"]),
    "IAA": ("Autoavaliação (IAA)", [
        "Portfólio de conquistas do aluno",
        "Feedback positivo estruturado",
        "Metas alcançáveis e progressivas",
        "Mentoria individual"]),
    "IPV": ("Protagonismo (IPV)", [
        "Projeto de protagonismo ou liderança",
        "Voluntariado ou apadrinhamento de colega",
        "Objetivo pessoal com acompanhamento"]),
    "Defasagem": ("Recuperação de defasagem", [
        "Plano de recuperação com metas por etapa",
        "Reforço intensivo nas competências do nível ideal",
        "Acompanhamento próximo da progressão série/idade"]),
}


def atividades_por_areas(areas):
    """[(rótulo, [atividades])] para uma lista de indicadores (ex.: vindos das alavancas)."""
    vistos, out = set(), []
    for a in areas:
        if a in BANCO_ATIVIDADES and a not in vistos:
            vistos.add(a)
            out.append(BANCO_ATIVIDADES[a])
    return out



def encaminhamentos(linha):
    """Encaminhamentos sugeridos com base nos indicadores."""
    out = []
    if to_num(linha["IDA"]) < 6:
        out.append("**Reforço / tutoria acadêmica** — desempenho abaixo do esperado.")
    if to_num(linha["IPP"]) < 6:
        out.append("**Avaliação psicopedagógica** — possível dificuldade de aprendizagem.")
    if to_num(linha["IPS"]) < 6:
        out.append("**Equipe psicossocial** — suporte social ou emocional frágil.")
    if to_num(linha["IEG"]) < 6:
        out.append("**Busca ativa / família** — baixo engajamento.")
    return out


# ─────────────────────────── alavancas ───────────────────────────────────────
ALAVANCAS_CANDIDATAS = ["IEG", "IDA", "IPV", "IPP", "IPS", "IAA"]


def detectar_alavancas(linha, predict_fn=None, alvo=8.0, top=5, max_combo=3):
    """Testa melhorar indicadores baixos (individuais e combinados) e mede o ganho real.

    Monta todos os cenários e prevê em UMA chamada (ver prever_lote).
    `predict_fn` é aceito por compatibilidade, mas o caminho em lote é o usado.
    Retorna (probabilidade base, [cenários ordenados por ganho]).
    """
    from itertools import combinations
    candidatos = [v for v in ALAVANCAS_CANDIDATAS if to_num(linha.get(v, 10)) < alvo]

    combos = []
    for k in range(1, min(max_combo, len(candidatos)) + 1):
        combos.extend(combinations(candidatos, k))

    # base + todos os cenários numa única predição
    cenarios_linhas = [dict(linha)]
    for combo in combos:
        d = dict(linha)
        for v in combo:
            d[v] = alvo
        cenarios_linhas.append(d)

    probas = prever_lote(cenarios_linhas)
    base_p = float(probas[0])

    cenarios = [{"vars": list(combo), "novo_risco": float(p), "ganho": (base_p - float(p)) * 100}
                for combo, p in zip(combos, probas[1:])]
    cenarios.sort(key=lambda c: (-c["ganho"], len(c["vars"])))
    return base_p, cenarios[:top]


def narrativa_pedagogica(linha, proba):
    """Laudo em linguagem natural. Devolve (abertura, [parágrafos]).

    Usado tanto na tela quanto no PDF, para que contem a mesma história.
    """
    faixa = classificar(proba)[0]
    faixa_txt = (faixa.replace("🔴", "").replace("🟡", "").replace("🟢", "").strip().lower())
    abertura = (f"Com base nos indicadores informados, a estimativa é de "
                f"**{proba*100:.0f}%** de chance de o aluno estar em defasagem no "
                f"**próximo ano**, o que o coloca na faixa de **{faixa_txt}**. ")
    if proba >= THRESHOLD:
        abertura += ("Por esse motivo, sugerimos incluí-lo na lista de acompanhamento "
                     "prioritário, para que o apoio chegue de forma preventiva.")
    else:
        abertura += ("No momento ele não figura entre os casos prioritários, mas vale manter "
                     "o acompanhamento regular para preservar essa trajetória.")

    paragrafos = []

    # 1) situação atual + equilíbrio dos blocos
    partes = []
    d = to_num(linha["Defasagem"])
    txt = defasagem_texto(d)
    if d >= 0:
        partes.append(f"hoje o aluno está {txt}, o que é um bom ponto de partida")
    else:
        partes.append(f"atualmente o aluno está {txt}, situação que pede atenção")
    _, gap_tipo, _, _ = gap_academico_socioemocional(linha)
    if gap_tipo == "academico":
        partes.append("observa-se que o lado acadêmico está mais frágil e merece prioridade "
                      "no apoio")
    elif gap_tipo == "socioemocional":
        partes.append("observa-se que o lado socioemocional está mais frágil e merece "
                      "prioridade no apoio")
    else:
        partes.append("os aspectos acadêmicos e socioemocionais aparecem equilibrados")
    paragrafos.append(". ".join(s[0].upper() + s[1:] for s in partes) + ".")

    # 2) comparação com a base
    abaixo, acima = comparacao_pares(linha)
    if abaixo and len(abaixo) < len(INDS_6):
        nomes_ab = ", ".join(NOME_FEATURE.get(i, i).split(" (")[0] for i in abaixo)
        nomes_ac = ", ".join(NOME_FEATURE.get(i, i).split(" (")[0] for i in acima)
        paragrafos.append(f"Frente à média dos alunos da base, ele aparece abaixo em "
                          f"{nomes_ab} — e igual ou acima em {nomes_ac}.")
    elif abaixo:
        paragrafos.append("Frente à média dos alunos da base, todos os índices estão abaixo do "
                          "usual, o que reforça a necessidade de um plano abrangente.")
    else:
        paragrafos.append("Frente à média dos alunos da base, todos os índices estão na média "
                          "ou acima — um perfil sólido.")

    # 3) alertas
    seg = []
    alerta, fatores = risco_evasao(linha)
    if alerta:
        seg.append("alguns sinais combinados — " + ", ".join(fatores) + " — sugerem atenção à "
                   "permanência do aluno; um acompanhamento próximo da família pode fazer "
                   "diferença")
    tipo, _ = perfil_autopercepcao(linha)
    if tipo == "superestima":
        seg.append("vale notar que ele tende a avaliar o próprio desempenho acima do observado, "
                   "o que abre espaço para um alinhamento gentil entre percepção e realidade")
    elif tipo == "subestima":
        seg.append("chama atenção que ele se avalia abaixo do que os indicadores mostram — "
                   "reforçar suas conquistas pode ajudar na autoconfiança")
    if seg:
        paragrafos.append(". ".join(s[0].upper() + s[1:] for s in seg) + ".")

    return abertura, paragrafos


def atividades_para(linha, cenarios):
    """Áreas a trabalhar = as apontadas pelas alavancas ∪ as frágeis do aluno."""
    areas = []
    for c in cenarios or []:
        for v in c["vars"]:
            if v not in areas:
                areas.append(v)
    for i in INDS_6:
        if to_num(linha.get(i, 10)) < 6 and i not in areas:
            areas.append(i)
    if to_num(linha.get("Defasagem", 0)) <= -1:
        areas.append("Defasagem")
    return atividades_por_areas(areas)


# ─────────────────────────── lote ────────────────────────────────────────────
COLS_LOTE = FEATURES


def validar_lote(df):
    """Valida o CSV do lote.

    Devolve (df_valido, lista_de_problemas, df_rejeitados).
    - `df_rejeitados` traz a coluna `_linha` (número da linha no arquivo, base 1, já
      considerando o cabeçalho) e `_motivo` explicando por que caiu fora.
    """
    problemas = []
    faltantes = [c for c in COLS_LOTE if c not in df.columns]
    if faltantes:
        problemas.append(f"Colunas ausentes: {', '.join(faltantes)}")
        return None, problemas, None

    d = df.copy()
    for c in NUM_FEATURES:
        d[c] = d[c].apply(to_num)

    motivos = pd.Series([[] for _ in range(len(d))], index=d.index)

    for c in INDS_6:
        fora = ((d[c] < 0) | (d[c] > 10)).fillna(False)
        if fora.any():
            problemas.append(f"{int(fora.sum())} linha(s) com {c} fora de 0–10")
        for i in d.index[fora]:
            motivos[i] = motivos[i] + [f"{c} fora de 0–10 (valor: {df.loc[i, c]})"]

    ped_ruim = ~d["Pedra"].isin(PEDRAS)
    if ped_ruim.any():
        problemas.append(f"{int(ped_ruim.sum())} linha(s) com PEDRA inválida "
                         f"(esperado: {', '.join(PEDRAS)})")
    for i in d.index[ped_ruim]:
        motivos[i] = motivos[i] + [f"PEDRA inválida (valor: {df.loc[i, 'Pedra']})"]

    for c in NUM_FEATURES:
        nulo = d[c].isna()
        if nulo.any():
            problemas.append(f"{int(nulo.sum())} linha(s) com {c} ausente ou não numérico")
        for i in d.index[nulo]:
            motivos[i] = motivos[i] + [f"{c} ausente ou não numérico"]

    ok = motivos.apply(len) == 0
    rejeitados = df[~ok].copy()
    if len(rejeitados):
        rejeitados.insert(0, "_motivo", motivos[~ok].apply(lambda m: " · ".join(m)).values)
        rejeitados.insert(0, "_linha", [int(i) + 2 for i in rejeitados.index])  # +2: cabeçalho
    return d[ok].copy(), problemas, rejeitados
