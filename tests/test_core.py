"""Testes da lógica pura (core.py) — modelo T->T+1."""
import numpy as np

import core


ALUNO = {"Defasagem": -1.0, "Fase_num": 3.0, "Idade": 13.0, "IAA": 6.0, "IEG": 5.0,
         "IPS": 5.0, "IPP": 5.0, "IDA": 4.0, "IPV": 5.0, "tempo_prog": 3.0,
         "Pedra": "Quartzo"}


# ── contrato do modelo ────────────────────────────────────────────────────────
def test_features_esperadas():
    assert core.FEATURES == ["Defasagem", "Fase_num", "Idade", "IAA", "IEG", "IPS",
                             "IPP", "IDA", "IPV", "tempo_prog", "Pedra"]


def test_ian_e_inde_fora_das_features():
    """IAN e INDE são derivados da Defasagem — não podem entrar (leakage)."""
    assert "IAN" not in core.FEATURES
    assert "INDE" not in core.FEATURES


def test_genero_fora_das_features():
    assert "Genero" not in core.FEATURES


def test_montar_X_respeita_ordem():
    X = core.montar_X(ALUNO)
    assert list(X.columns) == core.FEATURES
    assert len(X) == 1


def test_threshold_em_faixa_valida():
    assert 0.0 < core.THRESHOLD < 1.0


def test_pedras_conhecidas():
    assert set(core.PEDRAS) == {"Agata", "Ametista", "Quartzo", "Topázio"}


# ── previsão / classificação ─────────────────────────────────────────────────
def test_prever_devolve_probabilidade():
    p = core.prever(ALUNO)
    assert 0.0 <= p <= 1.0


def test_classificar_faixas():
    assert core.classificar(0.90)[0].endswith("Alto Risco")
    assert core.classificar(0.50)[0].endswith("Risco Moderado")
    assert core.classificar(0.10)[0].endswith("Baixo Risco")


def test_aluno_forte_tem_risco_menor_que_fraco():
    forte = dict(ALUNO)
    forte.update({"IDA": 9.5, "IEG": 9.5, "IPV": 9.5, "IAA": 9.0, "IPS": 9.0, "IPP": 9.0})
    assert core.prever(forte) < core.prever(ALUNO)


def test_defasagem_maior_aumenta_risco():
    pior = dict(ALUNO); pior["Defasagem"] = -3.0
    melhor = dict(ALUNO); melhor["Defasagem"] = 0.0
    assert core.prever(pior) > core.prever(melhor)


# ── utilidades ───────────────────────────────────────────────────────────────
def test_to_num_aceita_virgula():
    assert core.to_num("7,5") == 7.5
    assert core.to_num("7.5") == 7.5
    assert core.to_num(3) == 3.0


def test_to_num_invalido_vira_nan():
    assert np.isnan(core.to_num("abc"))
    assert np.isnan(core.to_num(None))


# ── camada pedagógica ────────────────────────────────────────────────────────
def test_defasagem_texto():
    assert "no nível" in core.defasagem_texto(0)
    assert "1 ano" in core.defasagem_texto(-1)
    assert "3 anos" in core.defasagem_texto(-3)
    assert "adiantado" in core.defasagem_texto(1)


def test_comparacao_pares():
    abaixo, acima = core.comparacao_pares(ALUNO)
    assert "IDA" in abaixo           # 4.0 < mediana 6.9
    assert set(abaixo) | set(acima) == set(core.INDS_6)


def test_gap_academico_socioemocional():
    msg, tipo, acad, socio = core.gap_academico_socioemocional(ALUNO)
    assert tipo in ("equilibrado", "academico", "socioemocional")
    assert isinstance(msg, str) and msg


def test_risco_evasao_dispara_com_3_fatores():
    alerta, fatores = core.risco_evasao(ALUNO)
    assert alerta is True
    assert len(fatores) >= 3


def test_risco_evasao_nao_dispara_para_aluno_forte():
    forte = dict(ALUNO)
    forte.update({"IDA": 9.0, "IEG": 9.0, "IPV": 9.0, "IPS": 9.0, "Defasagem": 0.0})
    alerta, _ = core.risco_evasao(forte)
    assert alerta is False


def test_perfil_autopercepcao_superestima():
    d = dict(ALUNO); d["IAA"] = 9.5; d["IDA"] = 4.0
    tipo, msg = core.perfil_autopercepcao(d)
    assert tipo == "superestima" and msg


def test_perfil_autopercepcao_neutro():
    d = dict(ALUNO); d["IAA"] = 6.0; d["IDA"] = 6.0
    tipo, _ = core.perfil_autopercepcao(d)
    assert tipo is None


# ── atividades / encaminhamentos ─────────────────────────────────────────────
def test_atividades_por_areas_cobre_alavancas():
    ativ = core.atividades_por_areas(["IPV", "IEG", "IPP"])
    rotulos = [r for r, _ in ativ]
    assert any("IPV" in r for r in rotulos)
    assert any("IEG" in r for r in rotulos)
    assert all(len(lst) >= 2 for _, lst in ativ)




def test_encaminhamentos():
    enc = core.encaminhamentos(ALUNO)
    assert len(enc) >= 1
    assert all(isinstance(e, str) for e in enc)


# ── alavancas ────────────────────────────────────────────────────────────────
def test_detectar_alavancas_ranqueia_por_ganho():
    base, cen = core.detectar_alavancas(ALUNO, core.prever)
    assert len(cen) >= 1
    ganhos = [c["ganho"] for c in cen]
    assert ganhos == sorted(ganhos, reverse=True)


def test_detectar_alavancas_tem_ganho_relevante():
    _, cen = core.detectar_alavancas(ALUNO, core.prever)
    assert cen[0]["ganho"] > 10   # melhora combinada reduz o risco de forma relevante


def test_alavancas_vazias_para_aluno_forte():
    forte = dict(ALUNO)
    for i in core.INDS_6:
        forte[i] = 9.5
    _, cen = core.detectar_alavancas(forte, core.prever)
    assert cen == []


# ── lote ─────────────────────────────────────────────────────────────────────
def test_validar_lote_ok():
    import pandas as pd
    df = pd.DataFrame([ALUNO, ALUNO])
    validos, problemas, rejeitados = core.validar_lote(df)
    assert validos is not None and len(validos) == 2
    assert rejeitados is None or len(rejeitados) == 0


def test_validar_lote_colunas_faltando():
    import pandas as pd
    df = pd.DataFrame([{"IDA": 5.0}])
    validos, problemas, rejeitados = core.validar_lote(df)
    assert validos is None and problemas


def test_validar_lote_descarta_pedra_invalida():
    import pandas as pd
    mau = dict(ALUNO); mau["Pedra"] = "Diamante"
    validos, problemas, rejeitados = core.validar_lote(pd.DataFrame([ALUNO, mau]))
    assert len(validos) == 1
    assert any("PEDRA" in p for p in problemas)


def test_validar_lote_descarta_indice_fora_de_faixa():
    import pandas as pd
    mau = dict(ALUNO); mau["IDA"] = 50.0
    validos, problemas, rejeitados = core.validar_lote(pd.DataFrame([ALUNO, mau]))
    assert len(validos) == 1


def test_validar_lote_detalha_motivo_por_linha():
    """O relatório de rejeitados deve dizer QUAL linha falhou e POR QUÊ."""
    import pandas as pd
    mau = dict(ALUNO); mau["Pedra"] = "Diamante"; mau["IDA"] = 99.0
    validos, problemas, rejeitados = core.validar_lote(pd.DataFrame([ALUNO, mau]))
    assert len(rejeitados) == 1
    assert "_linha" in rejeitados.columns and "_motivo" in rejeitados.columns
    motivo = rejeitados.iloc[0]["_motivo"]
    assert "PEDRA" in motivo and "IDA" in motivo          # acumula os dois problemas
    assert int(rejeitados.iloc[0]["_linha"]) == 3          # índice 1 -> linha 3 do arquivo


def test_validar_lote_detecta_valor_ausente():
    import pandas as pd
    mau = dict(ALUNO); mau["IEG"] = None
    validos, problemas, rejeitados = core.validar_lote(pd.DataFrame([ALUNO, mau]))
    assert len(validos) == 1
    assert "IEG" in rejeitados.iloc[0]["_motivo"]


def test_metadados_do_modelo_presentes():
    """Transparência: o app precisa saber quando o modelo foi treinado."""
    assert core.DATA_TREINO and core.DATA_TREINO != "não informada"
    assert core.VERSAO


# ── métricas embutidas ───────────────────────────────────────────────────────
def test_metricas_carregadas():
    for k in ["AUC-PR", "AUC-ROC", "Recall_risco", "Precision_risco", "F1"]:
        assert 0.0 <= core.METRICAS[k] <= 1.0


def test_faixa_realidade_monotonica():
    """Calibração: a taxa real deve subir com a faixa de risco prevista."""
    taxas = [f["taxa_real"] for f in core.FAIXA_REALIDADE]
    assert taxas == sorted(taxas)


# ── seleção de linha no lote (lógica por trás do clique) ─────────────────────
def test_alavancas_a_partir_de_linha_do_lote():
    """Ao clicar numa linha do lote, o app monta o dict e roda as alavancas."""
    import pandas as pd
    df = pd.DataFrame([ALUNO, ALUNO])
    validos, _, _ = core.validar_lote(df)
    out = validos.copy()
    out["prob_risco"] = core.MODEL.predict_proba(out[core.FEATURES])[:, 1]
    aluno = out.iloc[0]
    linha_sel = {f: aluno[f] for f in core.FEATURES}   # é o que a UI faz na seleção
    base_p, cen = core.detectar_alavancas(linha_sel, core.prever)
    assert 0.0 <= base_p <= 1.0
    assert len(cen) >= 1
    assert all(set(c["vars"]) <= set(core.ALAVANCAS_CANDIDATAS) for c in cen)


def test_aluno_com_defasagem_severa_tem_alavanca_menor():
    """Honestidade do modelo: defasagem de 2+ anos é mais difícil de reverter em 1 ano."""
    leve = dict(ALUNO); leve["Defasagem"] = -1.0
    severo = dict(ALUNO); severo["Defasagem"] = -2.0
    _, cen_leve = core.detectar_alavancas(leve, core.prever)
    _, cen_sev = core.detectar_alavancas(severo, core.prever)
    assert cen_leve[0]["ganho"] > cen_sev[0]["ganho"]


# ── narrativa compartilhada (tela e PDF contam a mesma história) ─────────────
def test_narrativa_pedagogica_estrutura():
    p = core.prever(ALUNO)
    abertura, paragrafos = core.narrativa_pedagogica(ALUNO, p)
    assert "chance de o aluno estar em defasagem" in abertura
    assert f"{p*100:.0f}%" in abertura
    assert len(paragrafos) >= 2
    assert all(par.endswith(".") for par in paragrafos)


def test_narrativa_muda_conforme_o_aluno():
    forte = dict(ALUNO); forte["Defasagem"] = 0.0
    for i in core.INDS_6:
        forte[i] = 9.5
    ab_fraco, _ = core.narrativa_pedagogica(ALUNO, core.prever(ALUNO))
    ab_forte, _ = core.narrativa_pedagogica(forte, core.prever(forte))
    assert ab_fraco != ab_forte
    assert "prioritário" in ab_fraco          # alto risco -> entra na triagem
    assert "não figura entre os casos prioritários" in ab_forte


def test_narrativa_reflete_a_faixa():
    p = core.prever(ALUNO)
    abertura, _ = core.narrativa_pedagogica(ALUNO, p)
    faixa = core.classificar(p)[0]
    esperado = faixa.replace("🔴", "").replace("🟡", "").replace("🟢", "").strip().lower()
    assert esperado in abertura


def test_atividades_para_cobre_alavancas_e_frageis():
    _, cen = core.detectar_alavancas(ALUNO, core.prever)
    ativ = core.atividades_para(ALUNO, cen)
    assert len(ativ) >= 1
    # aluno com Defasagem = -1 deve receber o plano de recuperação
    assert any("Recuperação de defasagem" in r for r, _ in ativ)
