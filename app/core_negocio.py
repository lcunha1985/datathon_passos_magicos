"""Agregados pré-calculados — Análise de Negócio (10 perguntas do case).

Fonte: PEDE_PASSOS_DATASET_FIAP.csv (base oficial FIAP, anos 2020-2022).
Cálculo feito uma única vez fora do app (fora do repositório fica a base bruta,
que tem nomes — aqui só ficam números agregados, sem dado individual de aluno).

⚠️ HOMOLOGAÇÃO: quando a base 2024 (mesma usada pelo model.joblib em produção)
estiver disponível, reprocessar este módulo com 2020+2021+2022 (teste) e
2024 (homologação) para validar se os padrões se mantêm fora da amostra.

Gerado a partir de: PEDE_PASSOS_DATASET_FIAP.csv (2020, 2021, 2022) · 1.349 alunos ·
2.275 linhas aluno-ano após limpeza.

Perguntas 3, 6 e 8 pedem relação/coerência entre indicadores (não evolução
temporal) — a resposta principal usa a base agregada dos 3 anos (mais poder
estatístico), com uma quebra por ano abaixo como checagem de estabilidade.
Perguntas 1, 2, 4, 5, 7 e 10 pedem evolução — são todas temporais.
"""

AGREGADOS = {'q1_faixa_ian_ano': {'Sem defasagem': {'2020': 49.4, '2021': 38.8, '2022': 30.0},
                      'Moderada': {'2020': 49.1, '2021': 59.8, '2022': 66.7},
                      'Severa': {'2020': 1.5, '2021': 1.5, '2022': 3.2}},
 'q1_severa_moderada_2022_pct': 69.9,
 'q2_ida_pedra_ano': {'Quartzo': {'2020': 2.01, '2021': 2.21, '2022': 3.1},
                      'Agata': {'2020': 5.43, '2021': 4.63, '2022': 5.24},
                      'Ametista': {'2020': 7.67, '2021': 6.39, '2022': 7.0},
                      'Topázio': {'2020': 9.04, '2021': 7.55, '2022': 8.24}},
 'q2_kruskal_ano': {'H': 82.434, 'p': 0.0},
 'q2_ida_medio_ano': {'2020': 6.32, '2021': 5.43, '2022': 6.07},
 'q3_corr': {'IEG': {'IEG': 1.0, 'IDA': 0.551, 'IPV': 0.487},
             'IDA': {'IEG': 0.551, 'IDA': 1.0, 'IPV': 0.45},
             'IPV': {'IEG': 0.487, 'IDA': 0.45, 'IPV': 1.0}},
 'q3_corr_ieg_ida_por_pedra': {'Quartzo': 0.196,
                               'Agata': -0.083,
                               'Ametista': 0.093,
                               'Topázio': -0.053},
 'q3_corr_ieg_ipv_por_pedra': {'Quartzo': 0.325,
                               'Agata': 0.078,
                               'Ametista': -0.045,
                               'Topázio': 0.115},
 'q4_corr_iaa_ida': 0.227,
 'q4_corr_iaa_ieg': 0.279,
 'q4_pct_superestimadores': 53.8,
 'q4_pct_subestimadores': 3.7,
 'q4_pct_coerentes': 42.5,
 'q4_mismatch_medio': 2.31,
 'q5_ips_medio_caiu': 6.84,
 'q5_ips_medio_nao_caiu': 6.82,
 'q5_n_caiu': 520,
 'q5_n_nao_caiu': 394,
 'q5_mannwhitney_p': 0.23857,
 'q6_corr_ipp_ian': 0.128,
 'q6_contingencia_ian_ipp': {'Baixo': {'Moderada': 38.5, 'Sem defasagem': 26.2, 'Severa': 34.7},
                             'Médio': {'Moderada': 32.0, 'Sem defasagem': 34.0, 'Severa': 46.9},
                             'Alto': {'Moderada': 29.4, 'Sem defasagem': 39.7, 'Severa': 18.4}},
 'q6_pct_divergencia': 10.2,
 'q7_coef_logit_pontovirada': {'IDA': 0.678, 'IEG': 1.238, 'IPS': 0.28, 'IPP': 1.006},
 'q7_n': 2247,
 'q7_taxa_pv': 14.0,
 'q8_coef_inde': {'IDA': 0.257, 'IEG': 0.256, 'IPS': 0.142, 'IPP': 0.127},
 'q8_r2': 0.885,
 'q8_corr_inde': {'IDA': 0.811, 'IEG': 0.8, 'IPS': 0.319, 'IPP': 0.238},
 'q10_inde_pedra_ano': {'Quartzo': {'2020': 5.27, '2021': 4.53, '2022': 5.22},
                        'Agata': {'2020': 6.77, '2021': 6.28, '2022': 6.61},
                        'Ametista': {'2020': 7.91, '2021': 7.54, '2022': 7.53},
                        'Topázio': {'2020': 8.84, '2021': 8.63, '2022': 8.37}},
 'q10_inde_medio_ano': {'2020': 7.3, '2021': 6.89, '2022': 7.03},
 'q10_kruskal_ano': {'H': 47.254, 'p': 0.0},
 'q3_corr_ieg_ida_por_ano': {'2020': 0.52, '2021': 0.626, '2022': 0.509},
 'q3_corr_ieg_ipv_por_ano': {'2020': 0.267, '2021': 0.705, '2022': 0.545},
 'q4_corr_iaa_ida_por_ano': {'2020': 0.247, '2021': 0.224, '2022': 0.186},
 'q4_pct_superestimadores_por_ano': {'2020': 44.6, '2021': 63.6, '2022': 53.8},
 'q6_corr_ipp_ian_por_ano': {'2020': 0.057, '2021': 0.017, '2022': 0.106},
 'q7_coef_por_ano': {'2020': {'IDA': 0.377, 'IEG': 0.784, 'IPS': -0.099, 'IPP': 1.871},
                     '2021': {'IDA': 0.863, 'IEG': 1.863, 'IPS': 1.295, 'IPP': 1.41},
                     '2022': {'IDA': 1.52, 'IEG': 0.911, 'IPS': 0.361, 'IPP': 1.077}},
 'q7_taxa_pv_por_ano': {'2020': 13.4, '2021': 15.8, '2022': 13.1},
 'q7_n_por_ano': {'2020': 701, '2021': 684, '2022': 862},
 'q8_corr_inde_por_ano': {'2020': {'IDA': 0.802, 'IEG': 0.748, 'IPS': 0.394, 'IPP': 0.134},
                          '2021': {'IDA': 0.853, 'IEG': 0.872, 'IPS': 0.307, 'IPP': 0.554},
                          '2022': {'IDA': 0.821, 'IEG': 0.805, 'IPS': 0.274, 'IPP': 0.285}},
 'q8_r2_por_ano': {'2020': 0.878, '2021': 0.922, '2022': 0.888},
 'q4_iaa_medio_por_ano': {'2020': 8.37, '2021': 8.15, '2022': 8.26},
 'q4_ida_medio_por_ano': {'2020': 6.32, '2021': 5.43, '2022': 6.07},
 'q4_mismatch_medio_por_ano': {'2020': 2.05, '2021': 2.73, '2022': 2.19},
 'q4_kruskal_mismatch_ano': {'H': 49.641, 'p': 0.0},
 'q5_por_par_ano': {'2020->2021': {'ips_medio_caiu': 6.79,
                                   'ips_medio_nao_caiu': 6.7,
                                   'n_caiu': 316,
                                   'n_nao_caiu': 141,
                                   'mannwhitney_p': 0.74387},
                    '2021->2022': {'ips_medio_caiu': 6.91,
                                   'ips_medio_nao_caiu': 6.88,
                                   'n_caiu': 204,
                                   'n_nao_caiu': 253,
                                   'mannwhitney_p': 0.22919}}}


# ── Metadados fixos usados pela página ────────────────────────────────────────
PEDRA_ORDEM = ["Quartzo", "Agata", "Ametista", "Topázio"]
ANOS_BASE = [2020, 2021, 2022]

PERGUNTAS = {
    1: "Adequação do nível (IAN)",
    2: "Desempenho acadêmico (IDA)",
    3: "Engajamento nas atividades (IEG)",
    4: "Autoavaliação (IAA)",
    5: "Aspectos psicossociais (IPS)",
    6: "Aspectos psicopedagógicos (IPP)",
    7: "Ponto de virada (IPV)",
    8: "Multidimensionalidade dos indicadores (INDE)",
    9: "Previsão de risco com Machine Learning",
    10: "Efetividade do programa",
}
