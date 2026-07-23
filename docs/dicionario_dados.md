# Dicionário de Dados

Fonte: `BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (abas `PEDE2022`, `PEDE2023`, `PEDE2024`).

## Variáveis usadas pelo modelo (ano T)

| Campo | Tipo | Faixa | Descrição |
|---|---|---|---|
| `Defasagem` | inteiro | −5 a +3 | Anos de defasagem escolar. `0` = no nível; negativo = atrasado. |
| `Fase_num` | numérico | 0 a 8 | Fase no programa. `ALFA` = 0. |
| `Idade` | inteiro | 7 a 25 | Idade do aluno no ano. |
| `tempo_prog` | inteiro | 0 a 15 | Anos no programa (`Ano` − `Ano ingresso`). |
| `Pedra` | categórico | Quartzo, Agata, Ametista, Topázio | Estágio de maturidade no programa. |
| `IDA` | numérico | 0 a 10 | Desempenho Acadêmico. |
| `IEG` | numérico | 0 a 10 | Engajamento. |
| `IPV` | numérico | 0 a 10 | Protagonismo. |
| `IAA` | numérico | 0 a 10 | Autoavaliação. |
| `IPS` | numérico | 0 a 10 | Psicossocial. |
| `IPP` | numérico | 0 a 10 | Psicopedagógico. |

## Alvo

| Campo | Definição |
|---|---|
| `alvo` | `1` se `Defasagem < 0` no ano **T+1**; `0` caso contrário. |

## Campos presentes na base mas **não** usados

| Campo | Motivo |
|---|---|
| `IAN` | Determinado pela Defasagem: `IAN=10` se Defasagem ≥ 0; `5` se −1; `2,5` se ≤ −2. |
| `INDE` | Índice composto que incorpora o IAN. |
| `Fase Ideal` | Usado apenas no diagnóstico de leakage (`Defasagem = Fase − Fase Ideal`). |
| `Genero` | Excluído (piora o modelo e levanta questão de equidade). |
| `Nome`, `Nome Anonimizado`, `RA` | Identificação — nunca entram no modelo nem no repositório. |

## Normalizações aplicadas
- `Fase`: texto → número (`ALFA` → 0; `"1A"`, `"1B"` → 1; etc.).
- `Pedra`: `Ágata` → `Agata`; `Incluir` → `Outros`; `Title Case`.
- Nomes de coluna variam por ano (`Defas`/`Defasagem`, `INDE 22`/`INDE 2023`/`INDE 2024`,
  `Idade 22`/`Idade`) — unificados no carregamento.

## Privacidade
A base contém nomes de menores. O arquivo **nunca** vai ao repositório (bloqueado no
`.gitignore`). No app, apenas agregados estatísticos (medianas, quartis) estão embutidos.
