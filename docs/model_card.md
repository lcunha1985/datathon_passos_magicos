# Model Card — Risco de Defasagem

> ⚠️ **Atualizado em 22/07/2026 com os números reais do `model.joblib` atual** (retreinado em
> 20/07/2026). O campo interno `versao` do bundle está gravado como `v0` — é só um
> identificador interno do bundle, sem relação com nenhuma numeração de versão do projeto.
> As seções **Dados**, **Vazamento** e **Baselines** abaixo não foram recalculadas para este
> retreino específico (estão marcadas onde isso importa); só **Desempenho** e **Calibração**
> vêm direto do bundle atual.

## Uso pretendido
Apoiar a equipe pedagógica da Passos Mágicos a **identificar cedo** alunos com maior chance de
estar em defasagem escolar **no ano seguinte**, para direcionar apoio preventivo.
**Não** é decisão automática nem veredito sobre o aluno.

## Alvo
`Defasagem < 0` no ano **T+1** (aluno abaixo do nível esperado no ano seguinte).

## Dados
> ⚠️ Contagens abaixo são do treino anterior, não reconfirmadas neste retreino.

Base PEDE 2022, 2023 e 2024. Pares aluno-ano consecutivos (transições):
- 2022 → 2023: 600
- 2023 → 2024: 765
- Total: 1.365 transições (897 alunos únicos)

## Variáveis (11)
`Defasagem`, `Fase_num`, `Idade`, `tempo_prog`, `Pedra` (categórica) e os seis índices
pedagógicos: `IDA`, `IEG`, `IPV`, `IAA`, `IPS`, `IPP` — todos do **ano T**.

### Variáveis excluídas e por quê
| Variável | Motivo |
|---|---|
| `IAN` | Derivado diretamente da Defasagem (leakage). |
| `INDE` | Incorpora o IAN em sua composição. |
| `Genero` | Piorava o desempenho (PR-AUC 0,830 → 0,816) e levanta questão de equidade. |

## Cuidado com vazamento (leakage)
A base contém a coluna `Fase Ideal`. Verificamos que **`Defasagem = Fase − Fase Ideal` em 100%
dos casos**, com `Fase Ideal` determinada pela `Idade`. Portanto, prever a defasagem *do mesmo
ano* a partir de Fase/Idade é reproduzir uma subtração já conhecida — não é previsão.

Evidência (modelo prevendo a defasagem do mesmo ano):
| Conjunto de variáveis | ROC-AUC |
|---|---|
| 14 features (incl. Fase e Idade) | 0,9152 |
| **Apenas Fase + Idade** | **0,8891** |
| Apenas os 6 índices | 0,6160 |

Por isso o alvo foi movido para o **ano seguinte** (T+1), genuinamente desconhecido no momento
da previsão.

## Modelo
RandomForest (400 árvores, profundidade 6, `class_weight='balanced'`) **calibrado** com
`CalibratedClassifierCV(method='sigmoid', cv=5)`, dentro de um `Pipeline` com imputação
(mediana/moda), padronização e one-hot da `Pedra`.

Comparação (teste out-of-time 2023→2024, PR-AUC — **do treino anterior, não reconfirmada**:
o PR-AUC do RandomForest já mudou de 0,830 para 0,900 no bundle atual): RandomForest 0,830 ·
GradientBoosting 0,815 · LightGBM 0,814 · LogisticRegression 0,780.

## Desempenho
**Teste out-of-time** (treino 2022→2023, teste 2023→2024) — extraído de `model.joblib`:
- AUC-PR **0,900** · AUC-ROC **0,934**
- Recall **0,860** · Precisão **0,818** · F1 **0,839**
- Limiar **0,53** (campo `threshold` do bundle atual)

**Robustez** (StratifiedGroupKFold por aluno, todas as transições): AUC-PR **0,865**
(bundle não grava desvio-padrão)

**Calibração** (faixa prevista × taxa real observada) — extraído de `model.joblib`:
| Faixa | n | Defasagem real em T+1 |
|---|---|---|
| Baixo (0–36%) | 319 | 4,1% |
| Moderado (36–67%) | 250 | 46,0% |
| Alto (67–100%) | 196 | 91,8% |

**Baselines** (números do treino anterior, **não reconfirmados** neste retreino — o AUC-ROC do
"modelo completo" abaixo, 0,864, já não bate com o AUC-ROC atual de 0,934; recalcular antes de
citar):
- Persistência pura ("continua como está"): AUC-ROC 0,677
- Só estado atual (Defasagem, Fase, Idade): 0,822
- Só os índices pedagógicos: 0,643
- Modelo completo (número antigo): 0,864

## Limitações
- A **defasagem é persistente**: o estado atual domina a previsão. O modelo é transparente
  quanto a isso (ver importância das variáveis).
- Houve **mudança de distribuição** entre os anos (taxa do alvo: 61,0% em 2022→2023 contra
  40,3% em 2023→2024). As métricas out-of-time já refletem esse cenário mais exigente.
- As análises de **alavancas** e do **simulador** são de **sensibilidade do modelo**
  (associação), não relações causais comprovadas.
- Base de uma única instituição; não deve ser generalizada sem revalidação.

## Ética e privacidade
- Uso para **direcionar apoio**, nunca para rotular, ranquear ou excluir alunos.
- O aplicativo **não armazena** dados inseridos.
- Dados individuais de alunos **não** fazem parte deste repositório.
- O modelo **não usa gênero**.
