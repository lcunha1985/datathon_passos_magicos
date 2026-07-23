# Passos Mágicos — Risco de Defasagem Educacional

Aplicativo Streamlit que estima a chance de um aluno estar em **defasagem escolar no ano
seguinte**, a partir dos indicadores do ano corrente. Ferramenta de **apoio à decisão** da
equipe pedagógica — não uma decisão automática.

## Como executar

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Deploy: Streamlit Community Cloud · Main file path: `app/app.py` · Python 3.11.

## Três páginas (seletor na barra lateral)

- **🙂 Análise Pedagógica** — uso diário da equipe: leitura em linguagem natural, semáforo de
  fatores (SHAP), comparação com a base, **detector de alavancas** (onde focar), **simulador
  "e se…"** (visualmente separado do resultado real) e **laudo em PDF**.
- **📊 Análise de Negócio** — as 10 perguntas analíticas do case (IAN, IDA, IEG, IAA, IPS, IPP,
  IPV, INDE, previsão de risco e efetividade do programa), com evolução ano a ano onde a
  pergunta pede e checagem de estabilidade nas demais. Base: `PEDE_PASSOS_DATASET_FIAP.csv`
  (2020-2022) — ver `app/core_negocio.py` para os agregados e `docs/` para o dicionário de
  dados. **Relatório de negócio em PDF**, em linguagem simples para leigos.
- **🔬 Análise Técnica** — dashboard do modelo: métricas, calibração, importância das
  variáveis, SHAP, sensibilidade, mapa de calor, posição na base e **relatório técnico em PDF**.

Todos os botões de PDF do app seguem o mesmo padrão: **um clique gera e já baixa o arquivo**,
sem precisar de um segundo botão "Baixar".

Também: identidade visual da instituição, **modo escuro**, ⓘ explicativo em cada gráfico e
**predição em lote** via CSV — com relatório das linhas descartadas (linha e motivo) e
**seleção de linha**: ao clicar num aluno da tabela, o laudo pedagógico completo dele é
gerado na tela e em PDF.

## O modelo

RandomForest calibrado. Alvo: `Defasagem < 0` no **ano seguinte (T+1)**.
Treino com as transições 2022→2023 e 2023→2024 da base PEDE.

> ℹ️ O campo interno `versao` do `model.joblib` está gravado como `v0` — é só um identificador
> interno do bundle treinado em 20/07/2026, sem relação com nenhuma numeração de versão do
> projeto (o projeto não versiona por número).

| Métrica (teste out-of-time 2023→2024) | Valor |
|---|---|
| AUC-PR | 0,900 |
| AUC-ROC | 0,934 |
| Recall | 0,860 |
| Precisão | 0,818 |
| F1 | 0,839 |
| Robustez (GroupKFold por aluno) | AUC-PR 0,865 |

Limiar **0,53**, calibrado no retreino de 20/07/2026 (extraído de `model.joblib`, campo
`threshold`).

**Calibração** (faixa prevista × taxa real observada, extraído de `model.joblib`):

| Faixa | n | Defasagem real em T+1 |
|---|---|---|
| Baixo (0–36%) | 319 | 4,1% |
| Moderado (36–67%) | 250 | 46,0% |
| Alto (67–100%) | 196 | 91,8% |

### Cuidado com vazamento de dados (leakage)

> ℹ️ A análise abaixo é da base de dados em si (não do modelo específico) e não muda com
> retreinos — mas os números não foram reconfirmados após o retreino de 20/07/2026.
> Recomendo rodar de novo no notebook antes de citar com confiança.

A base traz a coluna `Fase Ideal`, e verificamos que **`Defasagem = Fase − Fase Ideal` em 100%
dos casos**, com a `Fase Ideal` determinada pela `Idade`. Prever a defasagem *do mesmo ano* a
partir de Fase/Idade é reproduzir uma subtração já conhecida:

| Variáveis (prevendo o mesmo ano) | ROC-AUC |
|---|---|
| Fase + Idade + indicadores | 0,915 |
| **Apenas Fase + Idade** | **0,889** |
| Apenas os indicadores | 0,616 |

Por isso: **`IAN` e `INDE` foram excluídos** (derivados da defasagem) e o **alvo foi movido para
o ano seguinte**. Detalhes em [`docs/model_card.md`](docs/model_card.md).

### Baselines (contexto honesto)

> ⚠️ Tabela não reconfirmada após o retreino de 20/07/2026 — os números do modelo completo
> abaixo (0,864) não batem com o AUC-ROC atual do `model.joblib` (0,934). Recomendo recalcular
> no notebook antes de publicar.

| Referência | ROC-AUC |
|---|---|
| Persistência pura ("continua como está") | 0,677 |
| Só o estado atual (Defasagem, Fase, Idade) | 0,822 |
| Só os índices pedagógicos | 0,643 |
| Modelo completo (número antigo, não reconfirmado) | 0,864 |

Os índices pedagógicos agregavam **+0,042** de AUC sobre o estado atual no treino anterior —
vale recalcular essa margem com o modelo atual.

## Estrutura

```
app/app.py           Interface Streamlit
app/core.py          Lógica pura (testável) do modelo de risco
app/core_negocio.py  Agregados pré-calculados da Análise de Negócio (10 perguntas)
app/pdf_report.py    Geração dos PDFs (laudo, técnico e de negócio)
app/assets/logo.png  Logo
models/model.joblib  Modelo treinado (bundle)
notebooks/           Notebook de treino + diagnóstico de leakage
docs/                Model card e dicionário de dados
tests/               39 testes (pytest)
```

## Ética e privacidade

- Estimativa de apoio, **não** um veredito. Usar para direcionar apoio, nunca para rotular.
- O app **não armazena** dados inseridos.
- O modelo **não usa gênero** — além da questão de equidade, incluí-lo piorava o desempenho
  (PR-AUC 0,830 → 0,816).
- A base contém nomes de menores e **não** faz parte deste repositório (bloqueada no
  `.gitignore`). Apenas agregados estatísticos estão embutidos no app.
- A **Análise de Negócio** usa uma base diferente da que treina o modelo de risco:
  `PEDE_PASSOS_DATASET_FIAP.csv` (2020-2022) em vez da base PEDE 2022-2024. É uso de
  desenvolvimento/teste — os agregados em `app/core_negocio.py` devem ser reprocessados com a
  base 2024 para homologação fora da amostra assim que ela estiver disponível.
