# Tech Challenge — Fase 3 | FIAP

## Predição e análise estratégica da alfabetização no Brasil

Projeto de Machine Learning desenvolvido na Fase 3 do Tech Challenge da FIAP, em continuidade ao tratamento e à consolidação de dados realizados na Fase 2.

O trabalho combina classificação supervisionada de alunos, análise de indicadores educacionais municipais, comparação de algoritmos, interpretabilidade, agrupamento de municípios e discussão de hipóteses causais.

## Sumário

- [Objetivos](#1-objetivos)
- [Dados e preparação](#2-origem-e-preparação-dos-dados)

- [Pipeline e validação](#5-pipeline-de-machine-learning)
- [Algoritmos e experimentos](#7-algoritmos-e-experimentos)

- [Resultados do modelo final](#8-modelo-final-e-avaliação)
- [Risco educacional municipal](#10-diagnóstico-de-risco-educacional-municipal)

- [K-Means](#11-agrupamento-municipal-com-k-means)
- [Enriquecimento socioeconômico com IBGE](#12-enriquecimento-socioeconômico-com-dados-do-ibge)

- [Reprodução do projeto](#16-reprodução-do-pipeline)
- [Respostas às cinco perguntas da FIAP](#21-perguntas-estratégicas-da-fiap--respostas-e-evidências)

- [Aplicação prática para políticas públicas](#22-aplicação-prática-para-políticas-públicas)
- [Possíveis evoluções futuras](#23-possíveis-evoluções-futuras)
- [Considerações finais](#24-considerações-finais)

## 1. Objetivos

O objetivo principal é prever a condição de alfabetização de um aluno, representada por uma variável binária:
- 1: alfabetizado.
- 0: não alfabetizado.

Objetivos complementares:
- Investigar fatores associados à alfabetização.
- Comparar algoritmos e estratégias de pré-processamento.
- Avaliar a capacidade de generalização dos modelos.
- Identificar municípios abaixo das metas educacionais.
- Comparar padrões regionais.
- Agrupar municípios com características educacionais semelhantes.
- Discutir limites da interpretação causal dos resultados.

## 2. Origem e preparação dos dados

O projeto utiliza dados das camadas Silver e Gold produzidas na Fase 2, além de bases intermediárias criadas para o treinamento e a avaliação dos modelos.

Diretórios:
- data/silver: dados tratados.
- data/gold: indicadores consolidados e bases analíticas.
- data/intermediate: conjuntos de treino, validação e teste.

A base supervisionada possui 6.419 registros de alunos com informações suficientes para definição do alvo.

A variável `target_alfabetizado` foi definida a partir da proficiência registrada na base educacional. Neste projeto, valores iguais ou superiores a 743 correspondem à classe 1 (alfabetizado), e valores inferiores correspondem à classe 0 (não alfabetizado). Esse critério operacional não substitui uma avaliação individual completa da aprendizagem.

A proficiência utilizada para definir o alvo não é utilizada como preditora, evitando um vazamento direto da resposta.

A amostra individual não deve ser interpretada como uma estimativa representativa de todos os estudantes brasileiros.

Os indicadores municipais são analisados separadamente dos resultados do classificador individual.

## 3. Análise exploratória

A análise exploratória investigou:
- Distribuição das classes.
- Valores ausentes.
- Características dos alunos.
- Distribuição territorial.
- Indicadores municipais.
- Relação entre alfabetização e participação nas avaliações.

Scripts relacionados:

- [src/analysis/audit_data.py](src/analysis/audit_data.py)
- [src/analysis/eda_alfabetizacao.py](src/analysis/eda_alfabetizacao.py)

- [src/analysis/eda_classes.py](src/analysis/eda_classes.py)
- [src/analysis/investigar_alunos.py](src/analysis/investigar_alunos.py)

## 4. Engenharia de atributos e prevenção de vazamento

O modelo final utiliza quatro variáveis:
- rede: rede de ensino.
- sigla_uf: unidade federativa.
- regiao: região brasileira.
- taxa_alfabetizacao_2023: taxa municipal de alfabetização do ano anterior.

A utilização do indicador de 2023 evita empregar diretamente a taxa municipal de alfabetização de 2024 para prever resultados individuais referentes a 2024.

A proficiência individual e os indicadores contemporâneos derivados do resultado não são usados como preditores do modelo final.

A engenharia de atributos históricos está implementada em:

- [src/preprocessing/add_historical_features.py](src/preprocessing/add_historical_features.py)

## 5. Pipeline de Machine Learning

O projeto utiliza componentes do scikit-learn:
- SimpleImputer para imputação de valores ausentes.
- OneHotEncoder para variáveis categóricas.
- StandardScaler para a variável numérica do modelo final.
- ColumnTransformer para organizar as transformações.
- Pipeline para integrar pré-processamento e classificador.

O pré-processamento é ajustado dentro do treinamento e de cada fold de validação, reduzindo o risco de vazamento de dados.

## 6. Separação dos dados e validação

A divisão entre treino, validação e teste foi realizada por município, utilizando GroupShuffleSplit.

Distribuição dos conjuntos:
| Conjunto | Alunos | Municípios |
|---|---:|---:|
| Treino | 4.137 | 1.361 |
| Validação | 1.152 | 454 |
| Teste | 1.130 | 454 |

Os municípios não se sobrepõem entre os três conjuntos.

Para comparar experimentos, foi utilizada validação cruzada com cinco folds e agrupamento por município, por meio de StratifiedGroupKFold.

Essa abordagem permite uma avaliação mais rigorosa da generalização territorial do que uma divisão aleatória de alunos.

## 7. Algoritmos e experimentos

Foram avaliados:
- Regressão Logística.
- Regressão Logística com balanceamento de classes.
- Random Forest.
- Regressão Logística com indicador histórico municipal.
- HistGradientBoostingClassifier.
- Regressão Logística com participação histórica adicional.
- Comparação entre StandardScaler e RobustScaler.

### Resultados médios de validação cruzada
| Experimento | Acurácia balanceada | ROC-AUC | F1-macro |
|---|---:|---:|---:|
| Regressão Logística com histórico | 0,5989 | 0,6445 | 0,5803 |
| Gradient Boosting | 0,5695 | 0,6359 | 0,5562 |
| Histórico + participação + StandardScaler | 0,5997 | 0,6501 | 0,5846 |
| Histórico + participação + RobustScaler | 0,5998 | 0,6501 | 0,5847 |

Os resultados são provenientes de validação cruzada agrupada, não do conjunto de teste final.

### Decisões técnicas

A Regressão Logística com balanceamento de classes foi mantida como modelo final pelo desempenho competitivo e pela facilidade de interpretação. Os experimentos com modelos mais complexos e atributos adicionais não demonstraram ganhos consistentes que justificassem sua substituição.

O Gradient Boosting não superou a Regressão Logística histórica nas métricas principais de validação cruzada.

A inclusão da participação de 2023 apresentou melhorias pequenas, sem evidência suficiente para justificar a substituição do modelo final já selecionado.

StandardScaler e RobustScaler apresentaram resultados praticamente equivalentes. O StandardScaler foi preservado no pipeline final.

Os experimentos adicionais foram avaliados por validação cruzada, sem utilizar o teste reservado para selecionar novos candidatos.

## 8. Modelo final e avaliação

Algoritmo: LogisticRegression com class_weight="balanced".

O treinamento final combina os conjuntos de treino e validação.

Modelo salvo em:
- models/modelo_final_alfabetizacao.joblib

### Resultados no teste reservado
| Métrica | Resultado |
|---|---:|
| Alunos avaliados | 1.130 |
| Municípios | 454 |
| Acurácia | 0,6204 |
| Acurácia balanceada | 0,6135 |
| ROC-AUC | 0,6678 |
| F1-macro | 0,6089 |
| Recall — não alfabetizados | 0,5839 |
| Recall — alfabetizados | 0,6432 |

Matriz de confusão, considerando as classes 0 e 1:

```text

[[254, 181],

 [248, 447]]

```

Arquivos relacionados:

- [reports/metricas_teste_final.json](reports/metricas_teste_final.json)
- [images/matriz_confusao_teste_final.png](images/matriz_confusao_teste_final.png)

O desempenho do modelo é moderado. Seus resultados não justificam decisões automatizadas ou definitivas sobre a alfabetização de estudantes.

## 9. Interpretabilidade

Foi utilizada importância por permutação para investigar quais variáveis contribuem para o comportamento preditivo do modelo final.

A taxa municipal de alfabetização de 2023 apresentou a maior contribuição no experimento.

Também foram identificadas contribuições da unidade federativa, região e rede.

Arquivos:

- [reports/importancia_variaveis_final.csv](reports/importancia_variaveis_final.csv)
- [images/importancia_variaveis_final.png](images/importancia_variaveis_final.png)

A importância foi calculada sobre os dados de treinamento e validação; não deve ser interpretada como estimativa de efeito causal nem como avaliação independente de generalização.

## 10. Diagnóstico de risco educacional municipal

A análise municipal compara taxas de alfabetização e metas educacionais.

Na comparação com a meta de 2024:
- 5.232 municípios apresentaram informações suficientes.
- 2.444 estavam abaixo da meta.
- 2.788 atingiram a meta.
- 120 não tinham informações suficientes para essa classificação.

Entre os municípios com dados válidos, 46,71% estavam abaixo da meta de 2024.

Ao comparar a taxa observada de 2024 com a meta de 2025:
- 5.352 municípios foram analisados.
- 3.030 estavam abaixo do patamar estabelecido para 2025.
- 2.322 estavam no patamar ou acima dele.

Essa comparação expressa a distância até a meta, não uma previsão validada dos resultados de 2025.

Arquivos:

- [src/analysis/analise_risco_municipal.py](src/analysis/analise_risco_municipal.py)
- [reports/diagnostico_risco_municipal_2024.csv](reports/diagnostico_risco_municipal_2024.csv)

- [reports/resumo_risco_regional_2024.csv](reports/resumo_risco_regional_2024.csv)
- [reports/relatorio_estrategico_municipal.md](reports/relatorio_estrategico_municipal.md)

- [images/risco_municipal_por_regiao_2024.png](images/risco_municipal_por_regiao_2024.png)

## 11. Agrupamento municipal com K-Means

Foi realizada uma análise descritiva de 5.352 municípios utilizando:
- Taxa municipal de alfabetização de 2024.
- Percentual de participação na avaliação em 2024.

As variáveis foram padronizadas com StandardScaler.

Foram testados de dois a seis clusters, utilizando o índice silhouette para apoiar a escolha da quantidade de grupos.

A melhor configuração foi de três clusters, com silhouette aproximado de 0,3842.

### Perfis encontrados
| Cluster | Municípios | Alfabetização média | Participação média |
|---|---:|---:|---:|
| 0 | 1.642 | 46,73% | 93,61% |
| 2 | 1.365 | 53,98% | 82,64% |
| 1 | 2.345 | 79,74% | 94,10% |

Interpretação:
- Cluster 0: baixa alfabetização e alta participação.
- Cluster 2: alfabetização intermediária e menor participação.
- Cluster 1: alta alfabetização e alta participação.

O índice silhouette indica separação moderada entre os grupos.

Os clusters são descritivos, dependem das variáveis escolhidas e não representam classificações oficiais.

Arquivos:

- [src/analysis/clusterizacao_municipios.py](src/analysis/clusterizacao_municipios.py)
- [reports/clusters_municipios_2024.csv](reports/clusters_municipios_2024.csv)

- [reports/resumo_clusters_municipios_2024.csv](reports/resumo_clusters_municipios_2024.csv)
- [reports/avaliacao_kmeans_silhouette.csv](reports/avaliacao_kmeans_silhouette.csv)

- [reports/interpretacao_clusters_municipais.md](reports/interpretacao_clusters_municipais.md)
- [images/clusters_municipios_2024.png](images/clusters_municipios_2024.png)

## 12. Enriquecimento socioeconômico com dados do IBGE

### 12.1 Origem e integração dos dados

Foi incorporado um indicador socioeconômico oficial do IBGE,

proveniente do Censo Demográfico 2022, tabela SIDRA 10295:

**rendimento nominal médio mensal domiciliar per capita**.

Fonte: https://sidra.ibge.gov.br/tabela/10295

O indicador corresponde ao ano de 2022 e foi utilizado como característica histórica. Os valores não representam renda individual dos estudantes nem renda observada em 2024.

O processamento seguiu a arquitetura de dados do projeto:
- **Bronze:** `data/bronze/ibge_renda_municipal_2022_original.csv`,

  preservando a extração original.
- **Silver:** `data/silver/ibge_renda_municipal_2022.parquet`,

  com padronização, conversão numérica e validações.
- **Gold:** `data/gold/municipio_enriquecido_socioeconomico.parquet`,

  integrando a renda aos indicadores educacionais municipais.

A Silver contém 5.570 municípios, sem rendas ausentes ou não positivas. A integração com a Gold contemplou os 5.352 municípios presentes na base educacional, sem perda de registros e com cobertura de renda de 100%.

A extração do IBGE utiliza estimativas preliminares da amostra do Censo 2022, conforme as notas metodológicas da fonte.

### 12.2 Relação entre renda e alfabetização municipal

A análise descritiva considerou 5.352 municípios, relacionando a renda per capita de 2022 à taxa de alfabetização de 2024.
- Correlação de Pearson: **0,1914**.
- Correlação de Spearman: **0,2029**.

Taxa média de alfabetização por quartil de renda:
| Faixa de renda | Taxa média de alfabetização |
|---|---:|
| 25% de menor renda | 56,91% |
| Intermediária baixa | 60,41% |
| Intermediária alta | 68,79% |
| 25% de maior renda | 66,05% |

As correlações de Pearson variaram por região: Norte (0,3399),

Sul (0,0648), Centro-Oeste (-0,0125), Nordeste (-0,1179) e Sudeste (-0,1354).

Os resultados mostram associação territorial fraca e heterogênea.

Não permitem inferir causalidade ou conclusões sobre a renda das famílias dos alunos.

### 12.3 Experimentos de Machine Learning

Foram comparadas três configurações de regressão logística,

utilizando os mesmos cinco folds de validação cruzada agrupados por município, com 5.289 alunos de treino e validação.
| Configuração | Balanced accuracy | ROC-AUC | F1-macro | Recall classe 0 |
|---|---:|---:|---:|---:|
| Modelo histórico atual | **0,5989** | **0,6445** | 0,5803 | 0,6344 |
| Modelo com renda | 0,5971 | 0,6423 | **0,5806** | 0,6190 |
| Modelo com interação renda × região | 0,5983 | 0,6435 | 0,5786 | **0,6448** |

A inclusão da renda e de suas interações com a região não produziu ganho consistente de desempenho geral. Por isso,

**o modelo histórico original foi mantido como modelo final**.

O conjunto de teste não foi utilizado nesses experimentos.

### 12.4 Evidências e reprodutibilidade

Scripts:
- `src/preprocessing/tratar_renda_ibge.py`
- `src/preprocessing/auditar_renda_ibge.py`
- `src/preprocessing/integrar_renda_gold.py`
- `src/modeling/train_socioeconomico.py`
- `src/modeling/train_interacao_renda_regiao.py`

Relatórios:
- `reports/qualidade_renda_ibge_2022.json`
- `reports/validacao_cruzada_socioeconomico.csv`
- `reports/validacao_cruzada_interacao_renda_regiao.csv`
- `reports/analise_enriquecimento_socioeconomico.md`
- `reports/conclusao_experimentos_renda_ibge.md`

## 13. Análise causal exploratória

Hipótese investigada:

Uma maior participação nas avaliações educacionais causa melhoria na alfabetização municipal?

A hipótese foi discutida com base nas variáveis disponíveis e em possíveis fatores de confusão.

A correlação observada entre participação e alfabetização em 2023 foi de aproximadamente 0,287.

Essa associação não demonstra causalidade.

Entre as limitações estão:
- Possíveis fatores de confusão socioeconômicos e escolares.
- Ausência de intervenção claramente identificada.
- Possibilidade de causalidade reversa.
- Viés de seleção dos estudantes avaliados.
- Disponibilidade de apenas dois anos de indicadores municipais.

O projeto não estima um efeito causal identificado. A análise apresenta uma hipótese, limitações e propostas para estudos futuros.

Documento:

- [reports/analise_causal_alfabetizacao.md](reports/analise_causal_alfabetizacao.md)

## 14. Estrutura do projeto

```text

tech-challenge-fase3/

├── data/

│   ├── bronze/
│   ├── silver/

│   ├── gold/

│   └── intermediate/

├── src/

│   ├── analysis/

│   ├── preprocessing/

│   ├── modeling/

│   ├── evaluation/

│   └── visualization/

├── notebooks/

├── models/

├── reports/

├── images/

├── requirements.txt

├── README.md

└── .gitignore

```

## 15. Requisitos e instalação

Ambiente utilizado: Python 3.12.

Criar um ambiente virtual:

  ```bash
python3.12 -m venv .venv
```

Ativar no macOS ou Linux:

```bash

source .venv/bin/activate

```

Instalar as dependências:

```bash

python -m pip install -r requirements.txt

```

Os comandos a seguir devem ser executados dentro da pasta tech-challenge-fase3.

## 16. Reprodução do pipeline

### Etapa 1 — Disponibilizar os dados

As bases das camadas Silver e Gold da Fase 2 devem estar presentes nos respectivos diretórios.

A preparação completa a partir das fontes originais depende dos dados e das etapas de engenharia desenvolvidas na Fase 2.

### Etapa 2 — Preparar a base supervisionada

```bash

python src/preprocessing/prepare_ml_data.py

```

### Etapa 3 — Enriquecer a base de Machine Learning

```bash

python src/preprocessing/enrich_ml_data.py

```

### Etapa 4 — Adicionar indicadores históricos

```bash

python src/preprocessing/add_historical_features.py

```

### Etapa 5 — Separar treino, validação e teste

```bash

python src/modeling/prepare_split.py

```

### Etapa 6 — Executar validação cruzada

```bash

python src/evaluation/cross_validation.py

```

### Etapa 7 — Treinar o modelo final

```bash

python src/modeling/train_final.py

```

### Etapa 8 — Avaliar o modelo final

```bash

python src/evaluation/evaluate_final.py

```

A avaliação final utiliza o conjunto de teste reservado. Ela não deve ser repetida para escolher hiperparâmetros ou selecionar novos modelos com base nesse teste.

### Etapa 9 — Executar experimentos adicionais

```bash

python src/modeling/train_gradient_boosting.py

```

```bash

python src/modeling/train_participacao_historica.py

```

```bash

python src/modeling/train_robust_scaler.py

```

### Etapa 10 — Executar análises municipais

```bash

python src/analysis/analise_risco_municipal.py

```

```bash

python src/analysis/clusterizacao_municipios.py

```

```bash

python src/analysis/gerar_relatorio_estrategico.py

```

### Etapa complementar — Reproduzir o enriquecimento socioeconômico

Esta análise adicional não altera o modelo final. A extração original da tabela SIDRA 10295 deve estar disponível em `data/bronze/ibge_renda_municipal_2022_original.csv`.

Execute, nesta ordem:

```bash
python src/preprocessing/tratar_renda_ibge.py
python src/preprocessing/auditar_renda_ibge.py
python src/preprocessing/integrar_renda_gold.py
python src/modeling/train_socioeconomico.py
python src/modeling/train_interacao_renda_regiao.py
```

A Silver do IBGE contém 5.570 municípios. A Gold educacional contempla 5.352 municípios, todos com renda correspondente. Nos experimentos individuais, a renda foi obtida diretamente da Silver para preservar a cobertura dos municípios presentes em treino e validação.

Resultados: `reports/validacao_cruzada_socioeconomico.csv`, `reports/validacao_cruzada_interacao_renda_regiao.csv` e `reports/conclusao_experimentos_renda_ibge.md`.

### Etapa 11 — Consultar os resultados

Os arquivos produzidos e documentados estão organizados principalmente em:
- reports/: métricas, tabelas e análises.
- images/: gráficos e visualizações.
- models/: modelos treinados.

Os scripts pressupõem que as bases e os arquivos intermediários necessários estejam disponíveis. A execução deve respeitar as dependências entre as etapas.

## 17. Limitações e uso responsável
- O classificador apresenta desempenho preditivo moderado.
- A amostra individual pode não representar a população brasileira de estudantes.
- Algumas variáveis socioeconômicas e escolares relevantes não estão disponíveis.
- Indicadores históricos municipais não substituem informações individuais de aprendizagem.
- A comparação com metas futuras não constitui previsão temporal validada.
- Correlações, importâncias preditivas e agrupamentos não demonstram causalidade.
- As análises devem apoiar investigação e decisões humanas, não rotular estudantes de forma definitiva.

## 18. Organização e versionamento

O código, os relatórios e as decisões metodológicas são organizados em diretórios específicos.

O projeto utiliza Git para controle de versão e GitHub como repositório remoto.

Repositório da Fase 3:

https://github.com/Gbrajao/Fase-3

As dependências estão documentadas em requirements.txt.

Os ambientes virtuais, arquivos temporários e informações sensíveis não devem ser versionados.

## 19. Continuidade da Fase 2

O projeto dá continuidade à organização dos dados e indicadores produzidos na Fase 2.

Repositório da Fase 2:

https://github.com/Gbrajao/Fase-2

## 20. Apresentação executiva

A entrega prevê uma apresentação em vídeo de até cinco minutos, destacando:
- Problema e objetivos.
- Dados e metodologia.
- Modelo final e resultados.
- Análises territoriais e agrupamentos.
- Principais limitações e recomendações.

## 21. Perguntas estratégicas da FIAP — Respostas e evidências

Esta seção responde diretamente às cinco perguntas estratégicas apresentadas no Tech Challenge da Fase 3.

### 1. Quais fatores mais impactam a alfabetização?

Na análise preditiva realizada, a variável com maior contribuição para o classificador final foi a taxa municipal de alfabetização de 2023.

A importância por permutação também identificou contribuições da unidade federativa, da região e da rede de ensino.

Esses resultados indicam quais informações ajudam o modelo a distinguir alunos alfabetizados e não alfabetizados.

Entretanto, importância preditiva não equivale a impacto causal.

Não é possível concluir que modificar essas variáveis provocaria, necessariamente, uma mudança na alfabetização.

**Evidências:**
- `reports/importancia_variaveis_final.csv`
- `images/importancia_variaveis_final.png`
- `reports/analise_causal_alfabetizacao.md`

### 2. Quais municípios apresentam maior risco educacional?

O projeto identificou municípios que estavam abaixo das metas educacionais de alfabetização em 2024.

Dos 5.232 municípios com informações suficientes para essa comparação, 2.444 não atingiram a meta de 2024, correspondendo a 46,71% dos municípios com dados válidos.

Outros 2.788 atingiram a meta e 120 não possuíam informações suficientes para essa classificação.

A análise também mostrou diferenças regionais. Entre os municípios com dados válidos, o Sul apresentou 66,13% abaixo da meta de 2024, seguido pelo Norte, com 56,30%.

O diagnóstico municipal permite localizar os municípios com maiores distâncias negativas em relação à meta, apoiando a priorização de análises e ações educacionais.

**Evidências:**
- `reports/diagnostico_risco_municipal_2024.csv`
- `reports/resumo_risco_regional_2024.csv`
- `reports/relatorio_estrategico_municipal.md`
- `images/risco_municipal_por_regiao_2024.png`

### 3. Quais regiões possuem padrões semelhantes?

Foram realizadas comparações regionais e uma análise de agrupamento K-Means com 5.352 municípios.

O agrupamento utilizou a taxa de alfabetização e o percentual de participação nas avaliações de 2024.

Foram identificados três perfis:
- **Cluster 0:** 1.642 municípios, alfabetização média de

  46,73% e participação média de 93,61%.
- **Cluster 2:** 1.365 municípios, alfabetização média de

  53,98% e participação média de 82,64%.
- **Cluster 1:** 2.345 municípios, alfabetização média de

  79,74% e participação média de 94,10%.

O melhor índice silhouette foi aproximadamente 0,3842, indicando separação moderada entre os grupos.

Na comparação com a meta de 2024, o percentual de municípios abaixo da meta variou entre as regiões: Sul (66,13%), Norte (56,30%), Nordeste (49,88%), Sudeste (33,77%) e Centro-Oeste (26,90%). Esses percentuais descrevem diferenças regionais, enquanto o K-Means reúne municípios por semelhança nos indicadores utilizados.

Os clusters agrupam municípios, e não regiões inteiras.

Por isso, municípios de diferentes regiões podem apresentar características educacionais semelhantes.

**Evidências:**
- `reports/resumo_risco_regional_2024.csv`
- `reports/resumo_clusters_municipios_2024.csv`
- `reports/clusters_municipios_2024.csv`
- `reports/interpretacao_clusters_municipais.md`
- `images/clusters_municipios_2024.png`

### 4. Como prever municípios que podem não atingir metas futuras?

O projeto realizou uma análise de distância entre a taxa de alfabetização observada em 2024 e a meta estabelecida para 2025.

Dos 5.352 municípios analisados, 3.030 apresentavam taxa de alfabetização de 2024 inferior à meta prevista para 2025, equivalendo a aproximadamente 56,61%.

Essa informação permite identificar municípios que precisariam melhorar seus indicadores para alcançar a meta seguinte.

Entretanto, **não foi desenvolvido nem validado um modelo temporal específico para prever o resultado municipal de 2025**.

Portanto, os 3.030 municípios identificados não devem ser apresentados como municípios que certamente deixarão de atingir a meta.

Uma previsão futura mais robusta exigiria séries históricas mais extensas, variáveis explicativas adicionais e validação temporal apropriada.

**Evidências:**
- `reports/diagnostico_risco_municipal_2024.csv`
- `reports/relatorio_estrategico_municipal.md`

### 5. Quais variáveis possuem maior influência nos modelos?

No classificador final, a importância por permutação foi calculada utilizando a acurácia balanceada como métrica.

As contribuições médias observadas foram:
| Variável | Importância média aproximada |
|---|---:|
| Taxa de alfabetização municipal de 2023 | 0,0708 |
| Unidade federativa | 0,0215 |
| Região | 0,0105 |
| Rede de ensino | 0,0063 |

A taxa histórica municipal foi a variável com maior importância preditiva no experimento.

A análise foi realizada sobre os dados de treino e validação, não constituindo uma avaliação independente no teste.

Os valores indicam a redução média da acurácia balanceada quando cada variável é embaralhada. Não representam percentuais de causalidade ou influência direta sobre a aprendizagem.

**Evidências:**
- `reports/importancia_variaveis_final.csv`
- `images/importancia_variaveis_final.png`
- `src/evaluation/feature_importance_final.py`

### Síntese das respostas

As perguntas 1, 2, 3 e 5 foram abordadas com análises preditivas, territoriais e de agrupamento.

A pergunta 4 apresenta um diagnóstico de risco municipal baseado na distância entre o resultado observado em 2024 e a meta de 2025. Essa análise apoia a priorização de municípios e não deve ser confundida com uma previsão temporal validada.

Todas as conclusões respeitam as limitações dos dados e não atribuem causalidade a associações estatísticas.

As respostas também estão reunidas em [`reports/respostas_perguntas_fiap.md`](reports/respostas_perguntas_fiap.md).

## 22. Aplicação prática para políticas públicas

Os resultados do projeto podem apoiar o planejamento de ações
educacionais, principalmente na identificação de municípios que
merecem uma investigação mais detalhada.

A análise das metas de 2024 identificou 2.444 municípios abaixo
do objetivo estabelecido, entre 5.232 com dados válidos.
Esse diagnóstico pode ser utilizado como ponto de partida
para organizar prioridades de acompanhamento.

Entre as aplicações possíveis estão:

- **Acompanhamento municipal:** utilizar a distância entre a taxa
  observada e a meta para identificar municípios que precisam
  de uma análise mais aprofundada dos indicadores educacionais.
- **Planejamento regional:** comparar os resultados entre regiões
  para orientar estudos sobre diferenças territoriais e necessidades
  de apoio técnico.
- **Análise de participação:** investigar municípios com menor
  participação nas avaliações, verificando se os resultados
  disponíveis representam adequadamente os estudantes.
- **Identificação de perfis semelhantes:** utilizar os três grupos
  encontrados pelo K-Means para comparar municípios com
  características educacionais próximas e compartilhar experiências.
- **Contextualização socioeconômica:** considerar a renda municipal
  do IBGE como informação complementar na análise de desigualdades,
  sem utilizá-la isoladamente para definir intervenções.

O classificador individual também pode apoiar estudos exploratórios
sobre padrões de alfabetização. Entretanto, seu desempenho moderado
não permite recomendar decisões automatizadas sobre estudantes.

As aplicações descritas são possibilidades de uso dos resultados,
não políticas públicas implementadas ou impactos já comprovados.
Qualquer decisão de alocação de recursos deve considerar informações
adicionais, critérios de equidade e avaliação por profissionais
da área educacional.

## 23. Possíveis evoluções futuras

O projeto estabeleceu uma base para análises preditivas e territoriais,
mas ainda existem oportunidades de evolução.

As principais possibilidades são:

- **Ampliar a série histórica:** incorporar novos anos de indicadores
  educacionais para estudar tendências e mudanças ao longo do tempo.
- **Desenvolver previsão municipal temporal:** construir e validar
  um modelo específico para estimar o risco de não cumprimento
  de metas futuras, utilizando validação temporal apropriada.
- **Adicionar variáveis educacionais:** investigar informações sobre
  infraestrutura escolar, formação docente, características das
  escolas e condições de aprendizagem, quando disponíveis.
- **Aprofundar a análise socioeconômica:** testar indicadores
  complementares e verificar sua contribuição em diferentes
  regiões, sem confundir associações municipais com efeitos
  individuais.
- **Avaliar generalização e estabilidade:** repetir os estudos
  com novas amostras e períodos, verificando possíveis mudanças
  de desempenho e diferenças entre grupos.
- **Investigar hipóteses causais:** utilizar desenhos de pesquisa
  adequados e dados adicionais para avaliar possíveis efeitos
  de intervenções educacionais.
- **Criar visualizações para gestores:** desenvolver um painel
  para consulta de indicadores municipais, metas e perfis
  educacionais, respeitando os limites de interpretação dos dados.

Essas propostas representam etapas futuras e não funcionalidades
já implementadas nesta versão do projeto.

## 24. Considerações finais

O Tech Challenge integra engenharia de dados, classificação supervisionada, validação cruzada, interpretabilidade e análise territorial da alfabetização.

O trabalho reuniu previsão individual e diagnóstico territorial. O classificador apresentou desempenho moderado, enquanto as análises municipais permitiram identificar diferenças regionais, distâncias em relação às metas e perfis educacionais semelhantes.

O enriquecimento com renda do IBGE mostrou associação territorial com a alfabetização, mas não trouxe ganho consistente na previsão individual. Por isso, o modelo final foi preservado e os resultados socioeconômicos foram mantidos como análise descritiva, sem atribuição de causalidade.
