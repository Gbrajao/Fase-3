# Tech Challenge — Fase 3 | FIAP

## Predição e análise estratégica da alfabetização no Brasil

Projeto de Machine Learning desenvolvido na Fase 3 do Tech Challenge da FIAP, em continuidade ao tratamento e à consolidação de dados realizados na Fase 2.

O trabalho combina classificação supervisionada de alunos, análise de indicadores educacionais municipais, comparação de algoritmos, interpretabilidade, agrupamento de municípios e discussão de hipóteses causais.

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

A variável target_alfabetizado foi derivada da condição de alfabetização registrada na base de origem.

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

- src/analysis/audit_data.py
- src/analysis/eda_alfabetizacao.py
- src/analysis/eda_classes.py
- src/analysis/investigar_alunos.py

## 4. Engenharia de atributos e prevenção de vazamento

O modelo final utiliza quatro variáveis:

- rede: rede de ensino.
- sigla_uf: unidade federativa.
- regiao: região brasileira.
- taxa_alfabetizacao_2023: taxa municipal de alfabetização do ano anterior.

A utilização do indicador de 2023 evita empregar diretamente a taxa municipal de alfabetização de 2024 para prever resultados individuais referentes a 2024.

A proficiência individual e os indicadores contemporâneos derivados do resultado não são usados como preditores do modelo final.

A engenharia de atributos históricos está implementada em:

- src/preprocessing/add_historical_features.py

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

A Regressão Logística com balanceamento de classes foi mantida como modelo final por oferecer uma solução relativamente simples, interpretável e com desempenho competitivo nos experimentos realizados.

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

    [[254, 181],
     [248, 447]]

Arquivos relacionados:

- reports/metricas_teste_final.json
- images/matriz_confusao_teste_final.png

O desempenho do modelo é moderado. Seus resultados não justificam decisões automatizadas ou definitivas sobre a alfabetização de estudantes.

## 9. Interpretabilidade

Foi utilizada importância por permutação para investigar quais variáveis contribuem para o comportamento preditivo do modelo final.

A taxa municipal de alfabetização de 2023 apresentou a maior contribuição no experimento.

Também foram identificadas contribuições da unidade federativa, região e rede.

Arquivos:

- reports/importancia_variaveis_final.csv
- images/importancia_variaveis_final.png

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

- src/analysis/analise_risco_municipal.py
- reports/diagnostico_risco_municipal_2024.csv
- reports/resumo_risco_regional_2024.csv
- reports/relatorio_estrategico_municipal.md
- images/risco_municipal_por_regiao_2024.png

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

- src/analysis/clusterizacao_municipios.py
- reports/clusters_municipios_2024.csv
- reports/resumo_clusters_municipios_2024.csv
- reports/avaliacao_kmeans_silhouette.csv
- reports/interpretacao_clusters_municipais.md
- images/clusters_municipios_2024.png

## 12. Análise causal exploratória

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

- reports/analise_causal_alfabetizacao.md

## 13. Estrutura do projeto

    tech-challenge-fase3/
    ├── data/
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

## 14. Requisitos e instalação

Ambiente utilizado: Python 3.12.

Criar um ambiente virtual:

    python3.12 -m venv .venv

Ativar no macOS ou Linux:

    source .venv/bin/activate

Instalar as dependências:

    python -m pip install -r requirements.txt

Os comandos a seguir devem ser executados dentro da pasta tech-challenge-fase3.

## 15. Reprodução do pipeline

### Etapa 1 — Disponibilizar os dados

As bases das camadas Silver e Gold da Fase 2 devem estar presentes nos respectivos diretórios.

A preparação completa a partir das fontes originais depende dos dados e das etapas de engenharia desenvolvidas na Fase 2.

### Etapa 2 — Preparar a base supervisionada

    python src/preprocessing/prepare_ml_data.py

### Etapa 3 — Enriquecer a base de Machine Learning

    python src/preprocessing/enrich_ml_data.py

### Etapa 4 — Adicionar indicadores históricos

    python src/preprocessing/add_historical_features.py

### Etapa 5 — Separar treino, validação e teste

    python src/modeling/prepare_split.py

### Etapa 6 — Executar validação cruzada

    python src/evaluation/cross_validation.py

### Etapa 7 — Treinar o modelo final

    python src/modeling/train_final.py

### Etapa 8 — Avaliar o modelo final

    python src/evaluation/evaluate_final.py

A avaliação final utiliza o conjunto de teste reservado. Ela não deve ser repetida para escolher hiperparâmetros ou selecionar novos modelos com base nesse teste.

### Etapa 9 — Executar experimentos adicionais

    python src/modeling/train_gradient_boosting.py
    python src/modeling/train_participacao_historica.py
    python src/modeling/train_robust_scaler.py

### Etapa 10 — Executar análises municipais

    python src/analysis/analise_risco_municipal.py
    python src/analysis/clusterizacao_municipios.py
    python src/analysis/gerar_relatorio_estrategico.py

### Etapa 11 — Consultar os resultados

Os arquivos produzidos e documentados estão organizados principalmente em:

- reports/: métricas, tabelas e análises.
- images/: gráficos e visualizações.
- models/: modelos treinados.

Os scripts pressupõem que as bases e os arquivos intermediários necessários estejam disponíveis. A execução deve respeitar as dependências entre as etapas.

## 16. Limitações e uso responsável

- O classificador apresenta desempenho preditivo moderado.
- A amostra individual pode não representar a população brasileira de estudantes.
- Algumas variáveis socioeconômicas e escolares relevantes não estão disponíveis.
- Indicadores históricos municipais não substituem informações individuais de aprendizagem.
- A comparação com metas futuras não constitui previsão temporal validada.
- Correlações, importâncias preditivas e agrupamentos não demonstram causalidade.
- As análises devem apoiar investigação e decisões humanas, não rotular estudantes de forma definitiva.

## 17. Organização e versionamento

O código, os relatórios e as decisões metodológicas são organizados em diretórios específicos.

O projeto utiliza Git para controle de versão e GitHub como repositório remoto.

Repositório da Fase 3:

https://github.com/Gbrajao/Fase-3

As dependências estão documentadas em requirements.txt.

Os ambientes virtuais, arquivos temporários e informações sensíveis não devem ser versionados.

## 18. Continuidade da Fase 2

O projeto dá continuidade à organização dos dados e indicadores produzidos na Fase 2.

Repositório da Fase 2:

https://github.com/Gbrajao/Fase-2

## 19. Apresentação executiva

A entrega prevê uma apresentação em vídeo de até cinco minutos, destacando:

- Problema e objetivos.
- Dados e metodologia.
- Modelo final e resultados.
- Análises territoriais e agrupamentos.
- Principais limitações e recomendações.

O link do vídeo deverá ser acrescentado após sua publicação.

## 20. Considerações finais

O Tech Challenge integra engenharia de dados, classificação supervisionada, validação cruzada, interpretabilidade e análise territorial da alfabetização.

O projeto demonstra como métodos de Inteligência Artificial podem apoiar a identificação de padrões educacionais e prioridades de investigação, respeitando as limitações dos dados e evitando conclusões causais não sustentadas.
