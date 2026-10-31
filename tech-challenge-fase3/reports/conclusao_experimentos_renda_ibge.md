# Conclusão dos experimentos com renda municipal — IBGE 2022

## Objetivo

Investigar a associação entre renda municipal e alfabetização e avaliar se a renda domiciliar per capita de 2022 melhora a previsão individual de alfabetização.

## Associação territorial

Foram analisados 5.352 municípios com indicadores educacionais de 2024 e renda per capita de 2022.

- Correlação de Pearson: 0,1914.
- Correlação de Spearman: 0,2029.

Taxa média de alfabetização por quartil de renda:

| Quartil de renda | Alfabetização média |
|---|---:|
| Menor renda | 56,91% |
| Intermediária baixa | 60,41% |
| Intermediária alta | 68,79% |
| Maior renda | 66,05% |

Correlações de Pearson por região:

| Região | Correlação |
|---|---:|
| Centro-Oeste | -0,0125 |
| Nordeste | -0,1179 |
| Norte | 0,3399 |
| Sudeste | -0,1354 |
| Sul | 0,0648 |

As associações são descritivas e não demonstram causalidade.

## Comparação dos modelos

Validação cruzada com cinco folds agrupados por município, utilizando 5.289 alunos de treino e validação.

| Modelo | Balanced accuracy | ROC-AUC | F1-macro | Recall classe 0 |
|---|---:|---:|---:|---:|
| Regressão logística atual | 0,5989 | 0,6445 | 0,5803 | 0,6344 |
| Regressão logística com renda | 0,5971 | 0,6423 | 0,5806 | 0,6190 |
| Regressão logística com interação renda × região | 0,5983 | 0,6435 | 0,5786 | 0,6448 |

O conjunto de teste não foi utilizado nesses experimentos.

## Interpretação

A renda municipal apresenta associação positiva fraca com a alfabetização municipal, com padrões distintos entre regiões.

A inclusão da renda, isoladamente ou em interação com a região, não produziu ganho consistente nas métricas gerais de previsão individual.

As diferenças são pequenas e não permitem concluir ausência de relevância socioeconômica ou de efeitos causais.

## Decisão

Manter a regressão logística histórica como modelo final.

Preservar os dados de renda nas camadas Bronze, Silver e Gold e utilizar os resultados na análise estratégica e na documentação do projeto.

## Arquivos de evidência

- `reports/qualidade_renda_ibge_2022.json`
- `reports/validacao_cruzada_socioeconomico.csv`
- `reports/validacao_cruzada_interacao_renda_regiao.csv`
- `data/gold/municipio_enriquecido_socioeconomico.parquet`
