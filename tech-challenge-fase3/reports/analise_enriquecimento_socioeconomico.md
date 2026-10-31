# Análise do enriquecimento socioeconômico com dados do IBGE

## 1. Objetivo

Investigar a relação entre renda municipal e alfabetização e avaliar se
a inclusão de informações socioeconômicas melhora o classificador
individual desenvolvido no Tech Challenge Fase 3.

## 2. Fonte e preparação dos dados

Foi utilizado o rendimento nominal médio mensal domiciliar per capita
do Censo Demográfico 2022, disponibilizado pelo IBGE na tabela
SIDRA 10295: https://sidra.ibge.gov.br/tabela/10295

O arquivo original foi preservado na camada Bronze. O tratamento
padronizou os códigos municipais, converteu os valores numéricos
e verificou valores ausentes, duplicados e rendas não positivas.

A base Silver contém 5.570 municípios, sem valores ausentes ou
não positivos. A integração à base Gold educacional contemplou
5.352 municípios, com cobertura de renda de 100% e sem perda
de registros.

A renda corresponde ao ano de 2022 e representa um indicador
municipal histórico, não a renda individual dos estudantes.
A fonte contém estimativas preliminares da amostra do Censo 2022.

## 3. Associação entre renda e alfabetização

Foram comparados os valores de renda de 2022 com as taxas
municipais de alfabetização de 2024 em 5.352 municípios.

- Correlação de Pearson: 0,1914.
- Correlação de Spearman: 0,2029.

Taxa média de alfabetização por quartil de renda:

| Quartil de renda | Alfabetização média |
|---|---:|
| 25% de menor renda | 56,91% |
| Intermediário inferior | 60,41% |
| Intermediário superior | 68,79% |
| 25% de maior renda | 66,05% |

As correlações de Pearson variaram por região:

| Região | Correlação |
|---|---:|
| Norte | 0,3399 |
| Sul | 0,0648 |
| Centro-Oeste | -0,0125 |
| Nordeste | -0,1179 |
| Sudeste | -0,1354 |

Os resultados indicam associação territorial fraca e heterogênea.
A comparação é ecológica e observacional; não demonstra
causalidade nem permite inferir a renda das famílias dos alunos.

## 4. Experimentos de classificação

Foram avaliadas três configurações de regressão logística
com cinco folds de validação cruzada agrupados por município.
Foram utilizados 5.289 alunos de treino e validação,
sem recorrer ao conjunto de teste reservado.

| Configuração | Acurácia balanceada | ROC-AUC | F1-macro | Recall classe 0 |
|---|---:|---:|---:|---:|
| Modelo histórico original | 0,5989 | 0,6445 | 0,5803 | 0,6344 |
| Modelo com renda | 0,5971 | 0,6423 | 0,5806 | 0,6190 |
| Modelo com renda e interação regional | 0,5983 | 0,6435 | 0,5786 | 0,6448 |

A inclusão da renda não trouxe melhoria consistente das métricas
gerais. A interação entre renda e região elevou o recall da classe 0,
mas não superou o modelo histórico nas métricas gerais.

## 5. Decisão técnica

O modelo histórico original foi mantido como modelo final.

O enriquecimento socioeconômico foi preservado como análise
complementar, pois acrescenta contexto territorial e permite
investigar desigualdades, embora não tenha produzido ganho
preditivo consistente na classificação individual.

Essa decisão foi baseada nos resultados de validação cruzada.
O conjunto de teste não foi reutilizado para selecionar modelos.

## 6. Arquivos relacionados

- data/bronze/ibge_renda_municipal_2022_original.csv
- data/silver/ibge_renda_municipal_2022.parquet
- data/gold/municipio_enriquecido_socioeconomico.parquet
- src/preprocessing/tratar_renda_ibge.py
- src/preprocessing/auditar_renda_ibge.py
- src/preprocessing/integrar_renda_gold.py
- src/modeling/train_socioeconomico.py
- src/modeling/train_interacao_renda_regiao.py
- reports/qualidade_renda_ibge_2022.json
- reports/validacao_cruzada_socioeconomico.csv
- reports/validacao_cruzada_interacao_renda_regiao.csv
- reports/conclusao_experimentos_renda_ibge.md
