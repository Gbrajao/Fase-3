# Análise de agrupamento de municípios — K-Means

## Objetivo

Identificar municípios com características educacionais semelhantes
a partir de indicadores oficiais municipais de 2024.

Esta análise é descritiva e não faz parte do treinamento do
classificador individual de alfabetização.

## Metodologia

- Unidade de análise: município.
- Total analisado: 5.352 municípios.
- Variáveis: taxa de alfabetização e percentual de participação.
- Pré-processamento: padronização com StandardScaler.
- Algoritmo: K-Means.
- Quantidade de grupos avaliada: 2 a 6.
- Critério de escolha: maior índice silhouette.
- Melhor resultado: 3 grupos, silhouette de 0,3842.

## Perfis identificados

### Cluster 0 — Baixa alfabetização, alta participação

- Municípios: 1.642.
- Taxa média de alfabetização: 46,73%.
- Participação média: 93,61%.

Esse grupo apresenta a menor taxa média de alfabetização,
apesar da participação elevada na avaliação.

Implicação estratégica: priorizar a investigação de dificuldades
de aprendizagem e necessidades de apoio pedagógico.

### Cluster 2 — Alfabetização intermediária, menor participação

- Municípios: 1.365.
- Taxa média de alfabetização: 53,98%.
- Participação média: 82,64%.

Esse grupo combina resultados intermediários de alfabetização
com menor participação na avaliação.

Implicação estratégica: investigar barreiras de participação,
cobertura da avaliação e possíveis diferenças de composição
entre alunos avaliados e não avaliados.

### Cluster 1 — Alta alfabetização, alta participação

- Municípios: 2.345.
- Taxa média de alfabetização: 79,74%.
- Participação média: 94,10%.

Esse grupo apresenta os melhores indicadores médios.

Implicação estratégica: investigar práticas e condições locais
potencialmente associadas aos resultados, sem presumir causalidade.

## Limitações

O índice silhouette de 0,3842 indica separação moderada entre grupos.

Os clusters dependem das variáveis escolhidas e do método de
padronização. Não representam classificações oficiais.

As diferenças observadas são associações descritivas.
Não demonstram relações de causa e efeito.

## Arquivos produzidos

- reports/clusters_municipios_2024.csv
- reports/resumo_clusters_municipios_2024.csv
- reports/avaliacao_kmeans_silhouette.csv
- images/clusters_municipios_2024.png
