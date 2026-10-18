# Relatório estratégico — Alfabetização municipal

## 1. Objetivo

Analisar os indicadores de alfabetização da rede municipal em 2024,
identificar municípios abaixo das metas e apoiar a priorização de
ações educacionais.

## 2. Base utilizada

Fonte analítica: camada Gold do projeto, arquivo
`municipio_enriquecido_ibge.parquet`.

Foram considerados 5,352 municípios, com dados referentes
ao ano de 2024 e à rede Municipal.

Os indicadores municipais são distintos das taxas calculadas
sobre a amostra de alunos utilizada no treinamento do modelo.

## 3. Situação da meta de 2024

- Meta atingida: 2,788 municípios.
- Meta não atingida: 2,444 municípios.
- Sem dados suficientes: 120 municípios.
- Municípios avaliáveis: 5,232.
- Percentual abaixo da meta entre os avaliáveis:
  46.71%.

A classificação foi calculada pela diferença entre o indicador
de alfabetização de 2024 e a meta municipal de 2024.

Municípios sem indicador ou meta não foram classificados
como descumpridores.

## 4. Distância até a meta de 2025

Foram encontrados 5,352 municípios com indicador
de 2024 e meta de 2025 disponíveis.

Destes, 3,030 (56.61%)
apresentaram indicador de 2024 inferior à meta de 2025.

Essa comparação representa uma distância entre o desempenho
observado em 2024 e a meta de 2025. Não é uma previsão
estatística de descumprimento futuro.

## 5. Comparação regional

| Região | Municípios | Abaixo da meta 2025 | Percentual | Distância média (p.p.) |
|---|---:|---:|---:|---:|
| Norte | 425 | 317 | 74.59% | +6.00 |
| Sul | 1093 | 768 | 70.27% | +8.90 |
| Nordeste | 1756 | 1086 | 61.85% | +3.32 |
| Sudeste | 1614 | 693 | 42.94% | -2.76 |
| Centro-Oeste | 464 | 166 | 35.78% | -4.00 |

A distância média positiva significa que a meta de 2025
está acima do indicador de 2024, em média. Uma distância
negativa indica a situação inversa.

As médias são calculadas entre municípios, sem ponderação
pela população de alunos.

## 6. Municípios com maior distância até a meta de 2025

| Município | UF | Indicador 2024 | Meta 2025 | Distância (p.p.) |
|---|---|---:|---:|---:|
| Sério | RS | 11.10% | 80.00% | 68.90 |
| Arroio do Padre | RS | 18.20% | 80.00% | 61.80 |
| São Nicolau | RS | 7.10% | 65.10% | 58.00 |
| Jaboticaba | RS | 25.00% | 80.00% | 55.00 |
| São Vendelino | RS | 25.00% | 80.00% | 55.00 |
| Chapada de Areia | TO | 6.25% | 59.77% | 53.52 |
| Monte do Carmo | TO | 19.71% | 73.10% | 53.39 |
| Dona Francisca | RS | 26.70% | 80.00% | 53.30 |
| Bozano | RS | 27.30% | 80.00% | 52.70 |
| Sentinela do Sul | RS | 23.50% | 75.70% | 52.20 |

Essa lista serve como ponto de partida para investigação.
Não constitui ranking definitivo de qualidade educacional.

## 7. Recomendações para gestores públicos

1. Priorizar o diagnóstico dos municípios com maiores
   distâncias em relação às metas, verificando também
   participação, cobertura e qualidade dos dados.

2. Analisar as diferenças regionais antes de definir
   políticas uniformes para todo o país.

3. Investigar fatores locais, como infraestrutura escolar,
   formação docente, frequência e condições socioeconômicas,
   antes de atribuir causas aos resultados.

4. Acompanhar os indicadores periodicamente e comparar
   os resultados efetivamente observados com as metas.

5. Utilizar o modelo de Machine Learning como apoio
   exploratório, nunca como critério único para decisões
   sobre alunos ou distribuição de recursos.

## 8. Limitações

- A análise municipal é descritiva e utiliza dados de 2024.
- Não representa uma previsão do indicador municipal de 2025.
- A avaliação da meta de 2024 exige indicador e meta de 2024.
- A comparação com 2025 exige indicador de 2024 e meta de 2025;
  por isso, os totais avaliáveis podem ser diferentes.
- As médias regionais não são ponderadas pela população escolar.
- Diferenças entre indicadores e metas não demonstram causalidade.
- Municípios com valores extremos exigem verificação contextual.
- Os resultados da amostra de alunos não devem ser generalizados
  automaticamente para toda a população municipal.

## 9. Arquivos de apoio

- `reports/diagnostico_risco_municipal_2024.csv`
- `reports/resumo_risco_regional_2024.csv`
- `images/risco_municipal_por_regiao_2024.png`

