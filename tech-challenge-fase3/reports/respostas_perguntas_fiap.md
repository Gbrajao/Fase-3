# Respostas às perguntas estratégicas da FIAP

## Perguntas estratégicas da FIAP — Respostas e evidências

Esta seção responde diretamente às cinco perguntas estratégicas
apresentadas no Tech Challenge da Fase 3.

### 1. Quais fatores mais impactam a alfabetização?

Na análise preditiva realizada, a variável com maior contribuição
para o classificador final foi a taxa municipal de alfabetização
de 2023.

A importância por permutação também identificou contribuições
da unidade federativa, da região e da rede de ensino.

Esses resultados indicam quais informações ajudam o modelo a
distinguir alunos alfabetizados e não alfabetizados.

Entretanto, importância preditiva não equivale a impacto causal.
Não é possível concluir que modificar essas variáveis provocaria,
necessariamente, uma mudança na alfabetização.

**Evidências:**
- `reports/importancia_variaveis_final.csv`
- `images/importancia_variaveis_final.png`
- `reports/analise_causal_alfabetizacao.md`

### 2. Quais municípios apresentam maior risco educacional?

O projeto identificou municípios que estavam abaixo das metas
educacionais de alfabetização em 2024.

Dos 5.232 municípios com informações suficientes para essa
comparação, 2.444 não atingiram a meta de 2024, correspondendo
a 46,71% dos municípios com dados válidos.

Outros 2.788 atingiram a meta e 120 não possuíam informações
suficientes para essa classificação.

A análise também mostrou diferenças regionais. Entre os municípios
com dados válidos, o Sul apresentou 66,13% abaixo da meta de 2024,
seguido pelo Norte, com 56,30%.

O diagnóstico municipal permite localizar os municípios com
maiores distâncias negativas em relação à meta, apoiando a
priorização de análises e ações educacionais.

**Evidências:**
- `reports/diagnostico_risco_municipal_2024.csv`
- `reports/resumo_risco_regional_2024.csv`
- `reports/relatorio_estrategico_municipal.md`
- `images/risco_municipal_por_regiao_2024.png`

### 3. Quais regiões possuem padrões semelhantes?

Foram realizadas comparações regionais e uma análise de
agrupamento K-Means com 5.352 municípios.

O agrupamento utilizou a taxa de alfabetização e o percentual
de participação nas avaliações de 2024.

Foram identificados três perfis:

- **Cluster 0:** 1.642 municípios, alfabetização média de
  46,73% e participação média de 93,61%.
- **Cluster 2:** 1.365 municípios, alfabetização média de
  53,98% e participação média de 82,64%.
- **Cluster 1:** 2.345 municípios, alfabetização média de
  79,74% e participação média de 94,10%.

O melhor índice silhouette foi aproximadamente 0,3842,
indicando separação moderada entre os grupos.

Os clusters agrupam municípios, e não regiões inteiras.
Por isso, municípios de diferentes regiões podem apresentar
características educacionais semelhantes.

**Evidências:**
- `reports/resumo_risco_regional_2024.csv`
- `reports/resumo_clusters_municipios_2024.csv`
- `reports/clusters_municipios_2024.csv`
- `reports/interpretacao_clusters_municipais.md`
- `images/clusters_municipios_2024.png`

### 4. Como prever municípios que podem não atingir metas futuras?

O projeto realizou uma análise de distância entre a taxa de
alfabetização observada em 2024 e a meta estabelecida para 2025.

Dos 5.352 municípios analisados, 3.030 apresentavam taxa de
alfabetização de 2024 inferior à meta prevista para 2025,
equivalendo a aproximadamente 56,61%.

Essa informação permite identificar municípios que precisariam
melhorar seus indicadores para alcançar a meta seguinte.

Entretanto, **não foi desenvolvido nem validado um modelo
temporal específico para prever o resultado municipal de 2025**.

Portanto, os 3.030 municípios identificados não devem ser
apresentados como municípios que certamente deixarão de
atingir a meta.

Uma previsão futura mais robusta exigiria séries históricas
mais extensas, variáveis explicativas adicionais e validação
temporal apropriada.

**Evidências:**
- `reports/diagnostico_risco_municipal_2024.csv`
- `reports/relatorio_estrategico_municipal.md`

### 5. Quais variáveis possuem maior influência nos modelos?

No classificador final, a importância por permutação foi
calculada utilizando a acurácia balanceada como métrica.

As contribuições médias observadas foram:

| Variável | Importância média aproximada |
|---|---:|
| Taxa de alfabetização municipal de 2023 | 0,0708 |
| Unidade federativa | 0,0215 |
| Região | 0,0105 |
| Rede de ensino | 0,0063 |

A taxa histórica municipal foi a variável com maior
importância preditiva no experimento.

A análise foi realizada sobre os dados de treino e validação,
não constituindo uma avaliação independente no teste.

Os valores indicam a redução média da acurácia balanceada
quando cada variável é embaralhada. Não representam
percentuais de causalidade ou influência direta sobre
a aprendizagem.

**Evidências:**
- `reports/importancia_variaveis_final.csv`
- `images/importancia_variaveis_final.png`
- `src/evaluation/feature_importance_final.py`

### Síntese das respostas

As perguntas 1, 2, 3 e 5 foram abordadas com análises
preditivas, territoriais e de agrupamento.

A pergunta 4 foi respondida parcialmente por uma análise
descritiva de distância até metas futuras, sem previsão
temporal validada.

Todas as conclusões respeitam as limitações dos dados
e não atribuem causalidade a associações estatísticas.
