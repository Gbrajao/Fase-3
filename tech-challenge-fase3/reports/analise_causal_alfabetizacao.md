# Análise causal exploratória — Alfabetização

## 1. Pergunta de pesquisa

Uma maior participação dos alunos nas avaliações educacionais
causa uma melhoria na taxa de alfabetização municipal?

## 2. Hipótese causal

Hipótese: políticas que ampliam a participação dos alunos
nas avaliações podem estar associadas a melhores condições
de acompanhamento pedagógico e, indiretamente, a melhores
resultados de alfabetização.

Essa hipótese não foi comprovada pelos dados disponíveis.

## 3. Variáveis de interesse

- Exposição: percentual de participação na avaliação.
- Desfecho: taxa municipal de alfabetização.
- Unidade de análise: município.
- Período disponível: 2023 e 2024.

## 4. Possíveis fatores de confusão

Variáveis que podem influenciar simultaneamente a participação
e os resultados educacionais:

- Condições socioeconômicas das famílias.
- Investimento municipal em educação.
- Infraestrutura e recursos escolares.
- Formação e disponibilidade de professores.
- Localização geográfica e características territoriais.
- Políticas educacionais implementadas anteriormente.

Esses fatores não estão adequadamente mensurados
na base analítica atual.

## 5. Evidências observacionais

Na análise municipal de 2023, a correlação de Pearson
entre participação e taxa de alfabetização foi 0,287.

O resultado indica associação linear positiva de
magnitude limitada.

O agrupamento K-Means de 2024 também identificou
municípios com alta participação e baixa alfabetização,
mostrando que os indicadores não variam necessariamente
na mesma direção.

Essas evidências não permitem concluir que aumentar
a participação provoque aumento da alfabetização.

## 6. Por que não é possível afirmar causalidade?

A base é observacional e não contém uma intervenção
claramente identificada nem uma estratégia de
identificação causal validada.

Além disso:

- Existem possíveis confundidores não observados.
- Pode haver causalidade reversa.
- A participação na avaliação pode afetar a composição
  da população efetivamente avaliada.
- Há somente dois anos de indicadores municipais.
- Não foi demonstrada comparabilidade entre grupos
  expostos e não expostos a uma intervenção.

Portanto, uma regressão simples entre participação e
alfabetização não seria suficiente para estimar um
efeito causal confiável.

## 7. Proposta para investigação futura

Uma pesquisa futura poderia avaliar uma política
educacional específica, com:

1. Definição clara da intervenção e da data de implantação.
2. Identificação de municípios tratados e comparáveis.
3. Dados anteriores e posteriores à intervenção.
4. Indicadores socioeconômicos e escolares adicionais.
5. Verificação das premissas de um desenho causal,
   como diferenças-em-diferenças, quando aplicável.
6. Testes de robustez e análises de sensibilidade.

A escolha do método dependeria da natureza da política,
da qualidade dos dados e das hipóteses de identificação.

## 8. Conclusão

O projeto apresenta evidências descritivas e preditivas
sobre alfabetização, mas não demonstra efeitos causais.

A importância de variáveis em modelos de machine learning,
as correlações e os clusters não devem ser interpretados
como provas de causalidade.

A contribuição desta análise é formular uma hipótese
causal verificável, explicitar suas limitações e propor
um caminho metodológico para estudos futuros.
