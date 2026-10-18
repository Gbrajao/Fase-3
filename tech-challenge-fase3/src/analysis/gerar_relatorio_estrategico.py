from pathlib import Path
import pandas as pd

PASTA = Path("reports")
PASTA.mkdir(exist_ok=True)

df = pd.read_csv(
    PASTA / "diagnostico_risco_municipal_2024.csv",
    dtype={"id_municipio": str}
)

total = len(df)

situacoes = df["situacao_meta_2024"].value_counts()
atingiram = int(situacoes.get("Meta atingida", 0))
nao_atingiram = int(situacoes.get("Meta não atingida", 0))
sem_dados = int(situacoes.get("Sem dados suficientes", 0))

validos_2024 = atingiram + nao_atingiram
percentual_nao_atingiram = (
    100 * nao_atingiram / validos_2024
    if validos_2024 else 0
)

validos_2025 = df.dropna(
    subset=["taxa_alfabetizacao", "meta_alfabetizacao_2025"]
).copy()

abaixo_2025 = validos_2025[
    validos_2025["distancia_meta_2025"] > 0
]

percentual_abaixo_2025 = (
    100 * len(abaixo_2025) / len(validos_2025)
    if len(validos_2025) else 0
)

regioes = (
    validos_2025.groupby("regiao")
    .agg(
        municipios=("id_municipio", "nunique"),
        distancia_media=("distancia_meta_2025", "mean"),
        abaixo_meta_2025=(
            "distancia_meta_2025",
            lambda x: int((x > 0).sum())
        )
    )
    .reset_index()
)

regioes["percentual_abaixo"] = (
    100 * regioes["abaixo_meta_2025"] / regioes["municipios"]
)

regioes = regioes.sort_values(
    "percentual_abaixo",
    ascending=False
)

linhas_tabela = [
    "| Região | Municípios | Abaixo da meta 2025 | Percentual | Distância média (p.p.) |",
    "|---|---:|---:|---:|---:|",
]

for _, linha in regioes.iterrows():
    linhas_tabela.append(
        f"| {linha['regiao']} "
        f"| {int(linha['municipios'])} "
        f"| {int(linha['abaixo_meta_2025'])} "
        f"| {linha['percentual_abaixo']:.2f}% "
        f"| {linha['distancia_media']:+.2f} |"
    )

maiores_distancias = validos_2025.nlargest(
    10, "distancia_meta_2025"
)

linhas_municipios = [
    "| Município | UF | Indicador 2024 | Meta 2025 | Distância (p.p.) |",
    "|---|---|---:|---:|---:|",
]

for _, linha in maiores_distancias.iterrows():
    linhas_municipios.append(
        f"| {linha['nome_municipio']} "
        f"| {linha['sigla_uf']} "
        f"| {linha['taxa_alfabetizacao']:.2f}% "
        f"| {linha['meta_alfabetizacao_2025']:.2f}% "
        f"| {linha['distancia_meta_2025']:.2f} |"
    )

conteudo = f"""# Relatório estratégico — Alfabetização municipal

## 1. Objetivo

Analisar os indicadores de alfabetização da rede municipal em 2024,
identificar municípios abaixo das metas e apoiar a priorização de
ações educacionais.

## 2. Base utilizada

Fonte analítica: camada Gold do projeto, arquivo
`municipio_enriquecido_ibge.parquet`.

Foram considerados {total:,} municípios, com dados referentes
ao ano de 2024 e à rede Municipal.

Os indicadores municipais são distintos das taxas calculadas
sobre a amostra de alunos utilizada no treinamento do modelo.

## 3. Situação da meta de 2024

- Meta atingida: {atingiram:,} municípios.
- Meta não atingida: {nao_atingiram:,} municípios.
- Sem dados suficientes: {sem_dados:,} municípios.
- Municípios avaliáveis: {validos_2024:,}.
- Percentual abaixo da meta entre os avaliáveis:
  {percentual_nao_atingiram:.2f}%.

A classificação foi calculada pela diferença entre o indicador
de alfabetização de 2024 e a meta municipal de 2024.

Municípios sem indicador ou meta não foram classificados
como descumpridores.

## 4. Distância até a meta de 2025

Foram encontrados {len(validos_2025):,} municípios com indicador
de 2024 e meta de 2025 disponíveis.

Destes, {len(abaixo_2025):,} ({percentual_abaixo_2025:.2f}%)
apresentaram indicador de 2024 inferior à meta de 2025.

Essa comparação representa uma distância entre o desempenho
observado em 2024 e a meta de 2025. Não é uma previsão
estatística de descumprimento futuro.

## 5. Comparação regional

{chr(10).join(linhas_tabela)}

A distância média positiva significa que a meta de 2025
está acima do indicador de 2024, em média. Uma distância
negativa indica a situação inversa.

As médias são calculadas entre municípios, sem ponderação
pela população de alunos.

## 6. Municípios com maior distância até a meta de 2025

{chr(10).join(linhas_municipios)}

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

"""

destino = PASTA / "relatorio_estrategico_municipal.md"
destino.write_text(conteudo, encoding="utf-8")

print("=" * 60)
print("RELATÓRIO ESTRATÉGICO MUNICIPAL")
print("=" * 60)
print(f"Arquivo gerado: {destino}")
print(f"Municípios analisados: {total:,}")
print(f"Avaliáveis para a meta de 2024: {validos_2024:,}")
print(f"Avaliáveis para a meta de 2025: {len(validos_2025):,}")
print("Relatório concluído.")
