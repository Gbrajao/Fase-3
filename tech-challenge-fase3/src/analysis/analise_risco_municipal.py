from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO = Path("data/gold/municipio_enriquecido_ibge.parquet")
PASTA_RELATORIOS = Path("reports")
PASTA_IMAGENS = Path("images")

PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ANÁLISE ESTRATÉGICA MUNICIPAL — INDICADORES DE 2024")
print("=" * 65)

df = pd.read_parquet(ARQUIVO)

df = df.loc[
    (df["ano"] == 2024) &
    (df["rede"] == "Municipal")
].copy()

if df["id_municipio"].duplicated().any():
    raise ValueError(
        "Existem municípios duplicados em 2024. "
        "Verifique a granularidade da base."
    )

print(f"\nMunicípios encontrados: {len(df):,}")

colunas = [
    "id_municipio",
    "nome_municipio",
    "sigla_uf",
    "regiao",
    "taxa_alfabetizacao",
    "meta_alfabetizacao_2024",
    "meta_alfabetizacao_2025",
    "percentual_participacao",
]

analise = df[colunas].copy()

analise["diferenca_meta_2024"] = (
    analise["taxa_alfabetizacao"]
    - analise["meta_alfabetizacao_2024"]
)

analise["distancia_meta_2025"] = (
    analise["meta_alfabetizacao_2025"]
    - analise["taxa_alfabetizacao"]
)

analise["situacao_meta_2024"] = "Sem dados suficientes"

dados_validos = (
    analise["taxa_alfabetizacao"].notna()
    & analise["meta_alfabetizacao_2024"].notna()
)

analise.loc[
    dados_validos & (analise["diferenca_meta_2024"] < 0),
    "situacao_meta_2024"
] = "Meta não atingida"

analise.loc[
    dados_validos & (analise["diferenca_meta_2024"] >= 0),
    "situacao_meta_2024"
] = "Meta atingida"

arquivo_municipios = (
    PASTA_RELATORIOS / "diagnostico_risco_municipal_2024.csv"
)

analise.sort_values(
    ["diferenca_meta_2024", "nome_municipio"],
    na_position="last"
).to_csv(
    arquivo_municipios,
    index=False,
    encoding="utf-8-sig"
)

print("\nSITUAÇÃO DAS METAS DE 2024:")
print(
    analise["situacao_meta_2024"]
    .value_counts()
    .to_string()
)

validos = analise.loc[dados_validos].copy()

resumo_regional = (
    validos.groupby("regiao")
    .agg(
        municipios_avaliados=("id_municipio", "nunique"),
        taxa_media_2024=("taxa_alfabetizacao", "mean"),
        diferenca_media_meta_2024=("diferenca_meta_2024", "mean"),
        municipios_abaixo_meta=(
            "situacao_meta_2024",
            lambda serie: (serie == "Meta não atingida").sum()
        ),
    )
    .reset_index()
)

resumo_regional["percentual_abaixo_meta"] = (
    100
    * resumo_regional["municipios_abaixo_meta"]
    / resumo_regional["municipios_avaliados"]
)

resumo_regional = resumo_regional.sort_values(
    "percentual_abaixo_meta",
    ascending=False
)

arquivo_regioes = (
    PASTA_RELATORIOS / "resumo_risco_regional_2024.csv"
)

resumo_regional.to_csv(
    arquivo_regioes,
    index=False,
    encoding="utf-8-sig"
)

print("\nRESUMO POR REGIÃO:")
print(
    resumo_regional.round(2).to_string(index=False)
)

print("\n10 MUNICÍPIOS COM MAIOR DISTÂNCIA ABAIXO DA META DE 2024:")

piores = validos.nsmallest(
    10,
    "diferenca_meta_2024"
)

print(
    piores[
        [
            "nome_municipio",
            "sigla_uf",
            "taxa_alfabetizacao",
            "meta_alfabetizacao_2024",
            "diferenca_meta_2024",
        ]
    ].round(2).to_string(index=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    resumo_regional["regiao"],
    resumo_regional["percentual_abaixo_meta"]
)

ax.invert_yaxis()
ax.set_xlabel("Municípios abaixo da meta de 2024 (%)")
ax.set_title(
    "Municípios abaixo da meta de alfabetização de 2024\n"
    "Por região — rede municipal"
)

ax.set_xlim(
    0,
    max(10, resumo_regional["percentual_abaixo_meta"].max() * 1.15)
)

for i, valor in enumerate(
    resumo_regional["percentual_abaixo_meta"]
):
    ax.text(
        valor + 0.5,
        i,
        f"{valor:.1f}%",
        va="center"
    )

plt.tight_layout()

arquivo_grafico = (
    PASTA_IMAGENS / "risco_municipal_por_regiao_2024.png"
)

plt.savefig(
    arquivo_grafico,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nARQUIVOS GERADOS:")
print(f" - {arquivo_municipios}")
print(f" - {arquivo_regioes}")
print(f" - {arquivo_grafico}")

print("\nOBSERVAÇÕES:")
print("- Análise descritiva baseada nos indicadores municipais de 2024.")
print("- Não representa previsões individuais do modelo de ML.")
print("- Distância até a meta de 2025 não é previsão de descumprimento.")
print("- Municípios sem indicador ou meta não foram classificados.")

print("\nExecução concluída.")
