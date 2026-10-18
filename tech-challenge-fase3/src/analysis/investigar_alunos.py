
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CAMINHO = BASE_DIR / "data/silver/alunos.parquet"

df = pd.read_parquet(CAMINHO)

print("\n=== DISTRIBUIÇÃO DE ALFABETIZAÇÃO ===")
print(df["alfabetizado"].value_counts(dropna=False))

print("\n=== DISTRIBUIÇÃO DE PRESENÇA ===")
print(df["presenca"].value_counts(dropna=False))

print("\n=== PRESENÇA X ALFABETIZAÇÃO ===")
print(pd.crosstab(
    df["presenca"],
    df["alfabetizado"],
    dropna=False
))

print("\n=== PROFICIÊNCIA POR ALFABETIZAÇÃO ===")
print(df.groupby("alfabetizado")["proficiencia"].agg(
    ["count", "min", "mean", "max"]
))

print("\n=== PROFICIÊNCIA AUSENTE POR PRESENÇA ===")
print(pd.crosstab(
    df["presenca"],
    df["proficiencia"].isna()
))

print("\n=== REGRA DE 743 PONTOS ===")
avaliados = df[
    df["proficiencia"].notna()
].copy()

avaliados["regra_743"] = avaliados["proficiencia"].ge(743)

print(pd.crosstab(
    avaliados["regra_743"],
    avaliados["alfabetizado"]
))

print("\n=== DISTRIBUIÇÃO POR REDE ===")
print(df["rede"].value_counts(dropna=False))

print("\n=== MUNICÍPIOS REPRESENTADOS ===")
print(df["id_municipio"].nunique())
