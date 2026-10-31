from pathlib import Path

import pandas as pd


BRONZE = Path("data/bronze/ibge_renda_municipal_2022_original.csv")
SILVER = Path("data/silver/ibge_renda_municipal_2022.parquet")


def main():
    if not BRONZE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {BRONZE}")

    # O arquivo SIDRA contém duas linhas descritivas e quatro de cabeçalho.
    df = pd.read_csv(
        BRONZE,
        sep=";",
        skiprows=6,
        header=None,
        names=[
            "id_municipio",
            "nome_municipio_ibge",
            "grupo_idade",
            "renda_per_capita_2022",
        ],
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="error",
    )

    # Mantém apenas registros com códigos municipais de sete dígitos.
    df["id_municipio"] = df["id_municipio"].str.strip()

    registros_validos = df["id_municipio"].str.fullmatch(
        r"\d{7}", na=False
    )

    df = df.loc[registros_validos].copy()

    # A seleção do SIDRA deve conter apenas o grupo etário Total.
    if not df["grupo_idade"].eq("Total").all():
        raise ValueError("Existem grupos de idade diferentes de Total.")

    df["renda_per_capita_2022"] = pd.to_numeric(
        df["renda_per_capita_2022"]
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df["ano_referencia"] = 2022

    if df["id_municipio"].duplicated().any():
        raise ValueError("Foram encontrados municípios duplicados.")

    if len(df) != 5570:
        raise ValueError(
            f"Esperados 5570 municípios; encontrados {len(df)}."
        )

    ausentes = df["renda_per_capita_2022"].isna().sum()

    if ausentes:
        raise ValueError(
            f"Foram encontrados {ausentes} valores de renda ausentes."
        )

    if (df["renda_per_capita_2022"] < 0).any():
        raise ValueError("Foram encontrados valores negativos de renda.")

    df = df[
        [
            "ano_referencia",
            "id_municipio",
            "nome_municipio_ibge",
            "renda_per_capita_2022",
        ]
    ].sort_values("id_municipio").reset_index(drop=True)

    SILVER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER, index=False)

    print("Camada Silver criada com sucesso!")
    print("Arquivo:", SILVER)
    print("Municípios:", len(df))
    print("Códigos únicos:", df["id_municipio"].nunique())
    print("Valores ausentes:", ausentes)
    print()
    print("Resumo da renda per capita (R$):")
    print(df["renda_per_capita_2022"].describe().round(2))
    print()
    print("Primeiros registros:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
