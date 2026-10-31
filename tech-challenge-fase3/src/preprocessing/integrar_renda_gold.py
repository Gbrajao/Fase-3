from pathlib import Path

import pandas as pd


GOLD_ORIGINAL = Path("data/gold/municipio_enriquecido_ibge.parquet")
SILVER_RENDA = Path("data/silver/ibge_renda_municipal_2022.parquet")
GOLD_NOVA = Path("data/gold/municipio_enriquecido_socioeconomico.parquet")


def main():
    gold = pd.read_parquet(GOLD_ORIGINAL)
    renda = pd.read_parquet(SILVER_RENDA)

    chave = "id_municipio"
    chave_gold = ["ano", "id_municipio", "rede"]

    if gold.duplicated(subset=chave_gold).any():
        raise ValueError("Duplicidades na chave composta da Gold original.")

    if renda.duplicated(subset=[chave]).any():
        raise ValueError("Duplicidades de municípios na Silver.")

    if "renda_per_capita_2022" in gold.columns:
        raise ValueError("A Gold original já contém a coluna de renda.")

    gold[chave] = gold[chave].astype("string").str.strip()
    renda[chave] = renda[chave].astype("string").str.strip()

    colunas_originais = gold.columns.tolist()
    quantidade_original = len(gold)

    # A renda é de 2022 e será utilizada como característica
    # histórica dos municípios nos registros de 2023 e 2024.
    enriquecida = gold.merge(
        renda[[chave, "renda_per_capita_2022"]],
        on=chave,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    if len(enriquecida) != quantidade_original:
        raise ValueError("A integração alterou a quantidade de linhas.")

    if not enriquecida["_merge"].eq("both").all():
        raise ValueError("Existem municípios sem correspondência na Silver.")

    enriquecida = enriquecida.drop(columns=["_merge"])

    if enriquecida["renda_per_capita_2022"].isna().any():
        raise ValueError("Existem valores de renda ausentes na Gold.")

    if not enriquecida["renda_per_capita_2022"].gt(0).all():
        raise ValueError("Existem valores de renda não positivos na Gold.")

    if enriquecida.duplicated(subset=chave_gold).any():
        raise ValueError("A integração gerou duplicidades.")

    # Confere se os dados educacionais foram preservados.
    pd.testing.assert_frame_equal(
        enriquecida[colunas_originais].reset_index(drop=True),
        gold[colunas_originais].reset_index(drop=True),
        check_dtype=True,
    )

    GOLD_NOVA.parent.mkdir(parents=True, exist_ok=True)
    enriquecida.to_parquet(GOLD_NOVA, index=False)

    print("=== GOLD SOCIOECONÔMICA ===")
    print("Arquivo:", GOLD_NOVA)
    print("Registros originais:", quantidade_original)
    print("Registros após integração:", len(enriquecida))
    print("Municípios únicos:", enriquecida[chave].nunique())
    print("Rendas ausentes:", enriquecida["renda_per_capita_2022"].isna().sum())
    print("Duplicidades:", enriquecida.duplicated(subset=chave_gold).sum())
    print("Colunas originais preservadas:", len(colunas_originais))
    print()
    print("Renda per capita de 2022 (R$):")
    print(
        enriquecida.drop_duplicates(subset=[chave])[
            "renda_per_capita_2022"
        ].describe().round(2)
    )
    print()
    print("INTEGRAÇÃO VALIDADA COM SUCESSO!")


if __name__ == "__main__":
    main()
