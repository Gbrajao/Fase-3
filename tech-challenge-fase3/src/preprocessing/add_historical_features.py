
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
GOLD = BASE_DIR / "data" / "gold"


def main():
    alunos = pd.read_parquet(
        GOLD / "alunos_ml_enriquecido.parquet"
    )

    municipios = pd.read_parquet(
        GOLD / "municipio_enriquecido_ibge.parquet"
    )

    historico = municipios.loc[
        (municipios["ano"] == 2023)
        & (municipios["rede"] == "Municipal"),
        ["id_municipio", "taxa_alfabetizacao"]
    ].rename(columns={
        "taxa_alfabetizacao": "taxa_alfabetizacao_2023"
    })

    resultado = alunos.merge(
        historico,
        on="id_municipio",
        how="left",
        validate="many_to_one"
    )

    assert len(resultado) == len(alunos), (
        "A integração alterou a quantidade de alunos."
    )

    assert (resultado["ano"] == 2024).all(), (
        "Foram encontrados alunos de outro ano."
    )

    destino = GOLD / "alunos_ml_historico.parquet"
    resultado.to_parquet(destino, index=False)

    print("=== BASE ENRIQUECIDA ===")
    print("Total de alunos:", len(resultado))
    print("Total de colunas:", len(resultado.columns))
    print(
        "Alunos com histórico:",
        resultado["taxa_alfabetizacao_2023"].notna().sum()
    )
    print(
        "Alunos sem histórico:",
        resultado["taxa_alfabetizacao_2023"].isna().sum()
    )
    print(f"\nArquivo salvo em: {destino}")


if __name__ == "__main__":
    main()
