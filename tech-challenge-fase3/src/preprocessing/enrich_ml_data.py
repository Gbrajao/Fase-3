
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

GOLD_PATH = BASE_DIR / "data" / "gold"
SILVER_PATH = BASE_DIR / "data" / "silver"

def main():
    alunos = pd.read_parquet(
        GOLD_PATH / "alunos_ml_base.parquet"
    )

    ibge = pd.read_parquet(
        SILVER_PATH / "ibge_municipios.parquet"
    )

    # Padronizar as chaves
    alunos["id_municipio"] = alunos["id_municipio"].astype(str)
    ibge["id_municipio"] = ibge["id_municipio"].astype(str)

    # Validar a unicidade da dimensão geográfica
    if ibge["id_municipio"].duplicated().any():
        raise ValueError(
            "Existem municípios duplicados na tabela IBGE."
        )

    colunas_ibge = [
        "id_municipio",
        "nome_municipio",
        "sigla_uf",
        "nome_uf",
        "regiao",
    ]

    # Integrar informações territoriais
    base_final = alunos.merge(
        ibge[colunas_ibge],
        on="id_municipio",
        how="left",
        validate="many_to_one",
    )

    # Garantir que não perdemos ou multiplicamos alunos
    if len(base_final) != len(alunos):
        raise ValueError(
            "A integração alterou o número de alunos."
        )

    print("=== ENRIQUECIMENTO TERRITORIAL ===")
    print(f"Registros antes: {len(alunos)}")
    print(f"Registros depois: {len(base_final)}")

    print("\nValores ausentes nas colunas geográficas:")
    print(
        base_final[
            ["nome_municipio", "sigla_uf", "nome_uf", "regiao"]
        ].isna().sum()
    )

    print("\nDistribuição por região:")
    print(
        base_final["regiao"].value_counts(dropna=False)
    )

    # Salvar resultado
    destino = GOLD_PATH / "alunos_ml_enriquecido.parquet"
    base_final.to_parquet(destino, index=False)

    print(f"\nArquivo gerado: {destino}")


if __name__ == "__main__":
    main()
