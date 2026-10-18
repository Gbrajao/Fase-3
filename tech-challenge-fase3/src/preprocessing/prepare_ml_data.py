
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

SILVER_PATH = BASE_DIR / "data" / "silver"
GOLD_PATH = BASE_DIR / "data" / "gold"

def main():
    # Carregar alunos tratados na Fase 2
    alunos = pd.read_parquet(
        SILVER_PATH / "alunos.parquet"
    )

    # Selecionar alunos efetivamente avaliados
    avaliados = alunos.loc[
        (alunos["presenca"] == "Presente")
        & alunos["proficiencia"].notna()
        & alunos["alfabetizado"].isin(["Sim", "Não"])
    ].copy()

    # Criar variável-alvo binária
    avaliados["target_alfabetizado"] = (
        avaliados["alfabetizado"] == "Sim"
    ).astype(int)

    # Manter apenas as colunas necessárias
    # Não incluir proficiência nas variáveis do modelo
    colunas = [
        "ano",
        "id_municipio",
        "id_aluno",
        "serie",
        "rede",
        "target_alfabetizado",
    ]

    base_ml = avaliados[colunas].copy()

    # Salvar uma nova Gold, sem alterar a Fase 2
    GOLD_PATH.mkdir(parents=True, exist_ok=True)

    destino = GOLD_PATH / "alunos_ml_base.parquet"
    base_ml.to_parquet(destino, index=False)

    print("=== BASE INICIAL DE MACHINE LEARNING ===")
    print(f"Registros: {len(base_ml)}")
    print(f"Colunas: {len(base_ml.columns)}")

    print("\nDistribuição da variável-alvo:")
    print(base_ml["target_alfabetizado"].value_counts())

    print("\nValores ausentes:")
    print(base_ml.isna().sum())

    print(f"\nArquivo gerado: {destino}")


if __name__ == "__main__":
    main()
