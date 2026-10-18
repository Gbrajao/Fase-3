
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data/gold/alunos_ml_enriquecido.parquet"
IMAGES_PATH = BASE_DIR / "images"
REPORTS_PATH = BASE_DIR / "reports"


def main():
    df = pd.read_parquet(DATA_PATH)

    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    print("=== DISTRIBUIÇÃO DAS CLASSES ===")
    print(df["target_alfabetizado"].value_counts())

    print("\nPercentuais:")
    print(
        df["target_alfabetizado"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    resumo_rede = (
        df.groupby("rede")["target_alfabetizado"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    resumo_rede.columns = [
        "rede",
        "total_alunos",
        "alfabetizados",
        "taxa_alfabetizacao",
    ]

    resumo_rede["taxa_alfabetizacao"] *= 100

    print("\n=== ALFABETIZAÇÃO POR REDE ===")
    print(resumo_rede.to_string(index=False))

    resumo_rede.to_csv(
        REPORTS_PATH / "alfabetizacao_por_rede.csv",
        index=False
    )

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    contagem = (
        df["target_alfabetizado"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    axes[0].bar(
        ["Não alfabetizado", "Alfabetizado"],
        contagem.values,
        color=["#D68A54", "#4C9F8A"]
    )

    axes[0].set_title("Distribuição das classes")
    axes[0].set_ylabel("Quantidade de alunos")

    for i, valor in enumerate(contagem.values):
        axes[0].text(
            i, valor + 30, str(valor),
            ha="center"
        )

    sns.barplot(
        data=resumo_rede,
        x="rede",
        y="taxa_alfabetizacao",
        ax=axes[1],
        color="steelblue"
    )

    axes[1].set_title("Alfabetização por rede")
    axes[1].set_xlabel("Rede de ensino")
    axes[1].set_ylabel("Alfabetizados na amostra (%)")
    axes[1].set_ylim(0, 100)

    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%")

    plt.tight_layout()

    destino = IMAGES_PATH / "analise_classes_rede.png"

    plt.savefig(destino, dpi=150)
    plt.close()

    print(f"\nGráfico salvo em: {destino}")


if __name__ == "__main__":
    main()
