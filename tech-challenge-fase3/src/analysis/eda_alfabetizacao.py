
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "gold"
IMAGES_PATH = BASE_DIR / "images"
REPORTS_PATH = BASE_DIR / "reports"

def main():
    df = pd.read_parquet(
        DATA_PATH / "alunos_ml_enriquecido.parquet"
    )

    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    # Resumo por região
    resumo = (
        df.groupby("regiao")["target_alfabetizado"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    resumo.columns = [
        "regiao",
        "total_alunos",
        "alfabetizados",
        "taxa_alfabetizacao",
    ]

    resumo["taxa_alfabetizacao"] *= 100
    resumo = resumo.sort_values(
        "taxa_alfabetizacao",
        ascending=False
    )

    print("=== ALFABETIZAÇÃO POR REGIÃO ===")
    print(resumo.to_string(index=False))

    # Salvar tabela para documentação
    resumo.to_csv(
        REPORTS_PATH / "alfabetizacao_por_regiao.csv",
        index=False
    )

    # Gerar gráfico
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=resumo,
        x="regiao",
        y="taxa_alfabetizacao",
        color="steelblue"
    )

    ax.set_title(
        "Percentual de alunos alfabetizados por região"
    )
    ax.set_xlabel("Região")
    ax.set_ylabel("Alfabetizados na amostra (%)")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    plt.xticks(rotation=15)
    plt.tight_layout()

    destino = (
        IMAGES_PATH / "alfabetizacao_por_regiao.png"
    )

    plt.savefig(destino, dpi=150)
    plt.close()

    print(f"\nGráfico salvo em: {destino}")


if __name__ == "__main__":
    main()
