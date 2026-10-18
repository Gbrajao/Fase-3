
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]

REPORTS_PATH = BASE_DIR / "reports"
IMAGES_PATH = BASE_DIR / "images"


def main():
    dados = [
        {
            "modelo": "Regressão Logística",
            "acuracia": 0.6328,
            "acuracia_balanceada": 0.5484,
            "roc_auc": 0.5989,
            "recall_classe_0": 85 / 439,
            "f1_classe_0": 0.29,
        },
        {
            "modelo": "Logística Balanceada",
            "acuracia": 0.5668,
            "acuracia_balanceada": 0.5678,
            "roc_auc": 0.5991,
            "recall_classe_0": 251 / 439,
            "f1_classe_0": 0.50,
        },
        {
            "modelo": "Random Forest",
            "acuracia": 0.5538,
            "acuracia_balanceada": 0.5647,
            "roc_auc": 0.5991,
            "recall_classe_0": 268 / 439,
            "f1_classe_0": 0.51,
        },
    ]

    df = pd.DataFrame(dados)

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    destino_csv = REPORTS_PATH / "comparacao_modelos.csv"
    df.to_csv(destino_csv, index=False)

    print("=== COMPARAÇÃO DOS MODELOS ===")
    print(df.round(4).to_string(index=False))

    metricas = [
        "acuracia_balanceada",
        "roc_auc",
        "recall_classe_0",
    ]

    grafico = df.set_index("modelo")[metricas]

    ax = grafico.plot(
        kind="bar",
        figsize=(11, 6),
        rot=10,
    )

    ax.set_title("Comparação dos modelos — validação")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Pontuação")
    ax.set_ylim(0, 1)

    ax.legend([
        "Acurácia balanceada",
        "ROC-AUC",
        "Recall classe 0",
    ])

    plt.tight_layout()

    destino_png = IMAGES_PATH / "comparacao_modelos.png"
    plt.savefig(destino_png, dpi=150)
    plt.close()

    print(f"\nTabela salva em: {destino_csv}")
    print(f"Gráfico salvo em: {destino_png}")


if __name__ == "__main__":
    main()
