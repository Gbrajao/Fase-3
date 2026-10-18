
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
REPORTS_PATH = BASE_DIR / "reports"
IMAGES_PATH = BASE_DIR / "images"

FEATURES = ["rede", "sigla_uf", "regiao"]
TARGET = "target_alfabetizado"


def main():
    treino = pd.read_parquet(DATA_PATH / "treino.parquet")
    validacao = pd.read_parquet(DATA_PATH / "validacao.parquet")

    preprocessador = ColumnTransformer([
        (
            "categoricas",
            Pipeline([
                (
                    "imputador",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                ),
            ]),
            FEATURES,
        )
    ])

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        ),
    ])

    modelo.fit(
        treino[FEATURES],
        treino[TARGET]
    )

    resultado = permutation_importance(
        modelo,
        validacao[FEATURES],
        validacao[TARGET],
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    importancia = pd.DataFrame({
        "variavel": FEATURES,
        "importancia_media": resultado.importances_mean,
        "desvio_padrao": resultado.importances_std,
    })

    importancia = importancia.sort_values(
        "importancia_media",
        ascending=False
    )

    print("=== IMPORTÂNCIA DAS VARIÁVEIS ===")
    print(importancia.to_string(index=False))

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    importancia.to_csv(
        REPORTS_PATH / "importancia_variaveis.csv",
        index=False
    )

    grafico = importancia.sort_values("importancia_media")

    plt.figure(figsize=(9, 5))

    plt.barh(
        grafico["variavel"],
        grafico["importancia_media"],
        xerr=grafico["desvio_padrao"],
        color="steelblue",
        capsize=4,
    )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Importância das variáveis — Logística balanceada")
    plt.xlabel("Queda na acurácia balanceada ao embaralhar")
    plt.ylabel("Variável")

    plt.tight_layout()

    destino = IMAGES_PATH / "importancia_variaveis.png"
    plt.savefig(destino, dpi=150)
    plt.close()

    print(f"\nGráfico salvo em: {destino}")


if __name__ == "__main__":
    main()
