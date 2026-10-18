
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "intermediate"

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

    X_val = validacao[FEATURES]
    y_val = validacao[TARGET]

    previsoes = modelo.predict(X_val)
    probabilidades = modelo.predict_proba(X_val)[:, 1]

    print("=== REGRESSÃO LOGÍSTICA BALANCEADA ===")

    print(
        f"Acurácia: {accuracy_score(y_val, previsoes):.4f}"
    )

    print(
        "Acurácia balanceada: "
        f"{balanced_accuracy_score(y_val, previsoes):.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc_score(y_val, probabilidades):.4f}"
    )

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_val,
            previsoes,
            zero_division=0
        )
    )

    print("Matriz de confusão:")
    print(confusion_matrix(y_val, previsoes))


if __name__ == "__main__":
    main()
