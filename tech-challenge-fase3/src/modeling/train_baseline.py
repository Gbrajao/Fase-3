
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
MODEL_PATH = BASE_DIR / "models"
REPORTS_PATH = BASE_DIR / "reports"

FEATURES = ["rede", "sigla_uf", "regiao"]
TARGET = "target_alfabetizado"


def carregar(nome):
    return pd.read_parquet(DATA_PATH / f"{nome}.parquet")


def avaliar(modelo, df, nome):
    X = df[FEATURES]
    y = df[TARGET]

    previsoes = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    metricas = {
        "accuracy": float(accuracy_score(y, previsoes)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, previsoes)
        ),
        "f1": float(f1_score(y, previsoes)),
        "roc_auc": float(
            roc_auc_score(y, probabilidades)
        ),
    }

    print(f"\n=== {nome.upper()} ===")

    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(
        y,
        previsoes,
        zero_division=0
    ))

    print("Matriz de confusão:")
    print(confusion_matrix(y, previsoes))

    return metricas


def main():
    treino = carregar("treino")
    validacao = carregar("validacao")

    preprocessador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline([
                    (
                        "imputador",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                ]),
                FEATURES,
            )
        ]
    )

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        (
            "classificador",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        ),
    ])

    modelo.fit(
        treino[FEATURES],
        treino[TARGET]
    )

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    metricas = {
        "treino": avaliar(modelo, treino, "treino"),
        "validacao": avaliar(
            modelo,
            validacao,
            "validacao"
        ),
    }

    joblib.dump(
        modelo,
        MODEL_PATH / "baseline_logistic.joblib"
    )

    with open(
        REPORTS_PATH / "metricas_baseline.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            metricas,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print("\nModelo e métricas salvos com sucesso.")
    print("Conjunto de teste preservado para avaliação final.")


if __name__ == "__main__":
    main()
