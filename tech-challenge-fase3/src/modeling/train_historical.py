
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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
GOLD_PATH = BASE_DIR / "data" / "gold"

CATEGORICAS = ["rede", "sigla_uf", "regiao"]
NUMERICAS = ["taxa_alfabetizacao_2023"]

FEATURES = CATEGORICAS + NUMERICAS
TARGET = "target_alfabetizado"


def main():
    treino_original = pd.read_parquet(
        DATA_PATH / "treino.parquet"
    )

    validacao_original = pd.read_parquet(
        DATA_PATH / "validacao.parquet"
    )

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    # Preservar os mesmos alunos dos conjuntos originais.
    # A junção usa a identificação individual apenas para
    # recuperar a variável histórica, nunca como preditora.
    chaves = ["ano", "id_municipio", "id_aluno"]

    atributos_historicos = historico[
        chaves + NUMERICAS
    ]

    assert not atributos_historicos.duplicated(
        subset=chaves
    ).any(), "Há chaves individuais duplicadas."

    treino = treino_original.merge(
        atributos_historicos,
        on=chaves,
        how="left",
        validate="one_to_one",
    )

    validacao = validacao_original.merge(
        atributos_historicos,
        on=chaves,
        how="left",
        validate="one_to_one",
    )

    assert len(treino) == len(treino_original)
    assert len(validacao) == len(validacao_original)

    X_treino = treino[FEATURES]
    y_treino = treino[TARGET]

    X_validacao = validacao[FEATURES]
    y_validacao = validacao[TARGET]

    preprocessador = ColumnTransformer([
        (
            "categoricas",
            Pipeline([
                (
                    "imputador",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            CATEGORICAS,
        ),
        (
            "numericas",
            Pipeline([
                (
                    "imputador",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "padronizador",
                    StandardScaler(),
                ),
            ]),
            NUMERICAS,
        ),
    ])

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_validacao)
    probabilidades = modelo.predict_proba(
        X_validacao
    )[:, 1]

    print("=== MODELO ENRIQUECIDO ===")

    print(
        "Alunos no treino:",
        len(treino)
    )

    print(
        "Alunos na validação:",
        len(validacao)
    )

    print(
        "\nAcurácia:",
        round(
            accuracy_score(y_validacao, previsoes),
            4,
        ),
    )

    print(
        "Acurácia balanceada:",
        round(
            balanced_accuracy_score(
                y_validacao,
                previsoes,
            ),
            4,
        ),
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_validacao,
                probabilidades,
            ),
            4,
        ),
    )

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_validacao,
            previsoes,
            digits=4,
        )
    )

    print("\nMatriz de confusão:")
    print(
        confusion_matrix(
            y_validacao,
            previsoes,
        )
    )


if __name__ == "__main__":
    main()
