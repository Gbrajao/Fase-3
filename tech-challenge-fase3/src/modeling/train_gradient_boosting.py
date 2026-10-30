from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "intermediate"
GOLD_PATH = BASE_DIR / "data" / "gold"
REPORTS_PATH = BASE_DIR / "reports"

CATEGORICAS = ["rede", "sigla_uf", "regiao"]
NUMERICAS = ["taxa_alfabetizacao_2023"]
FEATURES = CATEGORICAS + NUMERICAS
TARGET = "target_alfabetizado"
CHAVES = ["ano", "id_municipio", "id_aluno"]


def main():
    treino = pd.read_parquet(DATA_PATH / "treino.parquet")
    validacao = pd.read_parquet(DATA_PATH / "validacao.parquet")

    desenvolvimento = pd.concat(
        [treino, validacao],
        ignore_index=True,
    )

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    atributos_historicos = historico[CHAVES + NUMERICAS]

    assert not atributos_historicos.duplicated(
        subset=CHAVES
    ).any(), "Existem registros duplicados no histórico."

    desenvolvimento = desenvolvimento.merge(
        atributos_historicos,
        on=CHAVES,
        how="left",
        validate="one_to_one",
    )

    assert len(desenvolvimento) == len(treino) + len(validacao)

    X = desenvolvimento[FEATURES]
    y = desenvolvimento[TARGET]
    grupos = desenvolvimento["id_municipio"]

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
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
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
            ]),
            NUMERICAS,
        ),
    ])

    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        (
            "classificador",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
    ])

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    metricas = {
        "acuracia_balanceada": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "recall_nao_alfabetizados": make_scorer(
            recall_score,
            pos_label=0,
        ),
    }

    resultados = cross_validate(
        modelo,
        X,
        y,
        groups=grupos,
        cv=cv,
        scoring=metricas,
        n_jobs=-1,
        error_score="raise",
    )

    resumo = pd.DataFrame({
        "fold": range(1, 6),
        **{
            nome: resultados[f"test_{nome}"]
            for nome in metricas
        },
    })

    print("\n=== GRADIENT BOOSTING — VALIDAÇÃO CRUZADA ===")
    print(f"Alunos: {len(desenvolvimento)}")
    print(f"Municípios: {grupos.nunique()}")
    print("\nResultados por fold:")
    print(resumo.round(4).to_string(index=False))

    print("\nMédia e desvio padrão:")
    print(
        resumo.drop(columns="fold")
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    destino = REPORTS_PATH / "validacao_cruzada_gradient_boosting.csv"
    resumo.to_csv(destino, index=False)

    print(f"\nRelatório salvo em: {destino}")
    print("Conjunto de teste não utilizado.")
    print("Modelo final existente não foi alterado.")


if __name__ == "__main__":
    main()
