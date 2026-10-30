from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "intermediate"
GOLD_DIR = BASE_DIR / "data" / "gold"
REPORTS_DIR = BASE_DIR / "reports"

CATEGORICAS = ["rede", "sigla_uf", "regiao"]

NUMERICAS = [
    "taxa_alfabetizacao_2023",
    "percentual_participacao_2023",
]

FEATURES = CATEGORICAS + NUMERICAS
TARGET = "target_alfabetizado"

CHAVES = ["ano", "id_municipio", "id_aluno"]


def main():
    treino = pd.read_parquet(DATA_DIR / "treino.parquet")
    validacao = pd.read_parquet(DATA_DIR / "validacao.parquet")

    desenvolvimento = pd.concat(
        [treino, validacao],
        ignore_index=True,
    )

    historico_alunos = pd.read_parquet(
        GOLD_DIR / "alunos_ml_historico.parquet"
    )

    historico_municipal = pd.read_parquet(
        GOLD_DIR / "municipio_enriquecido_ibge.parquet"
    )

    historico_municipal = historico_municipal.loc[
        historico_municipal["ano"] == 2023,
        ["id_municipio", "percentual_participacao"],
    ].copy()

    historico_municipal = historico_municipal.rename(
        columns={
            "percentual_participacao":
                "percentual_participacao_2023"
        }
    )

    assert not historico_municipal.duplicated(
        subset=["id_municipio"]
    ).any(), "Municípios duplicados no histórico."

    historico_alunos = historico_alunos[
        CHAVES + ["taxa_alfabetizacao_2023"]
    ]

    desenvolvimento = desenvolvimento.merge(
        historico_alunos,
        on=CHAVES,
        how="left",
        validate="one_to_one",
    )

    desenvolvimento = desenvolvimento.merge(
        historico_municipal,
        on="id_municipio",
        how="left",
        validate="many_to_one",
    )

    assert len(desenvolvimento) == (
        len(treino) + len(validacao)
    ), "A integração alterou o número de alunos."

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

    print("\n=== MODELO COM PARTICIPAÇÃO HISTÓRICA ===")
    print(f"Alunos: {len(desenvolvimento)}")
    print(f"Municípios: {grupos.nunique()}")

    print("\nRESULTADOS POR FOLD:")
    print(resumo.round(4).to_string(index=False))

    print("\nMÉDIA E DESVIO PADRÃO:")
    print(
        resumo.drop(columns="fold")
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    destino = (
        REPORTS_DIR
        / "validacao_cruzada_participacao_historica.csv"
    )

    resumo.to_csv(destino, index=False)

    print(f"\nRelatório salvo em: {destino}")
    print("Conjunto de teste não utilizado.")
    print("Modelo final existente não foi alterado.")


if __name__ == "__main__":
    main()
