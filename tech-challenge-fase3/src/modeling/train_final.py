
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# CAMINHOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
GOLD_PATH = BASE_DIR / "data" / "gold"
MODELS_PATH = BASE_DIR / "models"


# ==========================================
# VARIÁVEIS
# ==========================================

CATEGORICAS = [
    "rede",
    "sigla_uf",
    "regiao",
]

NUMERICAS = [
    "taxa_alfabetizacao_2023",
]

FEATURES = CATEGORICAS + NUMERICAS

TARGET = "target_alfabetizado"

CHAVES = [
    "ano",
    "id_municipio",
    "id_aluno",
]


def main():

    # ======================================
    # 1. CARREGAR TREINO E VALIDAÇÃO
    # ======================================

    treino = pd.read_parquet(
        DATA_PATH / "treino.parquet"
    )

    validacao = pd.read_parquet(
        DATA_PATH / "validacao.parquet"
    )

    desenvolvimento = pd.concat(
        [treino, validacao],
        ignore_index=True,
    )

    # ======================================
    # 2. CARREGAR HISTÓRICO
    # ======================================

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    atributos_historicos = historico[
        CHAVES + NUMERICAS
    ]

    assert not atributos_historicos.duplicated(
        subset=CHAVES
    ).any(), "Existem alunos duplicados no histórico."

    desenvolvimento = desenvolvimento.merge(
        atributos_historicos,
        on=CHAVES,
        how="left",
        validate="one_to_one",
    )

    assert len(desenvolvimento) == (
        len(treino) + len(validacao)
    ), "A integração alterou o total de alunos."

    # ======================================
    # 3. SEPARAR VARIÁVEIS E ALVO
    # ======================================

    X = desenvolvimento[FEATURES]
    y = desenvolvimento[TARGET]

    # ======================================
    # 4. PRÉ-PROCESSAMENTO
    # ======================================

    preprocessador = ColumnTransformer([
        (
            "categoricas",
            Pipeline([
                (
                    "imputador",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
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
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "padronizador",
                    StandardScaler(),
                ),
            ]),
            NUMERICAS,
        ),
    ])

    # ======================================
    # 5. PIPELINE FINAL
    # ======================================

    modelo = Pipeline([
        (
            "preprocessamento",
            preprocessador,
        ),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # ======================================
    # 6. TREINAR
    # ======================================

    modelo.fit(X, y)

    # ======================================
    # 7. SALVAR MODELO
    # ======================================

    MODELS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino = (
        MODELS_PATH
        / "modelo_final_alfabetizacao.joblib"
    )

    joblib.dump(
        modelo,
        destino,
    )

    print("=== MODELO FINAL TREINADO ===")
    print("Algoritmo: Regressão Logística balanceada")
    print("Alunos utilizados:", len(desenvolvimento))
    print("Municípios:", desenvolvimento["id_municipio"].nunique())
    print("Variáveis:", FEATURES)

    print("\nModelo salvo em:")
    print(destino)

    print(
        "\nConjunto de teste preservado: "
        "nenhum dado de teste foi utilizado."
    )


if __name__ == "__main__":
    main()
