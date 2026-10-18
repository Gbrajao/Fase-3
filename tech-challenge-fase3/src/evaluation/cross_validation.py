
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# CAMINHOS DO PROJETO
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
GOLD_PATH = BASE_DIR / "data" / "gold"
REPORTS_PATH = BASE_DIR / "reports"


# ==========================================
# VARIÁVEIS DO MODELO
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

    # O conjunto de teste não será utilizado.

    desenvolvimento = pd.concat(
        [treino, validacao],
        ignore_index=True,
    )

    # ======================================
    # 2. CARREGAR O INDICADOR HISTÓRICO
    # ======================================

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    chaves = [
        "ano",
        "id_municipio",
        "id_aluno",
    ]

    atributos_historicos = historico[
        chaves + NUMERICAS
    ]

    assert not atributos_historicos.duplicated(
        subset=chaves
    ).any(), "Existem alunos duplicados na base histórica."

    desenvolvimento = desenvolvimento.merge(
        atributos_historicos,
        on=chaves,
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

    grupos = desenvolvimento["id_municipio"]

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
    # 5. MODELO
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
    # 6. VALIDAÇÃO CRUZADA POR MUNICÍPIO
    # ======================================

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    resultados = cross_validate(
        modelo,
        X,
        y,
        groups=grupos,
        cv=cv,
        scoring={
            "acuracia_balanceada": "balanced_accuracy",
            "roc_auc": "roc_auc",
            "f1_macro": "f1_macro",
        },
        n_jobs=-1,
    )

    # ======================================
    # 7. ORGANIZAR RESULTADOS
    # ======================================

    resumo = pd.DataFrame({
        "fold": range(1, 6),
        "acuracia_balanceada": resultados[
            "test_acuracia_balanceada"
        ],
        "roc_auc": resultados[
            "test_roc_auc"
        ],
        "f1_macro": resultados[
            "test_f1_macro"
        ],
    })

    print(
        "\n=== VALIDAÇÃO CRUZADA "
        "DO MODELO ENRIQUECIDO ==="
    )

    print(
        resumo.round(4).to_string(
            index=False
        )
    )

    print("\n=== MÉDIA E DESVIO PADRÃO ===")

    estatisticas = (
        resumo.drop(columns="fold")
        .agg(["mean", "std"])
        .round(4)
    )

    print(estatisticas)

    # ======================================
    # 8. SALVAR RELATÓRIO
    # ======================================

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino = (
        REPORTS_PATH
        / "validacao_cruzada_historico.csv"
    )

    resumo.to_csv(
        destino,
        index=False,
    )

    print(
        f"\nRelatório salvo em: {destino}"
    )


if __name__ == "__main__":
    main()
