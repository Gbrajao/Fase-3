from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_socioeconomico import carregar_dados


CATEGORICAS = ["rede", "sigla_uf", "regiao"]
ALVO = "target_alfabetizado"
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
SAIDA = Path("reports/validacao_cruzada_interacao_renda_regiao.csv")


def preparar_variaveis(dados):
    dados = dados.copy()

    # As interações são criadas sem usar o alvo.
    # A renda será imputada e escalada dentro do pipeline.
    # Como a renda da Silver está completa, não há valores
    # ausentes nesta variável no experimento atual.
    if dados["renda_per_capita_2022"].isna().any():
        raise ValueError("A renda contém valores ausentes.")

    for regiao in REGIOES:
        coluna = "renda_x_" + regiao.lower().replace("-", "_")

        dados[coluna] = (
            dados["renda_per_capita_2022"]
            * dados["regiao"].eq(regiao).astype(int)
        )

    return dados


def criar_pipeline(numericas):
    preprocessador = ColumnTransformer([
        (
            "categoricas",
            Pipeline([
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAS,
        ),
        (
            "numericas",
            Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]),
            numericas,
        ),
    ])

    return Pipeline([
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


def main():
    dados = preparar_variaveis(carregar_dados())

    interacoes = [
        "renda_x_" + regiao.lower().replace("-", "_")
        for regiao in REGIOES
    ]

    modelos = {
        "logistica_atual": [
            "taxa_alfabetizacao_2023",
        ],
        "logistica_com_renda": [
            "taxa_alfabetizacao_2023",
            "renda_per_capita_2022",
        ],
        "logistica_interacao_renda_regiao": [
            "taxa_alfabetizacao_2023",
            "renda_per_capita_2022",
            *interacoes,
        ],
    }

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    divisoes = list(
        cv.split(
            dados,
            dados[ALVO],
            groups=dados["id_municipio"],
        )
    )

    resultados = []

    for nome, numericas in modelos.items():
        X = dados[CATEGORICAS + numericas]
        y = dados[ALVO]

        for fold, (idx_treino, idx_validacao) in enumerate(
            divisoes, start=1
        ):
            modelo = criar_pipeline(numericas)

            modelo.fit(
                X.iloc[idx_treino],
                y.iloc[idx_treino],
            )

            previsoes = modelo.predict(X.iloc[idx_validacao])
            probabilidades = modelo.predict_proba(
                X.iloc[idx_validacao]
            )[:, 1]

            y_validacao = y.iloc[idx_validacao]

            resultados.append({
                "modelo": nome,
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(
                    y_validacao, previsoes
                ),
                "roc_auc": roc_auc_score(
                    y_validacao, probabilidades
                ),
                "f1_macro": f1_score(
                    y_validacao, previsoes, average="macro"
                ),
                "recall_classe_0": recall_score(
                    y_validacao, previsoes, pos_label=0
                ),
            })

    tabela = pd.DataFrame(resultados)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)

    metricas = [
        "balanced_accuracy",
        "roc_auc",
        "f1_macro",
        "recall_classe_0",
    ]

    print("\n=== MÉDIA DOS CINCO FOLDS ===")
    print(
        tabela.groupby("modelo")[metricas]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n=== DESVIO-PADRÃO ===")
    print(
        tabela.groupby("modelo")[metricas]
        .std()
        .round(4)
        .to_string()
    )

    print("\nArquivo:", SAIDA)
    print("Conjunto de teste: NÃO UTILIZADO")


if __name__ == "__main__":
    main()
