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


CHAVES_ALUNO = ["ano", "id_municipio", "id_aluno"]
CATEGORICAS = ["rede", "sigla_uf", "regiao"]
NUMERICAS_BASE = ["taxa_alfabetizacao_2023"]
NUMERICAS_SOCIO = NUMERICAS_BASE + ["renda_per_capita_2022"]
ALVO = "target_alfabetizado"

SAIDA = Path("reports/validacao_cruzada_socioeconomico.csv")


def criar_pipeline(numericas):
    preprocessador = ColumnTransformer(
        transformers=[
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
        ]
    )

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


def carregar_dados():
    treino = pd.read_parquet("data/intermediate/treino.parquet")
    validacao = pd.read_parquet("data/intermediate/validacao.parquet")
    historico = pd.read_parquet("data/gold/alunos_ml_historico.parquet")
    socio = pd.read_parquet(
        "data/silver/ibge_renda_municipal_2022.parquet"
    )

    dados = pd.concat([treino, validacao], ignore_index=True)

    historico = historico[
        CHAVES_ALUNO + ["taxa_alfabetizacao_2023"]
    ].copy()

    if historico.duplicated(CHAVES_ALUNO).any():
        raise ValueError("Chaves de alunos duplicadas no histórico.")

    dados = dados.merge(
        historico,
        on=CHAVES_ALUNO,
        how="left",
        validate="one_to_one",
    )

    # A Silver do IBGE possui uma linha por município,
    # com renda histórica referente ao Censo de 2022.
    renda = socio[
        ["id_municipio", "renda_per_capita_2022"]
    ].copy()

    if renda["id_municipio"].duplicated().any():
        raise ValueError("Municípios duplicados na base de renda.")

    renda["id_municipio"] = renda["id_municipio"].astype(str)
    dados["id_municipio"] = dados["id_municipio"].astype(str)

    quantidade_antes = len(dados)

    dados = dados.merge(
        renda,
        on="id_municipio",
        how="left",
        validate="many_to_one",
    )

    if len(dados) != quantidade_antes:
        raise ValueError("A junção da renda alterou a quantidade de alunos.")

    ausentes = dados["renda_per_capita_2022"].isna().sum()

    print("=== BASE DO EXPERIMENTO ===")
    print("Alunos:", len(dados))
    print("Municípios:", dados["id_municipio"].nunique())
    print("Alunos sem renda municipal:", ausentes)
    print(
        "Alunos sem taxa histórica:",
        dados["taxa_alfabetizacao_2023"].isna().sum(),
    )

    if ausentes:
        raise ValueError("Existem alunos sem renda municipal.")

    if dados[ALVO].isna().any():
        raise ValueError("Existem alunos sem classificação.")

    return dados


def avaliar_modelo(dados, nome, numericas, divisao):
    features = CATEGORICAS + numericas
    X = dados[features]
    y = dados[ALVO]

    resultados = []

    for fold, (indices_treino, indices_validacao) in enumerate(
        divisao, start=1
    ):
        modelo = criar_pipeline(numericas)

        X_treino = X.iloc[indices_treino]
        X_validacao = X.iloc[indices_validacao]
        y_treino = y.iloc[indices_treino]
        y_validacao = y.iloc[indices_validacao]

        modelo.fit(X_treino, y_treino)

        previsoes = modelo.predict(X_validacao)
        probabilidades = modelo.predict_proba(X_validacao)[:, 1]

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

    return resultados


def main():
    dados = carregar_dados()

    X_grupos = dados["id_municipio"]
    y = dados[ALVO]

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Mesmas divisões para comparar os dois modelos.
    divisoes = list(
        cv.split(dados, y, groups=X_grupos)
    )

    resultados = []

    resultados.extend(
        avaliar_modelo(
            dados, "logistica_atual", NUMERICAS_BASE, divisoes
        )
    )

    resultados.extend(
        avaliar_modelo(
            dados, "logistica_com_renda_2022",
            NUMERICAS_SOCIO, divisoes
        )
    )

    tabela = pd.DataFrame(resultados)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)

    metricas = [
        "balanced_accuracy",
        "roc_auc",
        "f1_macro",
        "recall_classe_0",
    ]

    print()
    print("=== RESULTADOS: MÉDIA DOS 5 FOLDS ===")
    print(
        tabela.groupby("modelo")[metricas]
        .mean()
        .round(4)
        .to_string()
    )

    print()
    print("=== DESVIO-PADRÃO DOS 5 FOLDS ===")
    print(
        tabela.groupby("modelo")[metricas]
        .std()
        .round(4)
        .to_string()
    )

    medias = tabela.groupby("modelo")[metricas].mean()

    diferenca = (
        medias.loc["logistica_com_renda_2022"]
        - medias.loc["logistica_atual"]
    )

    print()
    print("=== DIFERENÇA: COM RENDA - ATUAL ===")
    print(diferenca.round(4).to_string())

    print()
    print("Resultados salvos em:", SAIDA)
    print("Conjunto de teste: NÃO UTILIZADO")


if __name__ == "__main__":
    main()
