
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==========================================
# CAMINHOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "intermediate"
GOLD_PATH = BASE_DIR / "data" / "gold"
MODELS_PATH = BASE_DIR / "models"
REPORTS_PATH = BASE_DIR / "reports"
IMAGES_PATH = BASE_DIR / "images"


# ==========================================
# VARIÁVEIS
# ==========================================

FEATURES = [
    "rede",
    "sigla_uf",
    "regiao",
    "taxa_alfabetizacao_2023",
]

TARGET = "target_alfabetizado"

CHAVES = [
    "ano",
    "id_municipio",
    "id_aluno",
]


def main():

    # ======================================
    # 1. CARREGAR TESTE
    # ======================================

    teste = pd.read_parquet(
        DATA_PATH / "teste.parquet"
    )

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    atributos_historicos = historico[
        CHAVES + ["taxa_alfabetizacao_2023"]
    ]

    assert not atributos_historicos.duplicated(
        subset=CHAVES
    ).any(), "Existem alunos duplicados no histórico."

    teste = teste.merge(
        atributos_historicos,
        on=CHAVES,
        how="left",
        validate="one_to_one",
    )

    assert len(teste) == 1130, (
        "Quantidade inesperada de alunos no teste."
    )

    # ======================================
    # 2. CARREGAR MODELO FINAL
    # ======================================

    modelo = joblib.load(
        MODELS_PATH / "modelo_final_alfabetizacao.joblib"
    )

    X_teste = teste[FEATURES]
    y_teste = teste[TARGET]

    # ======================================
    # 3. GERAR PREVISÕES
    # ======================================

    previsoes = modelo.predict(X_teste)

    probabilidades = modelo.predict_proba(
        X_teste
    )[:, 1]

    # ======================================
    # 4. CALCULAR MÉTRICAS
    # ======================================

    metricas = {
        "total_alunos": int(len(teste)),
        "total_municipios": int(
            teste["id_municipio"].nunique()
        ),
        "acuracia": float(
            accuracy_score(y_teste, previsoes)
        ),
        "acuracia_balanceada": float(
            balanced_accuracy_score(
                y_teste,
                previsoes,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_teste,
                probabilidades,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_teste,
                previsoes,
                average="macro",
            )
        ),
        "precision_classe_0": float(
            precision_score(
                y_teste,
                previsoes,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_classe_0": float(
            recall_score(
                y_teste,
                previsoes,
                pos_label=0,
            )
        ),
        "recall_classe_1": float(
            recall_score(
                y_teste,
                previsoes,
                pos_label=1,
            )
        ),
    }

    print("=== AVALIAÇÃO FINAL NO TESTE ===")

    print(
        f"Alunos avaliados: {metricas['total_alunos']}"
    )

    print(
        f"Municípios: {metricas['total_municipios']}"
    )

    print(
        f"Acurácia: {metricas['acuracia']:.4f}"
    )

    print(
        "Acurácia balanceada: "
        f"{metricas['acuracia_balanceada']:.4f}"
    )

    print(
        f"ROC-AUC: {metricas['roc_auc']:.4f}"
    )

    print(
        f"F1 macro: {metricas['f1_macro']:.4f}"
    )

    print(
        "Recall classe 0: "
        f"{metricas['recall_classe_0']:.4f}"
    )

    print(
        "Recall classe 1: "
        f"{metricas['recall_classe_1']:.4f}"
    )

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")

    print(
        classification_report(
            y_teste,
            previsoes,
            labels=[0, 1],
            target_names=[
                "Não alfabetizado",
                "Alfabetizado",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # ======================================
    # 5. MATRIZ DE CONFUSÃO
    # ======================================

    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=[0, 1],
    )

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(matriz)

    IMAGES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[
            "Não alfabetizado",
            "Alfabetizado",
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Matriz de confusão — Teste final"
    )

    fig.tight_layout()

    destino_imagem = (
        IMAGES_PATH
        / "matriz_confusao_teste_final.png"
    )

    fig.savefig(
        destino_imagem,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ======================================
    # 6. SALVAR MÉTRICAS
    # ======================================

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino_json = (
        REPORTS_PATH
        / "metricas_teste_final.json"
    )

    with open(
        destino_json,
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            metricas,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    print("\n=== ARQUIVOS GERADOS ===")
    print(destino_json)
    print(destino_imagem)


if __name__ == "__main__":
    main()
