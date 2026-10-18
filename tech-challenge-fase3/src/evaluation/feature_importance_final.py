
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance


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
    # 1. CARREGAR DADOS DE DESENVOLVIMENTO
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

    historico = pd.read_parquet(
        GOLD_PATH / "alunos_ml_historico.parquet"
    )

    atributos_historicos = historico[
        CHAVES + ["taxa_alfabetizacao_2023"]
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

    X = desenvolvimento[FEATURES]
    y = desenvolvimento[TARGET]

    # ======================================
    # 2. CARREGAR MODELO FINAL
    # ======================================

    modelo = joblib.load(
        MODELS_PATH / "modelo_final_alfabetizacao.joblib"
    )

    # ======================================
    # 3. CALCULAR IMPORTÂNCIA
    # ======================================

    resultado = permutation_importance(
        modelo,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    importancia = pd.DataFrame({
        "variavel": FEATURES,
        "importancia_media": resultado.importances_mean,
        "desvio_padrao": resultado.importances_std,
    })

    importancia = importancia.sort_values(
        "importancia_media",
        ascending=False,
    )

    print("=== IMPORTÂNCIA DAS VARIÁVEIS ===")

    print(
        importancia.round(4).to_string(
            index=False
        )
    )

    # ======================================
    # 4. SALVAR RELATÓRIO
    # ======================================

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino_csv = (
        REPORTS_PATH
        / "importancia_variaveis_final.csv"
    )

    importancia.to_csv(
        destino_csv,
        index=False,
    )

    # ======================================
    # 5. GERAR GRÁFICO
    # ======================================

    IMAGES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    grafico = importancia.sort_values(
        "importancia_media"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        grafico["variavel"],
        grafico["importancia_media"],
        xerr=grafico["desvio_padrao"],
        capsize=5,
    )

    ax.set_title(
        "Importância das variáveis — Modelo final"
    )

    ax.set_xlabel(
        "Queda média da acurácia balanceada"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    destino_imagem = (
        IMAGES_PATH
        / "importancia_variaveis_final.png"
    )

    fig.savefig(
        destino_imagem,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n=== ARQUIVOS GERADOS ===")
    print(destino_csv)
    print(destino_imagem)


if __name__ == "__main__":
    main()
