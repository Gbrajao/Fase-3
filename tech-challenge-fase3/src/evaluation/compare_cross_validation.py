
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

REPORTS_PATH = BASE_DIR / "reports"
IMAGES_PATH = BASE_DIR / "images"

METRICAS = [
    "acuracia_balanceada",
    "roc_auc",
    "f1_macro",
]

NOMES = [
    "Acurácia balanceada",
    "ROC-AUC",
    "F1 macro",
]


def main():

    # ======================================
    # 1. CARREGAR RESULTADOS DOS MODELOS
    # ======================================

    original = pd.read_csv(
        REPORTS_PATH / "validacao_cruzada.csv"
    )

    enriquecido = pd.read_csv(
        REPORTS_PATH / "validacao_cruzada_historico.csv"
    )

    # ======================================
    # 2. CALCULAR MÉDIAS E DESVIOS
    # ======================================

    registros = []

    for nome, dados in [
        ("Original", original),
        ("Enriquecido", enriquecido),
    ]:
        for metrica in METRICAS:
            registros.append({
                "modelo": nome,
                "metrica": metrica,
                "media": dados[metrica].mean(),
                "desvio_padrao": dados[metrica].std(),
            })

    comparacao = pd.DataFrame(registros)

    print("=== COMPARAÇÃO DA VALIDAÇÃO CRUZADA ===")

    print(
        comparacao.round(4).to_string(
            index=False
        )
    )

    # ======================================
    # 3. SALVAR RELATÓRIO
    # ======================================

    REPORTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    destino_csv = (
        REPORTS_PATH
        / "comparacao_validacao_cruzada.csv"
    )

    comparacao.to_csv(
        destino_csv,
        index=False,
    )

    # ======================================
    # 4. GERAR GRÁFICO COM DESVIO-PADRÃO
    # ======================================

    IMAGES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    medias_original = [
        original[metrica].mean()
        for metrica in METRICAS
    ]

    medias_enriquecido = [
        enriquecido[metrica].mean()
        for metrica in METRICAS
    ]

    desvios_original = [
        original[metrica].std()
        for metrica in METRICAS
    ]

    desvios_enriquecido = [
        enriquecido[metrica].std()
        for metrica in METRICAS
    ]

    posicoes = np.arange(len(METRICAS))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        posicoes - largura / 2,
        medias_original,
        largura,
        yerr=desvios_original,
        capsize=5,
        label="Original",
    )

    ax.bar(
        posicoes + largura / 2,
        medias_enriquecido,
        largura,
        yerr=desvios_enriquecido,
        capsize=5,
        label="Enriquecido",
    )

    ax.set_title(
        "Validação cruzada por município\n"
        "Modelo original x modelo enriquecido"
    )

    ax.set_ylabel("Pontuação média")
    ax.set_xticks(posicoes)
    ax.set_xticklabels(NOMES)
    ax.set_ylim(0, 0.75)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    destino_imagem = (
        IMAGES_PATH
        / "comparacao_validacao_cruzada.png"
    )

    fig.savefig(
        destino_imagem,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nRelatório salvo em: {destino_csv}")
    print(f"Gráfico salvo em: {destino_imagem}")


if __name__ == "__main__":
    main()
