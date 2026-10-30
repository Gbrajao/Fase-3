from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
GOLD_PATH = BASE_DIR / "data" / "gold"
REPORTS_PATH = BASE_DIR / "reports"
IMAGES_PATH = BASE_DIR / "images"

VARIAVEIS = [
    "taxa_alfabetizacao",
    "percentual_participacao",
]


def main():
    df = pd.read_parquet(
        GOLD_PATH / "municipio_enriquecido_ibge.parquet"
    )

    municipios = df.loc[
        df["ano"] == 2024,
        [
            "id_municipio",
            "nome_municipio",
            "sigla_uf",
            "regiao",
        ] + VARIAVEIS,
    ].copy()

    assert not municipios["id_municipio"].duplicated().any(), (
        "Existem municípios duplicados em 2024."
    )

    # Evitar criar perfis com valores artificiais
    # quando faltam os dois indicadores.
    municipios = municipios.dropna(subset=VARIAVEIS)

    assert len(municipios) >= 10, (
        "Municípios insuficientes para o agrupamento."
    )

    preprocessador = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("padronizador", StandardScaler()),
    ])

    X = preprocessador.fit_transform(municipios[VARIAVEIS])

    resultados = []

    for k in range(2, 7):
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        grupos = modelo.fit_predict(X)

        silhouette = silhouette_score(
            X,
            grupos,
            sample_size=min(2000, len(X)),
            random_state=42,
        )

        resultados.append({
            "k": k,
            "silhouette": silhouette,
        })

    avaliacao = pd.DataFrame(resultados)

    melhor_k = int(
        avaliacao.loc[
            avaliacao["silhouette"].idxmax(),
            "k",
        ]
    )

    modelo_final = KMeans(
        n_clusters=melhor_k,
        random_state=42,
        n_init=10,
    )

    municipios["cluster"] = modelo_final.fit_predict(X)

    resumo = (
        municipios.groupby("cluster")
        .agg(
            municipios=("id_municipio", "nunique"),
            taxa_media=("taxa_alfabetizacao", "mean"),
            participacao_media=(
                "percentual_participacao",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("taxa_media")
    )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    municipios.to_csv(
        REPORTS_PATH / "clusters_municipios_2024.csv",
        index=False,
    )

    resumo.to_csv(
        REPORTS_PATH / "resumo_clusters_municipios_2024.csv",
        index=False,
    )

    avaliacao.to_csv(
        REPORTS_PATH / "avaliacao_kmeans_silhouette.csv",
        index=False,
    )

    plt.figure(figsize=(10, 7))

    dispersao = plt.scatter(
        municipios["taxa_alfabetizacao"],
        municipios["percentual_participacao"],
        c=municipios["cluster"],
        cmap="tab10",
        alpha=0.55,
        s=16,
    )

    plt.xlabel("Taxa de alfabetização em 2024 (%)")
    plt.ylabel("Participação na avaliação em 2024 (%)")
    plt.title("Perfis educacionais municipais — K-Means")
    plt.colorbar(dispersao, label="Cluster")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    destino_grafico = (
        IMAGES_PATH / "clusters_municipios_2024.png"
    )

    plt.savefig(destino_grafico, dpi=180)
    plt.close()

    print("\n=== AGRUPAMENTO MUNICIPAL COM K-MEANS ===")
    print(f"Municípios analisados: {len(municipios)}")

    print("\nAVALIAÇÃO DOS NÚMEROS DE GRUPOS:")
    print(avaliacao.round(4).to_string(index=False))

    print(f"\nNúmero de grupos escolhido: {melhor_k}")

    print("\nPERFIL DOS GRUPOS:")
    print(resumo.round(2).to_string(index=False))

    print("\nArquivos gerados:")
    print("- reports/clusters_municipios_2024.csv")
    print("- reports/resumo_clusters_municipios_2024.csv")
    print("- reports/avaliacao_kmeans_silhouette.csv")
    print("- images/clusters_municipios_2024.png")

    print("\nAnálise descritiva concluída.")
    print("Modelo supervisionado final não foi alterado.")


if __name__ == "__main__":
    main()
