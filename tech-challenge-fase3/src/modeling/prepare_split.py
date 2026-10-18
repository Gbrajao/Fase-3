
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/gold/alunos_ml_enriquecido.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "intermediate"


def separar_grupos(df, test_size, random_state):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    treino_idx, teste_idx = next(
        splitter.split(
            df,
            y=df["target_alfabetizado"],
            groups=df["id_municipio"]
        )
    )

    return (
        df.iloc[treino_idx].copy(),
        df.iloc[teste_idx].copy()
    )


def main():
    df = pd.read_parquet(DATA_PATH)

    # Reservar aproximadamente 20% dos municípios
    # para o conjunto de teste
    desenvolvimento, teste = separar_grupos(
        df,
        test_size=0.20,
        random_state=42
    )

    # Separar aproximadamente 25% dos municípios
    # restantes para validação
    treino, validacao = separar_grupos(
        desenvolvimento,
        test_size=0.25,
        random_state=42
    )

    conjuntos = {
        "treino": treino,
        "validacao": validacao,
        "teste": teste,
    }

    # Verificar ausência de municípios compartilhados
    grupos = {
        nome: set(dados["id_municipio"])
        for nome, dados in conjuntos.items()
    }

    assert grupos["treino"].isdisjoint(grupos["validacao"])
    assert grupos["treino"].isdisjoint(grupos["teste"])
    assert grupos["validacao"].isdisjoint(grupos["teste"])

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("=== SEPARAÇÃO DOS DADOS ===")

    for nome, dados in conjuntos.items():
        destino = OUTPUT_PATH / f"{nome}.parquet"
        dados.to_parquet(destino, index=False)

        print(f"\n{nome.upper()}")
        print(f"Registros: {len(dados)}")
        print(
            f"Municípios: {dados['id_municipio'].nunique()}"
        )
        print(
            "Alfabetizados (%): "
            f"{dados['target_alfabetizado'].mean() * 100:.2f}"
        )
        print(f"Arquivo: {destino}")

    print("\nNenhum município compartilhado entre os conjuntos.")


if __name__ == "__main__":
    main()
