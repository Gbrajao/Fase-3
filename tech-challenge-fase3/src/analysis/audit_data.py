
from pathlib import Path
import pandas as pd

# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

def auditar_arquivo(caminho):
    print("\n" + "=" * 70)
    print(f"ARQUIVO: {caminho.name}")
    print(f"CAMADA: {caminho.parent.name.upper()}")
    print("=" * 70)

    try:
        df = pd.read_parquet(caminho)

        print(f"Quantidade de registros: {len(df)}")
        print(f"Quantidade de colunas: {len(df.columns)}")

        print("\nCOLUNAS E TIPOS:")
        print(df.dtypes.to_string())

        print("\nVALORES AUSENTES:")
        nulos = df.isnull().sum()
        print(nulos[nulos > 0].to_string()
              if nulos.sum() > 0 else "Nenhum")

        print(f"\nRegistros duplicados: {df.duplicated().sum()}")

        print("\nAMOSTRA DOS DADOS:")
        print(df.head(3).to_string(index=False))

    except Exception as erro:
        print(f"ERRO ao processar {caminho.name}: {erro}")
        raise


def main():
    for camada in ["gold", "silver"]:
        pasta = DATA_DIR / camada

        print(f"\n\n######## CAMADA {camada.upper()} ########")

        arquivos = sorted(pasta.glob("*.parquet"))

        if not arquivos:
            print("Nenhum arquivo encontrado.")
            continue

        for arquivo in arquivos:
            auditar_arquivo(arquivo)


if __name__ == "__main__":
    main()
