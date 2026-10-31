import json
from datetime import date
from pathlib import Path

import pandas as pd


BRONZE = Path("data/bronze/ibge_renda_municipal_2022_original.csv")
SILVER = Path("data/silver/ibge_renda_municipal_2022.parquet")
RELATORIO = Path("reports/qualidade_renda_ibge_2022.json")


def main():
    original = pd.read_csv(
        BRONZE,
        sep=";",
        skiprows=6,
        header=None,
        names=["id_municipio", "municipio", "grupo_idade", "renda"],
        dtype=str,
        encoding="utf-8-sig",
    )

    silver = pd.read_parquet(SILVER)

    original["id_municipio"] = original["id_municipio"].str.strip()

    codigos_validos = original["id_municipio"].str.fullmatch(
        r"\d{7}", na=False
    )

    municipios_bronze = original.loc[codigos_validos].copy()
    registros_complementares = original.loc[~codigos_validos]

    codigos_bronze = set(municipios_bronze["id_municipio"])
    codigos_silver = set(silver["id_municipio"])

    renda = silver["renda_per_capita_2022"]

    validacoes = {
        "quantidade_municipios_5570": len(silver) == 5570,
        "codigos_unicos": not silver["id_municipio"].duplicated().any(),
        "codigos_com_sete_digitos": silver["id_municipio"].str.fullmatch(
            r"\d{7}", na=False
        ).all(),
        "sem_rendas_ausentes": renda.notna().all(),
        "rendas_positivas": (renda > 0).all(),
        "sem_perda_de_municipios": codigos_bronze == codigos_silver,
        "grupo_idade_total": municipios_bronze["grupo_idade"].eq(
            "Total"
        ).all(),
    }

    relatorio = {
        "fonte": "IBGE - Censo Demográfico 2022 - SIDRA",
        "tabela_sidra": 10295,
        "indicador": "Rendimento nominal médio mensal domiciliar per capita",
        "ano_referencia": 2022,
        "observacao_fonte": (
            "Resultados preliminares da amostra, conforme nota "
            "metodológica presente no arquivo original."
        ),
        "data_auditoria": date.today().isoformat(),
        "bronze": {
            "registros_lidos_apos_cabecalho": len(original),
            "registros_municipais": len(municipios_bronze),
            "registros_complementares": len(registros_complementares),
            "municipios_duplicados": int(
                municipios_bronze["id_municipio"].duplicated().sum()
            ),
        },
        "silver": {
            "registros": len(silver),
            "municipios_unicos": int(silver["id_municipio"].nunique()),
            "rendas_ausentes": int(renda.isna().sum()),
            "rendas_negativas": int((renda < 0).sum()),
            "rendas_iguais_zero": int((renda == 0).sum()),
            "renda_media_reais": round(float(renda.mean()), 2),
            "renda_mediana_reais": round(float(renda.median()), 2),
            "renda_minima_reais": round(float(renda.min()), 2),
            "renda_maxima_reais": round(float(renda.max()), 2),
        },
        "validacoes": {
            nome: bool(resultado)
            for nome, resultado in validacoes.items()
        },
        "aprovado": all(validacoes.values()),
    }

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    RELATORIO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== AUDITORIA DA SILVER ===")

    for nome, aprovado in validacoes.items():
        status = "OK" if aprovado else "FALHOU"
        print(f"{status}: {nome}")

    print()
    print("Relatório:", RELATORIO)
    print("Resultado geral:", "APROVADO" if relatorio["aprovado"] else "REPROVADO")

    if not relatorio["aprovado"]:
        raise ValueError("A Silver não passou em todas as validações.")


if __name__ == "__main__":
    main()
