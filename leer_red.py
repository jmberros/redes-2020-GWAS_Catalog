import pandas as pd
import networkx as nx


def leer_red_df():
    columnas = [
        "gen_mapeado_elegido",
        "alelo_riesgo",
        # "OR_or_beta",
        "fenotipo",
        "categoria_fenotipo"
    ]

    fp = "results/gwas_cat.filtrado.tsv.gz"
    assoc = pd.read_table(fp, usecols=columnas)[columnas]
    assoc = assoc.sort_values(by=["categoria_fenotipo", "fenotipo", "gen_mapeado_elegido"])
    assoc = assoc.reset_index(drop=True)
    
    return assoc

def leer_subred_df():
    fp = "results/subred.tsv"
    return pd.read_table(fp)

def leer_red_nx():
    return nx.read_gml("results/red_completa.gml")

def leer_subred_nx():
    return nx.read_gml("results/subred.gml")