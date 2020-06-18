from os import makedirs
from operator import itemgetter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import igraph

pd.set_option("max_columns", 100)
pd.set_option("colwidth", 400)

sns.set(context="notebook", style="darkgrid")

data_dir = "data"
results_dir = "results"

makedirs(results_dir, exist_ok=True)