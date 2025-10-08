import pandas as pd
from pathlib import Path

SAVE_DATA = True
home_path = Path(__file__).resolve().parent.parent
estimates = home_path / "data" / "estimates"

btl_sym = pd.read_csv( estimates / "btl_scores_symmetry.csv", delimiter = ",", header = 0)
btl_bor = pd.read_csv( estimates / "btl_scores_border.csv", delimiter = ",", header = 0)
btl_col = pd.read_csv( estimates / "btl_scores_colour.csv", delimiter = ",", header = 0)

btl_sym.rename( columns = {"pi": "pi_sym"}, inplace = True)
btl_bor.rename( columns = {"pi": "pi_bor"}, inplace = True)
btl_col.rename( columns = {"pi": "pi_col"}, inplace = True)

cv_data = pd.read_csv( estimates / "cv-data.csv", delimiter = ",", header = 0)

# outer merge while-ever you have incomplete data.
data = cv_data\
        .merge(btl_sym, on="id", how="outer")\
        .merge(btl_bor, on="id", how="outer")\
        .merge(btl_col, on="id", how="outer")

if SAVE_DATA:
    data.to_csv( estimates / "btl_cv_data.csv", index = False)

btl_only = ["id", "malignant", "pi_sym", "pi_bor", "pi_col"]
data = data[btl_only]

if SAVE_DATA:
    data.to_csv( estimates / "btl_data.csv", index = False)
