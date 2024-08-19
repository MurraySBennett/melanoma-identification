from pathlib import Path
import pandas as pd

save_data = False
home_path = Path(__file__).resolve().parent.parent 
estimates = home_path / "data" / "estimates"

btl_sym = pd.read_csv(
    estimates / "btl-scores-symmetry.csv",
    delimiter = ",", header = 0
)
btl_sym.rename(
    columns = {"pi": "pi_sym"}, inplace = True
)

btl_bor = pd.read_csv(
    estimates / "btl-scores-border.csv",
    delimiter = ",", header = 0
)
btl_bor.rename(
    columns = {"pi": "pi_bor"}, inplace = True
)

btl_col = pd.read_csv(
    estimates / "btl-scores-colour.csv",
    delimiter = ",", header = 0
)
btl_col.rename(
    columns = {"pi": "pi_col"}, inplace = True
)

#btl_global = pd.read_csv(path.join(paths['btl_data'], 'btl-scores-global.csv'), delimiter=',',header=0)
#btl_global.rename(columns={'r':'r_global'}, inplace=True)

cv_data = pd.read_csv(
    estimates / "cv-data.csv",
    delimiter = ",", header = 0
)

# outer merge while-ever you have incomplete data.
data = cv_data\
        .merge(btl_sym, on="id", how="outer")\
        .merge(btl_bor, on="id", how="outer")\
        .merge(btl_col, on="id", how="outer")

if save_data:
    data.to_csv(
        estimates / "btl-cv-data.csv",
        index = False
    )

btl_only = ["id", "malignant", "pi_sym", "pi_bor", "pi_col"]
data = data[btl_only]

if save_data:
    data.to_csv(
        estimates / "btl-data.csv",
        index = False
    )
