from os import path
import pandas as pd

save_data = True
home_path = path.join(path.expanduser('~'), 'win_home', 'melanoma-identification', 'feature-rating')
paths = dict(cv_data=path.join(home_path, 'cv-data'),
        btl_data=path.join(home_path, 'btl-feature-data')
        )

btl_sym = pd.read_csv(path.join(paths['btl_data'],'btl-scores-symmetry.csv'), delimiter=',', header=0)
btl_sym.rename(columns={'r':'r_sym'}, inplace=True)

btl_bor = pd.read_csv(path.join(paths['btl_data'],'btl-scores-border.csv'), delimiter=',', header=0)
btl_bor.rename(columns={'r':'r_bor'}, inplace=True)

btl_col = pd.read_csv(path.join(paths['btl_data'],'btl-scores-colour.csv'), delimiter=',', header=0)
btl_col.rename(columns={'r':'r_col'}, inplace=True)

cv_data = pd.read_csv(path.join(paths['cv_data'], 'cv-data.csv'), delimiter=',', header=0)

# outer merge while-ever you have incomplete data.
data = cv_data\
        .merge(btl_sym, on='id', how='outer')\
        .merge(btl_bor, on='id', how='outer')\
        .merge(btl_col, on='id', how='outer')

if save_data:
    data.to_csv(path.join(paths['btl_data'],'btl-cv-data.csv'), index=False)
