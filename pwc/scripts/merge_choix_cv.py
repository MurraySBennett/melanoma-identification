from os import path
import pandas as pd

save_data = True
home_path = path.join(path.expanduser('~'), 'win_home', 'melanoma-identification', 'feature-rating')
paths = dict(cv_data=path.join(home_path, 'cv-data'),
        btl_data=path.join(home_path, 'btl-feature-data')
        )

btl_sym = pd.read_csv(path.join(paths['btl_data'],'choix-btl-symmetry.csv'), delimiter=',', header=0)
btl_sym = btl_sym[['id','pi']]
btl_sym.rename(columns={'pi':'pi_sym'}, inplace=True)

btl_bor = pd.read_csv(path.join(paths['btl_data'],'choix-btl-border.csv'), delimiter=',', header=0)
btl_bor = btl_bor[['id','pi']]
btl_bor.rename(columns={'pi':'pi_bor'}, inplace=True)

btl_col = pd.read_csv(path.join(paths['btl_data'],'choix-btl-colour.csv'), delimiter=',', header=0)
btl_col = btl_col[['id','pi']]
btl_col.rename(columns={'pi':'pi_col'}, inplace=True)

btl_global = pd.read_csv(path.join(paths['btl_data'], 'choix-btl-ugly.csv'), delimiter=',',header=0)
btl_global = btl_global[['id','pi']]
btl_global.rename(columns={'pi':'pi_global'}, inplace=True)

cv_data = pd.read_csv(path.join(paths['cv_data'], 'cv-data.csv'), delimiter=',', header=0)

# outer merge while-ever you have incomplete data.
data = cv_data\
        .merge(btl_sym, on='id', how='outer')\
        .merge(btl_bor, on='id', how='outer')\
        .merge(btl_col, on='id', how='outer')

if save_data:
    data.to_csv(path.join(paths['btl_data'],'choix-cv-data.csv'), index=False)
