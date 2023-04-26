import os 
import pandas as pd
import numpy as np


def main(save_data=False):
    home_path = os.path.join(os.path.expanduser('~'), 'win_home', 'melanoma-identification')
    paths = dict(
        home=home_path,
        data=os.path.join(home_path, "images", "metadata")
        )
    data = pd.read_csv(os.path.join(paths['data'], 'metadata.csv'), sep=',')
    data = data[["isic_id", "benign_malignant"]]
    # data['malignant'] = [1 if 'malignant' else 0 if 'benign' for i in data['benign_malignant']]
    data['malignant'] = data['benign_malignant'].apply(lambda label: 1 if label=='malignant' else 0)
    # data.drop(data[data.malignant == 0].index, inplace=True)
    
    print(data.head())
    if save_data:
        np.savetxt(r'malignant_ids.txt', data, fmt='%s,%s,%s', header='isic_id,b-m,malignant',comments='')


if __name__ == '__main__':
    main(save_data=True)
