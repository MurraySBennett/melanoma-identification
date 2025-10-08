from os import path
import pandas as pd
import numpy as np
import choix

def main(feature, save_data):
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/"
    data_path = path.join(home_path, "feature-rating", "btl-feature-data")
    for f in feature: 
        if f == "symmetry":
            data = pd.read_csv(path.join(data_path, 'btl-asymmetry.csv'))
        elif f == "border":
            data = pd.read_csv(path.join(data_path, 'btl-border.csv'))
        elif f == "colour":
            data = pd.read_csv(path.join(data_path, 'btl-colour.csv'))
        elif f == "ugly":
            data = pd.read_csv(path.join(data_path, 'data-processed.csv'))
        print(f'working on {f}')

        labels = set(data['winner'].tolist() + data['loser'].tolist())
        # mapping between labels and IDs
        img_id = {label: i for i, label in enumerate(labels)}
        data['winner_id'] = data['winner'].map(img_id)
        data['loser_id'] = data['loser'].map(img_id)
        choix_data = [(row['winner_id'], row['loser_id']) for _, row in data.iterrows()]
        data = data.sort_values(by='winner_id')

        n_items = len(labels)
        alpha = 0.1 # 0 = no regularization, a little (0.01) regularization can help with sparsity
        initial_params = None # None = the function initialises parameters
        max_iter = 200 # max iterations for optimization
        tol = 1e-8 # convergence tolerance

        params = np.exp(choix.ilsr_pairwise(n_items, choix_data, alpha=alpha, initial_params=initial_params, max_iter=max_iter, tol=tol))
        rank = np.argsort(params)
        # print(img_id)

        data = pd.DataFrame({'id': list(labels), 'int': [img_id[label] for label in labels], 'rank': rank})
        data = data.sort_values(by='rank')
        data['pi'] = sorted(params)
        data = data.sort_values(by='pi')

        if save_data:
            data.to_csv(path.join(data_path, 'choix-btl-' + f + '.csv'), index=False)
   


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Apply BTL to specified feature data")
    # parser.add_argument("feature", choices=["symmetry", "border", "colour"], help="Feature to process")
    # args = parser.parse_args()
    # main(args.feature)
    # feature=None
    feature = ["symmetry", "border", "colour", "ugly"]
    save_data = True
    main(feature, save_data)


