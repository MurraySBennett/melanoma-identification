import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pareto_front import pareto_frontier, plot3d
from read_data import set_paths, read_data


def main():
    """ import data, get pareto, plot, return values.
    """
    home_path = "/mnt/c/Users/qlm573/melanoma-identification/feature-rating/"
    paths = set_paths(home_path)
    df = read_data(paths)

    hyper_parms = dict(
        n_trials=np.unique(df["n_trials"]),
        n_images=np.unique(df["n_players"])
    )

    df_summary = df.groupby(["n_players", "n_trials"], as_index=False).agg("mean")

    threshold = 0.95
    df_pareto = pareto_frontier(df_summary, threshold)

    x = df_summary["n_players"]
    y = df_summary["n_trials"]
    z = df_summary["connected"]
   
    plot3d(x, y, z, pareto=df_pareto)
    plt.show()
    
    print(df_pareto)

if __name__ == '__main__':
    main()
