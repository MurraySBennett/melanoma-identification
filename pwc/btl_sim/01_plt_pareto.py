import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def main():
    """ import data, get pareto, plot, return values."""
    
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "simulation_p10-250_images5000-75000.csv")) 
    
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
    plt.savefig(os.path.join(here, "btl_pareto.pdf"), format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    
    print(df_pareto)


def pareto_frontier(df, threshold):
    """ Return Pareto frontier for our specific case."""
    df = df[df["connected"] >= threshold].reset_index(drop=True)
    df = df.groupby("n_players")[["n_trials", "connected"]].min().reset_index()
    return df


def plot3d(x, y, z, pareto=None):
    """ plot 3d data specific to simulation."""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d')

    ax.set_xlabel("$\it{n}$ Images")
    ax.set_ylabel("$\it{n}$ Trials")
    ax.set_zlabel("P(connected)")
    ax.set_zlim([0, 1])

    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    Z = np.empty(x_vals.shape + y_vals.shape)
    Z.fill(np.nan)
    Z[x_idx, y_idx] = z
    Z = Z.T

    ax.plot_surface(X, Y, Z, cmap="plasma", zorder=-1)

    if pareto is not None:
        x = pareto["n_players"]
        y = pareto["n_trials"]
        z = pareto["connected"]

        n_vals = 100
        new_x = np.linspace(0, len(x) - 1, n_vals)
        new_y = np.linspace(0, len(y) - 1, n_vals)
        new_z = np.linspace(0, len(z) - 1, n_vals)

        new_x = np.interp(new_x, range(len(x)), x)
        new_y = np.interp(new_y, range(len(y)), y)
        new_z = np.interp(new_z, range(len(z)), z)

        ax.scatter(new_x, new_y, new_z, c='black', zorder=1,s=20)
        ax.scatter(new_x, new_y, new_z, c='#ff6ec7', zorder=0,s=10)
        ax.view_init(elev=30, azim=-80)


if __name__ == '__main__':
    main()

