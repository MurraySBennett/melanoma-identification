import os
import glob
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pareto_frontier_multi(array):
    # sort on first dimension
    array = array[array[:0].argsort()]
    # add first row to pareto frontier
    pareto_frontier = array[0:1, :]
    for r in array[1:,:]:
        if sum([r[x] >= pareto_frontier[-1][x] for x in range(len(r))]) == len(r):
            pareto_frontier = np.concatenate((pareto_frontier, [r]))
    return pareto_frontier


home_path = os.path.join(os.path.expanduser("~"), "melanoma-identification", "feature-rating")
paths = dict(
    home=home_path,
    data=os.path.join(home_path, "btl-simulation", "simulation-data"),
    scripts=os.path.join(home_path, "scripts")
)
files = glob.glob(paths["data"] + "\simulation*.csv")
df = pd.read_csv(files[0])
df.replace([np.inf, -np.inf], np.nan, inplace=True)

hyper_parms = dict(
    n_trials=np.unique(df["n_trials"]),
    n_images=np.unique(df["n_players"])
)

# df = df.loc[ (df["n_trials"] >= hyper_parms["n_trials"][-2]) & (df["n_players"] >= hyper_parms["n_images"][-2]) ]
df_summary = df.groupby(["n_players", "n_trials"], as_index=False).agg("mean")

print(df_summary)

z_var = "connected" # "n_components" #"sp_rho"

x = df_summary["n_players"]
y = df_summary["n_trials"]
z = df_summary[z_var]

print(np.corrcoef(df_summary["connected"], df_summary["sp_rho"]))


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax = plt.axes(projection="3d")
# ax.scatter(x, y, z+0.005)
ax.set_title("BTL Simulations")
ax.set_xlabel("$\it{n}$ Images")
ax.set_ylabel("$\it{n}$ Trials")
ax.set_zlabel(z_var)
# ax.set_zlim([0, 1])

# plt.show()

# X, Y = np.meshgrid(x, y)
X, Y = np.meshgrid(np.unique(x), np.unique(y))
x_vals, x_idx = np.unique(x, return_inverse=True)
y_vals, y_idx = np.unique(y, return_inverse=True)
Z = np.empty(x_vals.shape + y_vals.shape)
Z.fill(np.nan)
Z[x_idx, y_idx] = z
Z = Z.T

# ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="plasma")
plt.show()

pareto_data = df_summary[["n_trials", "n_players", "sp_rho", "connected"]]
# set some goals/objectives
pareto_data = pareto_data[(pareto_data["connected"] > 0.9) & (pareto_data["sp_rho"] > 0.5)]
# maximise images and rho
