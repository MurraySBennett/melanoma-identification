import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    base_dir = Path(os.getcwd()).parent.parent
    data_dir = base_dir / 'melnet' / 'data' / 'activations'
    fig_dir = base_dir / 'melnet' / 'figures' / 'dim_reductions'

    data = pd.read_csv(data_dir / 'best_model_avg_pool_activations.csv')
    ids = data['id']
    activations = data.drop('id', axis=1).values

    n_components = 40

    # SVD
    svd_reduced, svd_cumulative_variance, svd_explained_variance = apply_svd(activations, n_components)
    svd_columns = [f"SVD_{i+1}" for i in range(n_components)]
    svd_df = pd.DataFrame(svd_reduced, columns=svd_columns)

    # PCA
    pca_reduced, pca_cumulative_variance, pca_explained_variance = apply_pca(activations, n_components)
    pca_columns = [f"PCA_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_reduced, columns=pca_columns)

    # t-SNE (only for 2D)
    tsne_df = pd.DataFrame()
    tsne_reduced = apply_tsne(activations, 2)
    tsne_columns = [f"tSNE_{i+1}" for i in range(2)]
    tsne_df = pd.DataFrame(tsne_reduced, columns=tsne_columns)


    combined_df = pd.concat([svd_df, pca_df, tsne_df], axis=1)
    combined_df.insert(0, 'id', ids)
    combined_df = combined_df.round(3)
    output_filename = f"combined_reduced_activations_{n_components}d.csv"
    output_path = data_dir / output_filename
    combined_df.to_csv(output_path, index=False)
    print(f"Combined reduced activations ({n_components}D) saved to: {output_path}")

    # Plotting
    plot_results(svd_reduced, pca_reduced, tsne_reduced, n_components, fig_dir, svd_cumulative_variance, pca_cumulative_variance)

    plot_explained_variance(svd_explained_variance, svd_cumulative_variance, "SVD", fig_dir)
    plot_explained_variance(pca_explained_variance, pca_cumulative_variance, "PCA", fig_dir)

def apply_svd(activations, n_components):
    svd = TruncatedSVD(n_components=n_components)
    reduced_activations = svd.fit_transform(activations)
    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(f"SVD Cumulative Variance: {cumulative_variance}")
    return reduced_activations, cumulative_variance, explained_variance_ratio

def apply_pca(activations, n_components):
    pca = PCA(n_components=n_components)
    reduced_activations = pca.fit_transform(activations)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(f"PCA Cumulative Variance: {cumulative_variance}")
    return reduced_activations, cumulative_variance, explained_variance_ratio

def apply_tsne(activations, n_components):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_activations = tsne.fit_transform(activations)
    return reduced_activations

def plot_results(svd_data, pca_data, tsne_data, n_components, figure_path, svd_cumulative_variance, pca_cumulative_variance):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(svd_data[:, 0], svd_data[:, 1])
    plt.title("SVD 2D")
    if svd_cumulative_variance is not None:
        plt.text(0.95, 0.95, f"Cumulative Variance: {svd_cumulative_variance[1]:.3f}",
                 transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top')

    plt.subplot(1, 3, 2)
    plt.scatter(pca_data[:, 0], pca_data[:, 1])
    plt.title("PCA 2D")
    if pca_cumulative_variance is not None:
        plt.text(0.95, 0.95, f"Cumulative Variance: {pca_cumulative_variance[1]:.3f}",
                 transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top')

    if tsne_data is not None:
        plt.subplot(1, 3, 3)
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
        plt.title("t-SNE 2D")
    plt.tight_layout()
    plt.savefig(figure_path / "combined_reduced_2d.png")
    plt.show()


    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(svd_data[:, 0], svd_data[:, 1], svd_data[:, 2])
    ax1.set_title("SVD 3D")
    if svd_cumulative_variance is not None:
        ax1.text2D(0.05, 0.95, f"Cumulative Variance: {svd_cumulative_variance[2]:.3f}",
                   transform=ax1.transAxes, verticalalignment='top')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2])
    ax2.set_title("PCA 3D")
    if pca_cumulative_variance is not None:
        ax2.text2D(0.05, 0.95, f"Cumulative Variance: {pca_cumulative_variance[2]:.3f}",
                   transform=ax2.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(figure_path / "combined_reduced_3d.png")
    plt.show()

def plot_explained_variance(explained_variance_ratio, cumulative_variance, method_name, figure_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(explained_variance_ratio)
    plt.xlabel("Component Number")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{method_name} Explained Variance")

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance)
    plt.xlabel("Component Number")
    plt.ylabel("Cumulative Variance Ratio")
    plt.title(f"{method_name} Cumulative Variance")

    plt.tight_layout()
    plt.savefig(figure_path / f"{method_name}_explained_variance.png")
    plt.show()

if __name__ == '__main__':
    main()