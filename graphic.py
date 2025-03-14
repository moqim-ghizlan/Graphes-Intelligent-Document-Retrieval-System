"""
==================================================
Visualization Utility Functions for Data Analysis
==================================================
This script provides a collection of functions for:
- Plotting subgraphs and analyzing graph structures.
- Visualizing classification and clustering results.
- Generating various types of plots such as:
  - Scatter plots
  - Line plots
  - Heatmaps
  - Confusion matrix visualizations

These functions facilitate the interpretation of machine learning models,
graph structures, and clustering assignments.

Author: JUILLARD Thibaut and GHIZLAN Moqim
@@ ChatGPT and Github copilot were used to write this code specifically to generate the documentation and the comments.
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_dense_subgraph(G, top_n=20):
    """
    Plots a dense subgraph using the top-degree nodes.

    Parameters:
    - G (networkx.Graph): The input graph.
    - top_n (int, optional): Number of top-degree nodes to include in the subgraph (default: 20).

    Displays:
    - A visual representation of the subgraph with top-degree nodes.
    """

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
    subgraph = G.subgraph(top_nodes)

    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray')
    plt.title("Visualization of a dense subgraph (Top degree nodes)")
    plt.show()


def analyze_connected_components(G):
    """
    Analyzes the connected components of the graph.

    Parameters:
    - G (networkx.Graph): The input graph.

    Prints:
    - Total number of connected components.
    - Size of the top 5 largest components.
    - A histogram displaying the distribution of component sizes.
    """

    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]

    print(f"Total connected components: {len(components)}")
    print(f"Largest components (top 5): {sorted(component_sizes, reverse=True)[:5]}")

    plt.figure(figsize=(10, 5))
    plt.hist(component_sizes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Component size")
    plt.ylabel("Number of components")
    plt.title("Distribution of connected component sizes")
    plt.show()


def create_conf_matrix_graphic(data, xlabel, ylabel, title):
    """
    Creates and displays a heatmap visualization of a confusion matrix.

    Parameters:
    - data (np.ndarray or pd.DataFrame): The confusion matrix to be visualized.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.

    Displays:
    - A heatmap of the confusion matrix.
    """

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(1, 9),
        yticklabels=range(1, 9),
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def create_plot_graphic(x, y, marker, linestyle, xlabel, ylabel, title):
    """
    Creates a line plot visualization.

    Parameters:
    - x (array-like): Data for the x-axis.
    - y (array-like): Data for the y-axis.
    - marker (str): Marker style for the points.
    - linestyle (str): Line style for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.

    Displays:
    - A line plot.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker=marker, linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def create_scatter_graphic(x, y, c, cmap, alpha, title, xlabel, ylabel, colorbar_label):
    """
    Creates a scatter plot visualization.

    Parameters:
    - x (array-like): Data for the x-axis.
    - y (array-like): Data for the y-axis.
    - c (array-like): Colors for the points.
    - cmap (str): Colormap for the points.
    - alpha (float): Opacity level of the points.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - colorbar_label (str): Label for the colorbar.

    Displays:
    - A scatter plot with a color gradient.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=c, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=colorbar_label)
    plt.show()


def create_heatmap_graphic(data, cmap, linewidths, vmax, vmin, title):
    """
    Creates and displays a heatmap visualization.

    Parameters:
    - data (np.ndarray or pd.DataFrame): The data to be visualized.
    - cmap (str): Colormap to use.
    - linewidths (float): Width of the lines separating cells.
    - vmax (float): Maximum value for the color scale.
    - vmin (float): Minimum value for the color scale.
    - title (str): Title of the heatmap.

    Displays:
    - A heatmap of the provided data.
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap=cmap, linewidths=linewidths, vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.show()
