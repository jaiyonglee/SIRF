# Load basic libraries
import numpy as np
import pandas as pd
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import geopandas as gpd
from shapely.geometry import Polygon
import warnings

# Load libraries from parent directory
from sirf_code.library.Data_Scaling import copy_position_to_link_data


# Initialize position
def initialize_position(df_bus_scaled, df_branch_regressed):
    # Get min, max of latitude, longitude
    min_lat = df_bus_scaled["Latitude"].min()
    max_lat = df_bus_scaled["Latitude"].max()
    min_long = df_bus_scaled["Longitude"].min()
    max_long = df_bus_scaled["Longitude"].max()

    # Randomize the position between min and max if the position is nan
    df_bus_initialized = df_bus_scaled.copy()
    df_bus_initialized["Latitude"] = df_bus_initialized["Latitude"].apply(
        lambda x: np.random.uniform(min_lat, max_lat) if np.isnan(x) else x
    )
    df_bus_initialized["Longitude"] = df_bus_initialized["Longitude"].apply(
        lambda x: np.random.uniform(min_long, max_long) if np.isnan(x) else x
    )

    # Add location and distance information to edge data
    df_branch_initialized = copy_position_to_link_data(
        df_bus_initialized, df_branch_regressed
    )

    return df_bus_initialized, df_branch_initialized


# Plot initialization figure
def plot_initialization_figure(
    path, retention_rate, iter_num, node_size, nation, boundary, truth_enabled, figsize=(40, 40)
):
    """
    Plot initialization figure.

    Parameters:
        path (str): The path to data files.
        retention_rate (float): Retention rate.
        iter_num (int): Iteration number.
        nation (str): Nation name.
        boundary (tuple): Boundaries for the plot (left, right, top, bottom).
        figsize (tuple, optional): Figure size. Defaults to (40, 40).
    """
    # Constants
    NODE_SIZE = node_size
    COLORS = ["F8E9A1", "F5F3ED"]

    # Load data
    bus_init_path = os.path.join(
        path,
        f"Result/Initialized_data/Bus_initialized/Bus_initialized_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    branch_init_path = os.path.join(
        path,
        f"Result/Initialized_data/Branch_initialized/Branch_initialized_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    bus_nan_path = os.path.join(
        path,
        f"Result/Preprocessed_data/Nan_position/Bus_nan_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    parent_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
    map_path = os.path.join(parent_dir, "Maps/", nation + ".geojson")

    df_bus_initialized = pd.read_csv(bus_init_path)
    df_branch_initialized = pd.read_csv(branch_init_path)
    df_bus_nan = pd.read_csv(bus_nan_path)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(df_branch_initialized["Node1"])
    G.add_nodes_from(df_branch_initialized["Node2"])
    G.add_edges_from(df_branch_initialized[["Node1", "Node2"]].values)

    # Create position dictionary
    pos = {
        node: (
            df_bus_initialized.loc[
                df_bus_initialized["Node"] == node, "Longitude"
            ].iloc[0],
            df_bus_initialized.loc[df_bus_initialized["Node"] == node, "Latitude"].iloc[
                0
            ],
        )
        for node in G.nodes
    }

    # Load map data
    try:
        with open(map_path, encoding="utf-8") as f:
            map_data = json.load(f)

        circuits = []
        for circuit in map_data["features"]:
            for polygon in circuit["geometry"]["coordinates"]:
                polygon_coordinates = []
                for coordinate in polygon:
                    polygon_coordinates.append((coordinate[0], coordinate[1]))
                circuits.append(polygon_coordinates)

        # Plot initialization figure
        fig, ax = plt.subplots(figsize=figsize)
        coll = PolyCollection(
            circuits, edgecolors="black", linewidths=1, facecolors="white"
        )
        ax.add_collection(coll)
    except:
        bbox = Polygon(
            [
                (boundary[0], boundary[2]),
                (boundary[1], boundary[2]),
                (boundary[1], boundary[3]),
                (boundary[0], boundary[3]),
            ]
        )
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry = [bbox])
        warnings.simplefilter(action='ignore', category=FutureWarning)
        map_boundary = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        map_boundary = map_boundary[map_boundary.continent == nation]
        map_boundary = map_boundary.overlay(bbox_gdf, how="intersection")

        # Plot initialization figure
        fig, ax = plt.subplots(figsize=figsize)
        map_boundary.boundary.plot(ax=ax, linewidth=1, color="black")
        map_boundary.plot(linewidth=0.8, ax=ax, edgecolor='k', legend=True,color ='white')
        ax.set_axis_off()

    # Hide axes outlines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set font
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"

    # Set legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Initialized",
            markersize=np.sqrt(NODE_SIZE),
            markerfacecolor=mcolors.to_rgb("#" + COLORS[0]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Known" if truth_enabled else "Searched",
            markersize=np.sqrt(NODE_SIZE),
            markerfacecolor=mcolors.to_rgb("#" + COLORS[1]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
    ]

    # Set node colors
    node_colors = [
        COLORS[0] if node in df_bus_nan["Node"].values else COLORS[1]
        for node in G.nodes
    ]
    normalized_colors = [mcolors.to_rgb("#" + color) for color in node_colors]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=int(NODE_SIZE / 4),
        alpha=1,
        node_color=normalized_colors,
        edgecolors="black",
        linewidths=3,
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.3, width=3)  # inferring

    # Add legend, adjust fontsize, and set 2x2 grid
    ax.legend(
        handles=[legend_elements[1], legend_elements[0]],
        fontsize=80,
        loc="upper right",
        ncol=2,
    )

    # Adjust margins
    ax.margins(x=0, y=0)
    ax.set_xlim(left=boundary[0], right=boundary[1])
    ax.set_ylim(top=boundary[2], bottom=boundary[3])

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Initialization/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir,
            f"Initialization_{round(retention_rate * 100)}p_{iter_num:04d}.pdf",
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()
