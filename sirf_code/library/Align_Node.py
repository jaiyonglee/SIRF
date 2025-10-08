# Load basic libraries
import networkx as nx
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
import geopandas as gpd
from shapely.geometry import Polygon
import warnings

# Load libraries from parent directory
from sirf_code.library.Data_Scaling import copy_position_to_link_data


def align_dangling_nodes(df_bus, df_branch, df_bus_nan, n_squared=1):

    # Copy the input data
    df_bus_copy = df_bus.copy()
    df_branch_copy = df_branch.copy()

    # Convert df_bus_nan to np.array
    bus_nan = df_bus_nan["Node"].to_numpy()

    # Store position to dictionary for nodes in bus_nan
    bus_pos = {}
    for i in df_bus_copy["Node"]:
        bus_pos[i] = df_bus_copy.loc[
            df_bus_copy["Node"] == i, ["Latitude", "Longitude"]
        ].values.flatten()

    # Get graph G_grid
    G_grid = nx.from_pandas_edgelist(
        df_branch_copy,
        source="Node1",
        target="Node2",
        edge_attr="Predicted_distance",
    )

    # Set node attributes
    nx.set_node_attributes(G_grid, bus_pos, "Position")

    # Align the position of the dangling nodes
    updated_pos = bus_pos.copy()
    dangling_nodes = []

    for node in G_grid.nodes():
        if G_grid.degree[node] == 1 and node in bus_nan:
            neighbors = list(G_grid.neighbors(node))
            neighbor = neighbors[0]
            node_pos = bus_pos[node]
            neighbor_pos = bus_pos[neighbor]

            x0_x1 = (
                (node_pos[0] - neighbor_pos[0]) ** 2
                + (node_pos[1] - neighbor_pos[1]) ** 2
            ) ** 0.5

            neighbor_neighbors = list(G_grid.neighbors(neighbor))
            neighbor_neighbors.remove(node)
            neighbor_dir_sum = [0.0, 0.0]

            for n in neighbor_neighbors:
                x1_x2 = (
                    (neighbor_pos[0] - bus_pos[n][0]) ** 2
                    + (neighbor_pos[1] - bus_pos[n][1]) ** 2
                ) ** 0.5

                # Update the position to match the orientation of the vectors
                x1_x2_dir = (
                    neighbor_pos[0] - bus_pos[n][0],
                    neighbor_pos[1] - bus_pos[n][1],
                )
                if x1_x2**n_squared != 0:
                    neighbor_dir_sum[0] += x1_x2_dir[0] / x1_x2**n_squared
                    neighbor_dir_sum[1] += x1_x2_dir[1] / x1_x2**n_squared
                else:
                    neighbor_dir_sum[0] += 0
                    neighbor_dir_sum[1] += 0

            if any(x != 0 for x in neighbor_dir_sum):
                x1_x2_length = (
                    neighbor_dir_sum[0] ** 2 + neighbor_dir_sum[1] ** 2
                ) ** 0.5  # Change the order of latitude and longitude
                updated_node_pos = (
                    neighbor_pos[0] + x0_x1 * neighbor_dir_sum[0] / x1_x2_length,
                    neighbor_pos[1] + x0_x1 * neighbor_dir_sum[1] / x1_x2_length,
                )
                # print(x0_x1, haversine(updated_node_pos[::-1], neighbor_pos[::-1]))
                updated_pos[node] = updated_node_pos
                dangling_nodes.append(node)

    # Update df_bus_copy with the updated position
    df_bus_copy["Latitude"] = df_bus_copy["Node"].map(lambda x: updated_pos[x][0])
    df_bus_copy["Longitude"] = df_bus_copy["Node"].map(lambda x: updated_pos[x][1])

    # Update df_branch
    df_branch_copy = copy_position_to_link_data(df_bus_copy, df_branch_copy)

    # Return the updated position and the list of dangling nodes
    return df_bus_copy, df_branch_copy


# Plot Alignment figure
def plot_alignment_figure(
    path,
    retention_rate,
    iter_num,
    node_size,
    nation,
    boundary,
    truth_enabled,
    figsize=(40, 40),
):
    """
    Plot alignment figure.

    Parameters:
        path (str): The path to data files.
        retention_rate (float): Retention rate.
        iter_num (int): Iteration number.
        nation (str): Nation name.
        boundary (tuple): Boundaries for the plot (left, right, top, bottom).
        figsize (tuple, optional): Figure size. Defaults to (40, 40).
        truth_enabled (bool, optional): Flag to enable/disable truth plot. Defaults to True.
    """
    # Constants
    NODE_SIZE = node_size
    COLORS = ["F5F3ED", "F5F3ED", "FACD47", "97D5B3", "FA986E"]

    # Load data
    bus_init_path_1 = os.path.join(
        path,
        f"Result/Inferred_data/Bus_inferred/Bus_inferred_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    bus_init_path_2 = os.path.join(
        path,
        f"Result/Aligned_data/Bus_aligned/Bus_aligned_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    branch_init_path_2 = os.path.join(
        path,
        f"Result/Aligned_data/Branch_aligned/Branch_aligned_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    bus_init_path_3 = os.path.join(path, f"Result/Dataset/Bus.csv")
    bus_nan_path = os.path.join(
        path,
        f"Result/Inferred_data/Nan_position/Bus_nan_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    parent_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
    map_path = os.path.join(parent_dir, "Maps/", nation + ".geojson")

    df_bus_inferred = pd.read_csv(bus_init_path_1)
    df_bus_aligned = pd.read_csv(bus_init_path_2)
    df_branch_aligned = pd.read_csv(branch_init_path_2)
    df_bus_truth = pd.read_csv(bus_init_path_3)
    df_bus_nan = pd.read_csv(bus_nan_path)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(df_branch_aligned["Node1"])
    G.add_nodes_from(df_branch_aligned["Node2"])
    G.add_edges_from(df_branch_aligned[["Node1", "Node2"]].values)

    # Get dangling nodes from G
    dangling_nodes = [node for node in G.nodes if G.degree[node] == 1]

    # Create position dictionary
    pos = {
        node: (
            df_bus_inferred.loc[df_bus_inferred["Node"] == node, "Longitude"].iloc[0],
            df_bus_inferred.loc[df_bus_inferred["Node"] == node, "Latitude"].iloc[0],
        )
        for node in G.nodes
    }
    pos_new = {
        node: (
            df_bus_aligned.loc[df_bus_aligned["Node"] == node, "Longitude"].iloc[0],
            df_bus_aligned.loc[df_bus_aligned["Node"] == node, "Latitude"].iloc[0],
        )
        for node in G.nodes
    }
    if truth_enabled:
        pos_truth = {
            node: (
                df_bus_truth.loc[df_bus_truth["Node"] == node, "Longitude_truth"].iloc[
                    0
                ],
                df_bus_truth.loc[df_bus_truth["Node"] == node, "Latitude_truth"].iloc[
                    0
                ],
            )
            for node in df_bus_truth["Node"].values
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
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[bbox])
        warnings.simplefilter(action="ignore", category=FutureWarning)
        map_boundary = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        map_boundary = map_boundary[map_boundary.continent == nation]
        map_boundary = map_boundary.overlay(bbox_gdf, how="intersection")

        # Plot initialization figure
        fig, ax = plt.subplots(figsize=figsize)
        map_boundary.boundary.plot(ax=ax, linewidth=1, color="black")
        map_boundary.plot(
            linewidth=0.8, ax=ax, edgecolor="k", legend=True, color="white"
        )
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
            label="Non-Dangling",
            markersize=np.sqrt(NODE_SIZE),  # Known
            markerfacecolor=mcolors.to_rgb("#" + COLORS[0]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
        # plt.Line2D([0], [0], marker='o', color='w', label='Dangling', markersize=np.sqrt(NODE_SIZE),
        #              markerfacecolor=mcolors.to_rgb('#' + COLORS[1]), markeredgecolor='black', markeredgewidth=5),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Before align",
            markersize=np.sqrt(NODE_SIZE),
            markerfacecolor=mcolors.to_rgb("#" + COLORS[2]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Aligned",
            markersize=np.sqrt(NODE_SIZE),
            markerfacecolor=mcolors.to_rgb("#" + COLORS[3]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Ground truth",
            markersize=np.sqrt(NODE_SIZE),
            markerfacecolor=mcolors.to_rgb("#" + COLORS[4]),
            markeredgecolor="black",
            markeredgewidth=5,
        ),
    ]

    # Set node colors
    node_colors = [
        (
            COLORS[0]
            if node not in df_bus_nan["Node"].values
            else (COLORS[4] if node in dangling_nodes else COLORS[0])
        )
        for node in G.nodes
    ]
    normalized_colors = [mcolors.to_rgb("#" + color) for color in node_colors]
    if truth_enabled:
        nx.draw_networkx_nodes(
            G,
            pos_truth,
            node_size=int(NODE_SIZE / 4),
            alpha=1,
            node_color=normalized_colors,
            edgecolors="black",
            linewidths=3,
        )
        nx.draw_networkx_edges(
            G, pos_truth, edge_color="gray", alpha=0.3, width=3
        )  # Truth

    node_colors = [
        (
            COLORS[0]
            if node not in df_bus_nan["Node"].values
            else (COLORS[2] if node in dangling_nodes else COLORS[0])
        )
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
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.3, width=3)  # inferred

    node_colors = [
        (
            COLORS[0]
            if node not in df_bus_nan["Node"].values
            else (COLORS[3] if node in dangling_nodes else COLORS[0])
        )
        for node in G.nodes
    ]
    normalized_colors = [mcolors.to_rgb("#" + color) for color in node_colors]
    nx.draw_networkx_nodes(
        G,
        pos_new,
        node_size=int(NODE_SIZE / 4),
        alpha=1,
        node_color=normalized_colors,
        edgecolors="black",
        linewidths=3,
    )
    nx.draw_networkx_edges(
        G, pos_new, edge_color="black", alpha=0.8, width=3
    )  # Aligned

    # Add legend, adjust fontsize, and set 2x2 grid
    if truth_enabled:
        ax.legend(handles=legend_elements, fontsize=80, loc="upper right", ncol=2)
    else:
        ax.legend(handles=legend_elements[:-1], fontsize=80, loc="upper right", ncol=1)

    # Adjust margins
    ax.margins(x=0, y=0)
    ax.set_xlim(left=boundary[0], right=boundary[1])
    ax.set_ylim(top=boundary[2], bottom=boundary[3])

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Alignment/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir, f"Alignment_{round(retention_rate * 100)}p_{iter_num:04d}.pdf"
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()


# Plot Alignment figure with truth
def compute_dangling_node_angles(df_bus, df_branch, n_squared=1):
    # Dangling nodes를 찾습니다.
    dangling_nodes = [
        node
        for node in df_bus["Node"]
        if (df_branch["Node1"] == node).sum() == 1
        or (df_branch["Node2"] == node).sum() == 1
    ]

    # Initialize the list to store the adjusted angles
    adjusted_angles = []

    # Calculate the adjusted angles for each dangling node
    for dangling_node in dangling_nodes:
        if (df_branch["Node1"] == dangling_node).sum() == 1:
            neighbor = df_branch.loc[
                df_branch["Node1"] == dangling_node, "Node2"
            ].values[0]
        else:
            neighbor = df_branch.loc[
                df_branch["Node2"] == dangling_node, "Node1"
            ].values[0]

        neighbor_neighbors = df_branch[
            (df_branch["Node1"] == neighbor) | (df_branch["Node2"] == neighbor)
        ]
        neighbor_neighbors = neighbor_neighbors[
            neighbor_neighbors["Node1"] != dangling_node
        ]
        neighbor_neighbors = neighbor_neighbors[
            neighbor_neighbors["Node2"] != dangling_node
        ]

        neighbor_positions = []
        for _, row in neighbor_neighbors.iterrows():
            if row["Node1"] == neighbor:
                neighbor_positions.append(
                    (
                        df_bus.loc[
                            df_bus["Node"] == row["Node2"], "Longitude_truth"
                        ].values[0],
                        df_bus.loc[
                            df_bus["Node"] == row["Node2"], "Latitude_truth"
                        ].values[0],
                    )
                )
            else:
                neighbor_positions.append(
                    (
                        df_bus.loc[
                            df_bus["Node"] == row["Node1"], "Longitude_truth"
                        ].values[0],
                        df_bus.loc[
                            df_bus["Node"] == row["Node1"], "Latitude_truth"
                        ].values[0],
                    )
                )

        if neighbor_positions:
            dir_vector = np.array([0.0, 0.0])
            for lon, lat in neighbor_positions:
                dir_vector_k = np.array([0.0, 0.0])
                dir_vector_k[0] = (
                    lon
                    - df_bus.loc[df_bus["Node"] == neighbor, "Longitude_truth"].values[
                        0
                    ]
                )
                dir_vector_k[1] = (
                    lat
                    - df_bus.loc[df_bus["Node"] == neighbor, "Latitude_truth"].values[0]
                )
                norm_k = np.linalg.norm(dir_vector_k)
                if norm_k != 0 and not np.isnan(norm_k):
                    dir_vector_k /= norm_k ** (n_squared + 1)
                dir_vector += dir_vector_k

            dangling_position = (
                df_bus.loc[df_bus["Node"] == dangling_node, "Longitude_truth"].values[
                    0
                ],
                df_bus.loc[df_bus["Node"] == dangling_node, "Latitude_truth"].values[0],
            )

            vec_neighbor_to_dangling = np.array(dangling_position) - np.array(
                (
                    df_bus.loc[df_bus["Node"] == neighbor, "Longitude_truth"].values[0],
                    df_bus.loc[df_bus["Node"] == neighbor, "Latitude_truth"].values[0],
                )
            )

            angle1 = np.arctan2(dir_vector[1], dir_vector[0])
            angle2 = np.arctan2(
                vec_neighbor_to_dangling[1], vec_neighbor_to_dangling[0]
            )

            adjusted_angle = np.degrees(angle1 - angle2) % 360
            adjusted_angles.append(adjusted_angle)

    return adjusted_angles


def plot_dangling_node_angles(path, adjusted_angles):
    # Set the number of bins
    num_bins = 18

    # Create the bins
    bins = np.linspace(-180, 180, num_bins + 1)

    # Compute the histogram
    adjusted_angles_copy = (np.array(adjusted_angles)) % 360 - 180
    adjusted_angles_copy += 180 / num_bins
    adjusted_angles_copy[adjusted_angles_copy >= 180] -= 360
    hist, _ = np.histogram(adjusted_angles_copy, bins=bins)

    # Compute the probability density function
    pdf = hist / np.sum(hist)

    # Compute the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_bins + 1, endpoint=True)

    # Create the figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plot the radar chart
    ax.plot(
        angles,
        np.concatenate((pdf, [pdf[0]]), 0),
        "o-",
        markersize=4,
        linewidth=1,
        color="#67BBA0",
    )

    # Fill the area inside the radar chart
    ax.fill(angles, np.concatenate((pdf, [pdf[0]]), 0), alpha=0.25, color="#67BBA0")

    # Set the labels for the radar chart
    angle_labels = [
        f"{(angle * 180 / np.pi if angle <= np.pi else angle * 180 / np.pi - 360):.0f}°"
        for angle in angles[:-1]
    ]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels=angle_labels, fontsize=13)

    # Ensure hist is not empty
    if np.sum(hist) == 0:
        pdf = np.zeros_like(hist)
    else:
        pdf = hist / np.sum(hist)

    # Check for NaN values in pdf and handle them
    if np.isnan(np.max(pdf)):
        max_prob = 0
    else:
        max_prob = np.max(pdf)

    max_prob_rounded = np.ceil(max_prob / 0.05) * 0.05  # Round up to the nearest 0.05
    prob_labels = [f"{i*5:.0f}%" for i in range(int(max_prob_rounded / 0.05) + 1)]

    # Set the radial labels for the radar chart
    ax.set_rgrids(
        np.linspace(0, max_prob_rounded, len(prob_labels)),
        labels=prob_labels,
        fontsize=13,
    )

    ax.set_rlabel_position(0)
    ax.set_rlim(0, max_prob * 1.1)

    # Make the outer circle thicker
    ax.spines["polar"].set_color("black")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Alignment/Angle/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figure_dir, f"Angle_of_dangling_node.pdf"), format="pdf", dpi=300
    )

    # Close the plot
    plt.close()
