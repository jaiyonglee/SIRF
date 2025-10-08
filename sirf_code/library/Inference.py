# Load basic libraries
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
from math import sin, cos, sqrt, atan2, radians

# Load libraries from parent directory
from sirf_code.library.Data_Scaling import find_distance, copy_position_to_link_data


# Adam optimizer
def adam(value, grad, t, m=0, v=0, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad**2
    m_hat = m_new / (1 - beta1 ** (t + 1))
    v_hat = v_new / (1 - beta2 ** (t + 1))
    value_new = value - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return value_new, m_new, v_new


# Filter the graph by the n-th largest cluster
def infer_position(
    df_bus, df_branch, df_bus_nan, outputs=[], iter_n=100000, rate_change=True
):

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

    # Set the output lists and variables
    loss_tau = []
    distance_rates = []
    distance_rate = 1.0
    t_tau = 100

    # Get graph G_grid
    G_grid = nx.from_pandas_edgelist(
        df_branch_copy,
        source="Node1",
        target="Node2",
        edge_attr="Predicted_distance",
    )

    # Set node attributes
    nx.set_node_attributes(G_grid, bus_pos, "Position")

    # Infer the position of the nodes
    for i in tqdm(range(iter_n)):
        # Copy the position of the nodes
        bus_pos_copy = bus_pos.copy()

        # Initialize the variables
        cnt = 0
        grad_dist_rate = 0
        iter_loss = 0
        epsilon = 1e-8

        # Initialize the Adam optimizer
        m_lat_long = 0
        v_lat_long = 0
        m_lat_squared_sum = 0
        m_long_squared_sum = 0
        v_lat_squared_sum = 0
        v_long_squared_sum = 0
        # m_lat = 0
        # m_long = 0
        # v_lat = 0
        # v_long = 0
        m_dist_rate = 0
        v_dist_rate = 0

        # Update the position of the nodes
        for node in G_grid.nodes:

            # Update the position of the node if the node is unknown
            if node in bus_nan:
                cnt += 1
                grad_lat = 0
                grad_long = 0

                # Calculate the gradient of the latitude and longitude
                for neighbor in G_grid[node]:
                    # Calculate the distance between the node and the neighbor
                    distance_tau = find_distance(bus_pos[node], bus_pos[neighbor])
                    adjusted_distance_tau = distance_tau if distance_tau != 0 else epsilon
                    if distance_tau == 0:
                        print(node, neighbor, bus_pos[node], bus_pos[neighbor])

                    # Calculate the difference between the predicted distance and the actual distance
                    distance_pred = (
                        distance_rate * G_grid[node][neighbor]["Predicted_distance"]
                    )


                    ## Geodesic distance
                    # Constants
                    R = 6371  # Earth's radius in kilometers

                    # Convert positions to radians
                    lat1 = np.radians(bus_pos[node][0])
                    lon1 = np.radians(bus_pos[node][1])
                    lat2 = np.radians(bus_pos[neighbor][0])
                    lon2 = np.radians(bus_pos[neighbor][1])

                    # Differences
                    delta_lat = lat2 - lat1
                    delta_lon = lon2 - lon1

                    # Haversine formula components
                    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2

                    # Ensure 'a' is not zero or one to avoid division by zero
                    a = np.clip(a, epsilon, 1 - epsilon)

                    # Compute derivatives of 'a' with respect to lat1 and lon1
                    da_dlat1 = (-np.sin(delta_lat) / 2) + \
                        np.sin(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2

                    da_dlon1 = -np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon) / 2

                    # Compute derivative of distance_tau with respect to 'a'
                    d_distance_tau_da = R / (np.sqrt(1 - a) * np.sqrt(a))

                    # Compute gradients
                    grad_lat -= -2 * (distance_tau - distance_pred) * d_distance_tau_da * da_dlat1
                    grad_long -= -2 * (distance_tau - distance_pred) * d_distance_tau_da * da_dlon1


                    ## Euclidean distance
                    # # Calculate the gradient of the latitude and longitude
                    # grad_lat += (
                    #     2
                    #     * (bus_pos[node][0] - bus_pos[neighbor][0])
                    #     * (distance_tau - distance_pred)
                    #     / adjusted_distance_tau
                    # )

                    # grad_long += (
                    #     2
                    #     * (bus_pos[node][1] - bus_pos[neighbor][1])
                    #     * (distance_tau - distance_pred)
                    #     / adjusted_distance_tau
                    # )


                    # Calculate the gradient of the distance rate
                    grad_dist_rate -= (
                        2
                        * G_grid[node][neighbor]["Predicted_distance"]
                        * (distance_tau - distance_pred)
                    )

                    # Calculate the loss
                    iter_loss += (distance_tau - distance_pred) ** 2

                # Get the updated position of the node
                pos_lat, m_lat_node, v_lat_node = adam(
                    bus_pos[node][0], grad_lat, i, m=m_lat_long, v=v_lat_long
                )
                pos_long, m_long_node, v_long_node = adam(
                    bus_pos[node][1], grad_long, i, m=m_lat_long, v=v_lat_long
                )

                # Update the position of the node
                bus_pos_copy[node] = [pos_lat, pos_long]

                # Update the squared sums
                m_lat_squared_sum += m_lat_node ** 2
                m_long_squared_sum += m_long_node ** 2
                v_lat_squared_sum += v_lat_node ** 2
                v_long_squared_sum += v_long_node ** 2

                # # Update the parameters
                # m_lat += abs(m_lat_node)
                # m_long += abs(m_long_node)
                # v_lat += abs(v_lat_node)
                # v_long += abs(v_long_node)

            # Pass if all nodes are known
            else:
                pass

        # Update the Adam parameters
        m_lat_long = np.sqrt((m_lat_squared_sum + m_long_squared_sum) / (2 * cnt))
        v_lat_long = np.sqrt((v_lat_squared_sum + v_long_squared_sum) / (2 * cnt))

        # # Update the Adam parameters
        # m_lat_long = np.linalg.norm([m_lat, m_long]) / (np.sqrt(2) * cnt)
        # v_lat_long = np.linalg.norm([v_lat, v_long]) / (np.sqrt(2) * cnt)

        # Update distance_rate
        if rate_change:
            distance_rate, m_dist_rate, v_dist_rate = adam(
                distance_rate, grad_dist_rate, i, m=m_dist_rate, v=v_dist_rate
            )

        # Update the position of the nodes
        bus_pos = bus_pos_copy.copy()

        # Store the loss
        loss_tau.append(np.sqrt(iter_loss) / cnt)
        distance_rates.append(distance_rate)

        # Break if the loss is not reduced
        if len(loss_tau) > 2 * t_tau:
            if np.sum(loss_tau[-t_tau:]) > np.sum(loss_tau[-2 * t_tau : -t_tau]):
                break

        # Max iteration reached
        if i == iter_n - 1:
            print("Max iteration reached")

    # Set the output values
    outputs = {}
    outputs["loss"] = loss_tau
    outputs["distance_rate"] = distance_rates

    # Update df_bus
    df_bus_copy["Latitude"] = df_bus_copy["Node"].map(lambda x: bus_pos[x][0])
    df_bus_copy["Longitude"] = df_bus_copy["Node"].map(lambda x: bus_pos[x][1])

    # Update df_branch
    df_branch_copy = copy_position_to_link_data(df_bus_copy, df_branch_copy)

    # Print the final loss
    print(f"Number of Iteration: {len(loss_tau)}")

    return df_bus_copy, df_branch_copy, outputs


# Filter data
def filter_data(df_bus_inferred, df_branch_inferred, df_bus_nan, outputs):
    # Extract data that meets filtering conditions
    df_branch_inferred_copy = df_branch_inferred.copy()
    df_branch_inferred_copy["Loss_branch"] = np.abs(
        df_branch_inferred_copy["Distance"]
        - df_branch_inferred["Predicted_distance"] * outputs["distance_rate"][-1]
    )

    # Compute loss for each bus
    df_bus_inferred_copy = df_bus_inferred.copy()
    df_bus_inferred_copy = df_bus_inferred_copy[~df_bus_inferred_copy["Node"].isin(df_bus_nan["Node"])]
    df_bus_inferred_copy["Loss_bus"] = 0.0
    for i in range(len(df_branch_inferred_copy)):
        df_bus_inferred_copy.loc[
            df_bus_inferred_copy["Node"] == df_branch_inferred_copy["Node1"].iloc[i],
            "Loss_bus",
        ] += df_branch_inferred_copy["Loss_branch"].iloc[i]
        df_bus_inferred_copy.loc[
            df_bus_inferred_copy["Node"] == df_branch_inferred_copy["Node2"].iloc[i],
            "Loss_bus",
        ] += df_branch_inferred_copy["Loss_branch"].iloc[i]
    df_bus_inferred_copy["Loss_bus"] /= 2

    # Filter the node with the largest loss
    df_bus_nan_copy = df_bus_nan.copy()
    node_to_add = df_bus_inferred_copy.nlargest(1, "Loss_bus")["Node"].values[0]
    new_row = pd.DataFrame({'Node': [node_to_add]})
    df_bus_nan_copy = pd.concat([df_bus_nan_copy, new_row], ignore_index=True)

    print(f"Node to add: {node_to_add}")

    return df_bus_nan_copy, node_to_add


# Plot Inference figure
def plot_inference_figure(
    path, retention_rate, iter_num, node_size, nation, boundary, truth_enabled, figsize=(40, 40)
):
    """
    Plot inference figure.

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
    COLORS = ["FACD47", "F5F3ED"]

    # Load data
    bus_init_path = os.path.join(
        path,
        f"Result/Inferred_data/Bus_inferred/Bus_inferred_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    branch_init_path = os.path.join(
        path,
        f"Result/Inferred_data/Branch_inferred/Branch_inferred_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    bus_nan_path = os.path.join(
        path,
        f"Result/Inferred_data/Nan_position/Bus_nan_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    parent_dir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
    map_path = os.path.join(parent_dir, "Maps/", nation + ".geojson")

    df_bus_inferred = pd.read_csv(bus_init_path)
    df_branch_inferred = pd.read_csv(branch_init_path)
    df_bus_nan = pd.read_csv(bus_nan_path)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(df_branch_inferred["Node1"])
    G.add_nodes_from(df_branch_inferred["Node2"])
    G.add_edges_from(df_branch_inferred[["Node1", "Node2"]].values)

    # Create position dictionary
    pos = {
        node: (
            df_bus_inferred.loc[df_bus_inferred["Node"] == node, "Longitude"].iloc[0],
            df_bus_inferred.loc[df_bus_inferred["Node"] == node, "Latitude"].iloc[0],
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
            label="Optimized",
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
    nx.draw_networkx_edges(G, pos, edge_color="black", alpha=0.8, width=3)  # inferring

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
    figure_dir = os.path.join(path, "Result/Figure/Inference/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir, f"Inference_{round(retention_rate * 100)}p_{iter_num:04d}.pdf"
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()


# Plot Inference loss
def plot_inference_loss(path, retention_rate, iter_num):
    """
    Plot inference loss.

    Parameters:
        path (str): The path to data files.
        retention_rate (float): Retention rate.
        iter_num (int): Iteration number.
        outputs (dict): Inference outputs.
    """
    # Load data
    loss_init_path = os.path.join(
        path,
        f"Result/Inferred_data/Inference_loss/Inference_loss_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    loss = np.genfromtxt(loss_init_path, delimiter=",")

    # Plot inference loss
    fig, ax = plt.subplots()
    ax.plot(loss, label="Loss [km]", color="#597EBA")

    # Label
    ax.set_xlabel("Invalidated anchor nodes (rounds)")
    ax.set_ylabel("Average target node loss [km]")
    # ax.set_title("Average target node loss over invalidation rounds")

    # Ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Inference/Loss/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir,
            f"Inference_Loss_{round(retention_rate * 100)}p_{iter_num:04d}.pdf",
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()


# Plot Inference distance rate
def plot_distance_rate(path, retention_rate, iter_num):
    """
    Plot inference distance rate.

    Parameters:
        path (str): The path to data files.
        retention_rate (float): Retention rate.
        iter_num (int): Iteration number.
        outputs (dict): Inference outputs.
    """
    # Load data
    distance_rate_init_path = os.path.join(
        path,
        f"Result/Inferred_data/Distance_rate/Distance_rate_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    distance_rate = np.genfromtxt(distance_rate_init_path, delimiter=",")

    # Plot inference distance rate
    fig, ax = plt.subplots()
    ax.plot(distance_rate, label="Distance Rate", color="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance Rate")
    ax.set_title("Inference Distance Rate")
    ax.legend()

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Inference/Distance_rate/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir,
            f"Inference_Distance_rate_{round(retention_rate * 100)}p_{iter_num:04d}.pdf",
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()


# Plot L_km vs Distance
def plot_L_km_vs_Distance(path, retention_rate, iter_num):
    """
    Plot L_km vs Distance.

    Parameters:
        path (str): The path to data files.
        retention_rate (float): Retention rate.
        iter_num (int): Iteration number.
        outputs (dict): Inference outputs.
    """
    # Load data
    branch_inferred_init_path = os.path.join(
        path,
        f"Result/Inferred_data/Branch_inferred/Branch_inferred_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    branch_inferred = pd.read_csv(branch_inferred_init_path)
    l_km = branch_inferred["L_km"]
    distance = branch_inferred["Distance"]
    predicted_distance = branch_inferred["Predicted_distance"]

    # Plot L_km vs Distance
    fig, ax = plt.subplots()
    ax.scatter(l_km, distance, label="Distance", color="darkgreen", s=3)
    ax.scatter(
        l_km, predicted_distance, label="Predicted Distance", color="darkred", s=3
    )
    ax.loglog()
    ax.set_xlabel("L_km")
    ax.set_ylabel("Distance")
    ax.set_title("L_km vs Distance")
    ax.legend()

    # y=x line
    x = np.linspace(min(l_km), max(l_km), 100)
    ax.plot(x, x, color="black", linestyle="--")

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Inference/L_km_vs_Distance/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir,
            f"L_km_vs_Distance_{round(retention_rate * 100)}p_{iter_num:04d}.pdf",
        ),
        format="pdf",
        dpi=300,
    )

    # Close the plot
    plt.close()
