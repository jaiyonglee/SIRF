# Load basic libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Link data regression
def regress_link_data(df_branch, regression_poly_degree, include_bias=True):
    df_branch_copy = df_branch.copy()
    df_branch_valid = df_branch_copy.dropna(subset=["Distance"])

    # Get input and target data
    branch_target = df_branch_valid["Distance"].to_numpy()
    branch_input = df_branch_valid.drop(
        [
            "Node1",
            "Node2",
            "Latitude_1",
            "Longitude_1",
            "Latitude_2",
            "Longitude_2",
            "Distance",
        ],
        axis=1,
    ).to_numpy()
    
    # Get input data for final prediction
    df_branch_input_final = df_branch_copy.drop(
        [
            "Node1",
            "Node2",
            "Latitude_1",
            "Longitude_1",
            "Latitude_2",
            "Longitude_2",
            "Distance",
        ],
        axis=1,
    ).to_numpy()

    # Create a second-order polynomial regression model for the data.
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=regression_poly_degree, include_bias=include_bias)),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )
    model = model.fit(branch_input, branch_target)
    coef = model.named_steps["linear"].coef_

    # Add the value model.predict(branch_input) to df_branch.
    df_branch_copy["Predicted_distance"] = model.predict(df_branch_input_final)

    # Change it to minimum positive value, if the value of df_branch_copy['DIST_PRED'] is less than zero.
    df_branch_copy.loc[df_branch_copy["Predicted_distance"] < 0, "Predicted_distance"] = 0 

    # Create a dataframe about coef
    df_coef = pd.DataFrame([coef], columns=model.named_steps["poly"].get_feature_names_out())

    return df_branch_copy, df_coef


# Get figure of the regression
def plot_regression_figure(path, retention_rate, iter_num):
    # Load data
    file_path = os.path.join(
        path,
        f"Result/Regressed_data/Regressed_branch/Branch_regressed_{round(retention_rate * 100)}p_{iter_num:04d}.csv",
    )
    df_branch_regressed = pd.read_csv(file_path)

    # Set plot settings
    sns.set_theme(rc={"figure.dpi": 300})
    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 5))
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "STIXGeneral",
            "font.size": 24,
        }
    )

    # Scatter plot settings
    plt.xscale("log")
    plt.yscale("log")
    point_size = 1
    plt.scatter(
        df_branch_regressed["Predicted_distance"],
        df_branch_regressed["Distance"],
        s=point_size,
        color="#597EBA",
    )
    plt.xlabel("Estimated distance $\hat{d}_{ij}$ (km)", size=16)
    plt.ylabel("Geodesic distance $d_{ij}$ (km)", size=16)
    plt.xlim(1e-3, 1e3)
    plt.ylim(1e-3, 1e3)

    # Draw trend line
    x = np.linspace(1e-3, 1e3, 100)
    plt.plot(x, x, linestyle="--", color="#182736", label="$f(x)=x$")

    # Pearson correlation coefficient: on right bottom over the figure with smaller box
    corr = df_branch_regressed["Predicted_distance"].corr(df_branch_regressed["Distance"])
    plt.text(
        0.98,
        0.02,
        f"$r={corr:.2f}$",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"),
    )

    # Save plot
    figure_dir = os.path.join(path, "Result/Figure/Regression/")
    os.makedirs(figure_dir, exist_ok=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figure_dir, f"Regression_{round(retention_rate * 100)}p_{iter_num:04d}.pdf"
        ),
        format="pdf",
        dpi=300,
    )

    # Close plot to release memory
    plt.close()