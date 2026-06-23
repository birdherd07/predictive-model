import os
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree


GRID_SIZE = 1000


def load_predictions(prediction_file):
    preds = pd.read_csv(
        prediction_file,
        header=None,
        names=["id", "r_pct", "d_pct"]
    )

    preds["id"] = preds["id"].astype(str)

    return preds


def create_geographic_map(df, value_column):
    lon = df["lon"].values
    lat = df["lat"].values
    values = df[value_column].values

    grid_lon, grid_lat = np.mgrid[
        lon.min():lon.max():complex(GRID_SIZE),
        lat.min():lat.max():complex(GRID_SIZE)
    ]

    # Smooth interpolation between counties
    grid = griddata(
        (lon, lat),
        values,
        (grid_lon, grid_lat),
        method="linear"
    )

    # Build nearest-neighbor tree from actual county locations
    county_points = np.column_stack([
        lon,
        lat
    ])

    tree = KDTree(county_points)

    # Estimate average county spacing
    distances, _ = tree.query(
        county_points,
        k=2
    )

    county_spacing = np.median(
        distances[:, 1]
    )

    # Distance from every rendered pixel
    query_points = np.column_stack([
        grid_lon.ravel(),
        grid_lat.ravel()
    ])

    pixel_distance, _ = tree.query(
        query_points,
        k=1
    )

    pixel_distance = pixel_distance.reshape(
        grid.shape
    )

    # Remove areas too far from any county
    mask_distance = county_spacing * 4

    grid[
        pixel_distance > mask_distance
    ] = np.nan

    return grid


def save_map(
    grid,
    title,
    output_file,
    cmap_name,
    vmin=None,
    vmax=None,
    colorbar_label=""
):
    plt.figure(figsize=(10, 8), facecolor='black')

    cmap = plt.get_cmap(cmap_name).copy()

    # transparent outside state
    cmap.set_bad(alpha=0)

    img = plt.imshow(
        grid.T,
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax
    )

    plt.title(title, color='white')

    cbar = plt.colorbar(img)
    cbar.set_label(colorbar_label, color='white')
    cbar.ax.tick_params(colors='white')

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        transparent=False
    )

    plt.close()

    print(f"Saved {output_file}")


def process_state(
    state_file,
    predictions,
    output_dir
):
    state_name = os.path.splitext(
        os.path.basename(state_file)
    )[0]

    print(f"\nProcessing {state_name}")

    counties = pd.read_csv(
        state_file,
        header=None,
        usecols=[0, 1, 2, 3],
        names=[
            "id",
            "lon",
            "lat",
            "population"
        ]
    )

    counties["id"] = counties["id"].astype(str)

    merged = counties.merge(
        predictions,
        on="id",
        how="inner"
    )

    if len(merged) == 0:
        print(
            f"No matching predictions found for {state_name}"
        )
        return

    print(
        f"Matched {len(merged)} predictions"
    )

    merged["margin"] = (
        merged["r_pct"]
        - merged["d_pct"]
    )

    # Margin map
    margin_grid = create_geographic_map(
        merged,
        "margin"
    )

    vmax = np.nanmax(
        np.abs(margin_grid)
    )

    save_map(
        margin_grid,
        f"{state_name} Margin",
        os.path.join(
            output_dir,
            f"{state_name}_margin.png"
        ),
        cmap_name="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        colorbar_label="R% - D%"
    )

    # Republican map
    republican_grid = create_geographic_map(
        merged,
        "r_pct"
    )

    save_map(
        republican_grid,
        f"{state_name} Republican %",
        os.path.join(
            output_dir,
            f"{state_name}_republican.png"
        ),
        cmap_name="Reds",
        vmin=0,
        vmax=100,
        colorbar_label="Republican %"
    )

    # Democrat map
    democrat_grid = create_geographic_map(
        merged,
        "d_pct"
    )

    save_map(
        democrat_grid,
        f"{state_name} Democrat %",
        os.path.join(
            output_dir,
            f"{state_name}_democrat.png"
        ),
        cmap_name="Blues",
        vmin=0,
        vmax=100,
        colorbar_label="Democrat %"
    )


def main():
    print(
        "\n=== Geographic Prediction Visualizer ===\n"
    )

    data_folder = input(
        "Folder containing original state CSV training files:\n"
    ).strip()

    prediction_file = input(
        "\nPrediction CSV:\n"
    ).strip()

    output_dir = "output_maps"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    predictions = load_predictions(
        prediction_file
    )

    state_files = glob.glob(
        os.path.join(
            data_folder,
            "*.csv"
        )
    )

    if not state_files:
        print("No CSV files found.")
        return

    print(
        f"\nFound {len(state_files)} state files."
    )

    for state_file in state_files:
        process_state(
            state_file,
            predictions,
            output_dir
        )

    print(
        f"\nFinished.\nMaps saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
