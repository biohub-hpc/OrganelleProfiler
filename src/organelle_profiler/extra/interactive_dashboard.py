import argparse
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import webbrowser
from threading import Timer


### TO RUN:     python -m organelle_profiler.extra.interactive_dashboard analysis/graphs/interactive_umaps/interactive_data_original.parquet


def create_dashboard(df: pd.DataFrame):
    """
    Initializes and runs a Dash dashboard for exploring UMAP data.
    """
    app = Dash(__name__)

    # --- FIX: Convert string cluster IDs to numeric for Plotly ---
    if "cluster" in df.columns and pd.api.types.is_string_dtype(df["cluster"]):
        print("  - Converting string cluster IDs to numeric for plotting...")
        df["cluster"] = df["cluster"].str.replace("c", "").astype(int)

    # --- Identify columns for different purposes ---
    umap_cols = ["umap_1", "umap_2"]
    # Define columns that are good candidates for the color dropdown
    colorable_cols = [
        col
        for col in df.columns
        if col not in ["cell_id", "umap_1", "umap_2"]
        and (df[col].dtype != "object" or df[col].nunique() < 100)
    ]
    # Ensure 'cluster' is the first option if it exists
    if "cluster" in colorable_cols:
        colorable_cols.insert(0, colorable_cols.pop(colorable_cols.index("cluster")))

    # Metadata for the hover tooltip (user-customized)
    metadata_cols = [
        col
        for col in ["cell_id", "gene_name", "sgRNA", "barcode", "gene_effect", "well"]
        if col in df.columns
    ]
    # All other columns are treated as features for the hover
    feature_cols = [
        col for col in df.columns if col not in umap_cols + metadata_cols + ["cluster"]
    ]

    # --- Define the layout of the web application ---
    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif"},
        children=[
            html.H1("Interactive UMAP Explorer", style={"textAlign": "center"}),
            html.Div(
                className="control-panel",
                children=[
                    html.Label("Color by:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[
                            {"label": col, "value": col} for col in colorable_cols
                        ],
                        value=default_color_col,  # Use the safe default
                        clearable=False,
                        style={"width": "300px", "display": "inline-block"},
                    ),
                ],
                style={"padding": "20px", "textAlign": "center"},
            ),
            dcc.Graph(id="umap-graph", style={"height": "80vh"}),
        ],
    )

    # --- Define the callback to update the graph ---
    @app.callback(Output("umap-graph", "figure"), Input("color-dropdown", "value"))
    def update_graph(color_by_col):
        if color_by_col is None:
            # If no column is selected, return an empty graph with a message
            fig = go.Figure()
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[
                    {
                        "text": "No data to display. Please select a property to color by.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 16},
                    }
                ],
            )
            return fig

        print(f"Updating plot to color by: {color_by_col}")

        marker_properties = {
            "color": df[color_by_col],
            "showscale": True,
            "colorbar": {"title": color_by_col.replace("_", " ").title()},
            "line_width": 0,
            "size": 5,
            "opacity": 0.7,
        }

        # Use a continuous color scale for numeric data
        if pd.api.types.is_numeric_dtype(df[color_by_col]):
            marker_properties["colorscale"] = "viridis"

        fig = go.Figure(
            data=go.Scattergl(
                x=df["umap_1"],
                y=df["umap_2"],
                mode="markers",
                marker=marker_properties,
                hovertemplate=(
                    "<b>Cell Info</b><br>"
                    + "<br>".join(
                        [
                            f"{col}: %{{customdata[{i}]}}"
                            for i, col in enumerate(metadata_cols)
                        ]
                    )
                    + "<br>----------<br>"
                    + "<b>Top Features</b><br>"
                    + "<br>".join(
                        [
                            f"{col}: %{{customdata[{len(metadata_cols) + i}]:.4f}}"
                            for i, col in enumerate(feature_cols)
                        ]
                    )
                    + "<extra></extra>"
                ),
                customdata=df[metadata_cols + feature_cols],
            )
        )

        fig.update_layout(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            margin=dict(l=40, r=40, t=40, b=40),
            transition_duration=500,  # Smooth transition when changing colors
        )
        return fig

    return app


def main():
    """Main execution function to launch the dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch an interactive UMAP dashboard."
    )
    parser.add_argument(
        "data_path", help="Path to the .parquet file containing the UMAP data."
    )
    args = parser.parse_args()

    try:
        print(f"--- Loading data from {args.data_path} ---")
        df = pd.read_parquet(args.data_path)
        print("--- Data loaded successfully. ---")
    except Exception as e:
        print(f"Error: Could not load data file. {e}")
        return

    # Check for required columns
    required_cols = ["umap_1", "umap_2", "cluster"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Data file must contain the following columns: {required_cols}")
        return

    app = create_dashboard(df)

    port = 8050
    url = f"http://127.0.0.1:{port}"

    # Open the web browser automatically after a short delay
    Timer(1, lambda: webbrowser.open_new(url)).start()

    print(f"--- Launching Dashboard on {url} ---")
    print("--- Use Ctrl+C in this terminal to shut down the server. ---")

    app.run(debug=False, port=port)


if __name__ == "__main__":
    main()
