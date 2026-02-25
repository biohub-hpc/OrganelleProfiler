import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import textwrap
from scipy.stats import ttest_ind


def load_gene_summary(
    experiment: str | None, gene_summary_path: Path | None
) -> tuple[pd.DataFrame, Path]:
    if gene_summary_path is not None:
        csv_path = Path(gene_summary_path)
    elif experiment is not None:
        # Lazy import to avoid hard dependency if user supplies a CSV directly
        from ops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(experiment)
        csv_path = dataset.analysis_path / "gene_summary_features.csv"
    else:
        raise ValueError("Provide either --experiment or --gene_summary.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Gene summary CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # print all column names
    print(df.columns)

    # save all column names to a file
    with open("column_names.txt", "w") as f:
        for col in df.columns:
            f.write(col + "\n")

    return df, csv_path


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # Fit y = m x + b via least squares
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


# def make_plot_scatter(
#     x: np.ndarray,
#     y: np.ndarray,
#     slope: float,
#     intercept: float,
#     r2: float,
#     x_label: str,
#     y_label: str,
#     title: str | None,
#     out_path: Path,
#     outliers: list[tuple[float, float, str]] | None = None,
# ):
#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.5)

#     x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
#     y_line = slope * x_line + intercept
#     ax.plot(x_line, y_line, color="crimson", lw=2, label="Linear fit")

#     eq_text = f"y = {slope:.4g} x + {intercept:.4g}\nR^2 = {r2:.4f}"
#     ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, va="top", ha="left",
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     if title:
#         ax.set_title(title)
#     ax.grid(True, alpha=0.25)

#     # Annotate outliers if provided
#     if outliers:
#         for xo, yo, label in outliers:
#             ax.annotate(
#                 label,
#                 (xo, yo),
#                 textcoords="offset points",
#                 xytext=(6, 6),
#                 ha="left",
#                 fontsize=8,
#                 color="black",
#                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7),
#             )

#     fig.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)


def make_plot_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    r2: float,
    x_label: str,
    y_label: str,
    title: str | None,
    out_path: Path,
    gridsize: int = 40,
    cmap: str = "viridis",
    outliers: list[tuple[float, float, str]] | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Gene KO density")

    # Regression line
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="crimson", lw=2, label="Linear fit")

    eq_text = f"y = {slope:.4g} x + {intercept:.4g}\nR^2 = {r2:.4f}"
    ax.text(
        0.05,
        0.95,
        eq_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.15)

    # Annotate outliers if provided
    if outliers:
        for xo, yo, label in outliers:
            ax.annotate(
                label,
                (xo, yo),
                textcoords="offset points",
                xytext=(6, 6),
                ha="left",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7),
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate scatter and heatmap plots correlating gene-median areas: "
            "(1) nuclei vs membrane/cell, (2) mitochondria vs cell, (3) mitochondria vs nuclei. "
            "Fits linear regressions and saves equations and R^2."
        )
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name (used to locate gene_summary_features.csv via OpsDataset)",
    )

    args = parser.parse_args()

    # Load the gene summary CSV next to the experiment's analysis path
    df, csv_path = load_gene_summary(args.experiment, None)
    out_dir = Path(csv_path).parent / "size_correlations"

    # make out_dir if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load guideRNA-level summary for violin analyses
    def load_guide_summary(experiment: str) -> tuple[pd.DataFrame, Path]:
        from ops_utils.data.experiment import OpsDataset

        dataset = OpsDataset(experiment)
        path = dataset.analysis_path / "guideRNA_summary_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"GuideRNA summary CSV not found at {path}")
        gdf_local = pd.read_csv(path)
        return gdf_local, path

    gdf, gcsv_path = load_guide_summary(args.experiment)

    # Smart column selection
    cols_lower = {c.lower(): c for c in df.columns}

    def pick_first_present(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    # Define candidate preference orders (all in lowercase)
    nuclei_candidates = [
        "nuclei_area_sum_mean",
        # "nuclei_area_sum_median",
    ]
    membrane_candidates = [
        # "cell_area_median",
        "cell_area_mean",
    ]
    mito_candidates = [
        "mitochondria_area_sum_mean",
        "mitochondria_area_sum_median",
        "mito_area_median",
    ]

    nuclei_col = pick_first_present(nuclei_candidates)
    membrane_or_cell_col = pick_first_present(membrane_candidates)
    mito_col = pick_first_present(mito_candidates)

    if nuclei_col is None:
        raise KeyError("Could not find a nuclei area column in gene summary.")
    if membrane_or_cell_col is None:
        raise KeyError(
            "Could not find cell membrane or cell area column in gene summary."
        )
    if mito_col is None:
        print(
            "Warning: Could not find mitochondria area column; mito plots will be skipped."
        )

    # Print group label suggestions for NTC and RPL if matches are unclear
    def _print_group_suggestions(
        df_in: pd.DataFrame, suggestions_out: Path | None = None
    ):
        names_series = df_in["gene_name"].astype(str)
        names_unique = names_series.dropna().unique().tolist()
        names_lower = [n.lower() for n in names_unique]

        ntc_patterns = ["ntc", "0", "control", "ctrl"]
        rpl_regex = re.compile(r"^rpl", re.IGNORECASE)
        rpl_alt_contains = "ribosomal protein l"

        ntc_matches = sorted(
            {n for n in names_unique if any(p in n.lower() for p in ntc_patterns)}
        )
        rpl_matches = sorted(
            {
                n
                for n in names_unique
                if rpl_regex.search(n)
                or (rpl_alt_contains in n.lower())
                or ("rpl" in n.lower())
            }
        )

        lines = []
        print("Group suggestions (from gene_name):")
        if ntc_matches:
            print(f"  NTC-like: {', '.join(ntc_matches[:50])}")
            lines.append("NTC-like: " + ", ".join(ntc_matches))
        else:
            print("  NTC-like: <no matches>")
            lines.append("NTC-like: <no matches>")

        if rpl_matches:
            print(f"  RPL-like: {', '.join(rpl_matches[:50])}")
            lines.append("RPL-like: " + ", ".join(rpl_matches))
        else:
            print("  RPL-like: <no matches>")
            lines.append("RPL-like: <no matches>")

        if suggestions_out is not None:
            try:
                suggestions_out.parent.mkdir(parents=True, exist_ok=True)
                suggestions_out.write_text("\n".join(lines))
            except Exception:
                pass

    _print_group_suggestions(gdf, (out_dir / "group_label_suggestions.txt"))

    def run_one_pair(x_col: str, y_col: str, title: str | None = None):
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skipping: missing columns {x_col} or {y_col} in {csv_path}")
            return

        pair_df = df[["gene_name", x_col, y_col]].replace([np.inf, -np.inf], np.nan)
        pair_df = pair_df.dropna(subset=[x_col, y_col])
        x_arr = pair_df[x_col].to_numpy(dtype=float)
        y_arr = pair_df[y_col].to_numpy(dtype=float)

        if x_arr.size == 0 or y_arr.size == 0:
            print(f"Skipping: no valid data for {x_col} vs {y_col}")
            return

        slope, intercept, r2 = fit_linear_regression(x_arr, y_arr)

        # Compute residuals and select top-10 absolute residual outliers
        y_pred = slope * x_arr + intercept
        residuals = np.abs(y_arr - y_pred)
        top_k = min(10, residuals.size)
        if top_k > 0:
            idx_sorted = np.argsort(-residuals)[:top_k]
            outlier_points = [
                (x_arr[i], y_arr[i], str(pair_df.iloc[i]["gene_name"]))
                for i in idx_sorted
            ]
        else:
            outlier_points = []

        # Also annotate extreme positive/negative (top-right and bottom-left) by x+y
        if x_arr.size > 0:
            sum_xy = x_arr + y_arr
            k = min(3, sum_xy.size)
            idx_top3 = np.argsort(sum_xy)[-k:]
            idx_bot3 = np.argsort(sum_xy)[:k]
            extreme_points = [
                (x_arr[i], y_arr[i], str(pair_df.iloc[i]["gene_name"]))
                for i in np.concatenate([idx_top3, idx_bot3])
            ]
        else:
            extreme_points = []

        # Merge and deduplicate labels, also add specific genes (case-insensitive)
        annotations = []
        seen = set()

        def _add_list(lst):
            for xo, yo, lab in lst:
                key = lab.upper()
                if key in seen:
                    continue
                seen.add(key)
                annotations.append((xo, yo, lab))

        _add_list(outlier_points)
        _add_list(extreme_points)

        # Specific genes to force-label (include TOMM20 synonym for TOM20)
        special_genes = {"COPB2", "KIF23", "TOM20", "TOMM20", "MTOR"}
        gene_to_idx = {str(g).upper(): i for i, g in enumerate(pair_df["gene_name"])}
        for g in special_genes:
            key = g.upper()
            if key in gene_to_idx:
                i = gene_to_idx[key]
                annotations.append(
                    (x_arr[i], y_arr[i], str(pair_df.iloc[i]["gene_name"]))
                )

        base = f"corr_{x_col}_vs_{y_col}".replace("/", "-").replace(" ", "_")

        # Heatmap (always)
        out_png_hm = out_dir / f"{base}_heatmap.png"
        make_plot_heatmap(
            x_arr,
            y_arr,
            slope,
            intercept,
            r2,
            x_label=x_col,
            y_label=y_col,
            title=title,
            out_path=out_png_hm,
            outliers=annotations,
        )
        print(f"Saved heatmap to {out_png_hm}")

        # Save outliers CSV
        outliers_csv = out_dir / f"{base}_top10_outliers.csv"
        if outlier_points:
            out_df = pair_df.copy()
            out_df = out_df.iloc[idx_sorted][["gene_name", x_col, y_col]].copy()
            out_df["abs_residual"] = residuals[idx_sorted]
            out_df.to_csv(outliers_csv, index=False)
            print(f"Saved top-10 outliers to {outliers_csv}")

    # =========================
    # Violin plots: NTC vs RPL
    # =========================
    # Column picking against guideRNA summary
    gcols_lower = {c.lower(): c for c in gdf.columns}

    def g_pick_first_present(candidates: list[str]) -> str | None:
        for c in candidates:
            low = c.lower()
            if low in gcols_lower:
                return gcols_lower[low]
        return None

    # Choose both mean and median columns (guide-level)
    cell_mean_col = g_pick_first_present(["cell_area_mean"])
    cell_median_col = g_pick_first_present(["cell_area_median"])
    nuclei_mean_col = g_pick_first_present(
        ["nuclei_area_sum_mean", "nuclei_area_mean_mean", "nuclei_area_mean"]
    )
    nuclei_median_col = g_pick_first_present(
        ["nuclei_area_sum_median", "nuclei_area_median"]
    )
    mito_mean_col = g_pick_first_present(
        ["mitochondria_area_sum_mean", "mitochondria_area_mean", "mito_area_mean"]
    )
    mito_median_col = g_pick_first_present(
        ["mitochondria_area_sum_median", "mitochondria_area_median", "mito_area_median"]
    )

    def _select_groups(
        df_in: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        names = df_in["gene_name"].astype(str).str.strip()
        is_ntc = names.eq("0")
        is_rpl = names.str.match(r"^RPL", case=False)
        is_mrpl = names.str.match(r"^MRPL", case=False)
        return df_in[is_ntc], df_in[is_rpl], df_in[is_mrpl]

    def _t_test_and_plot_pair(
        metric_col: str,
        pretty_name: str,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        label_a: str,
        label_b: str,
        out_suffix: str,
    ):
        if metric_col is None or metric_col not in gdf.columns:
            print(
                f"Skipping violin for {pretty_name} ({label_a} vs {label_b}): column not found"
            )
            return
        if df_a.empty or df_b.empty:
            print(
                f"Skipping violin for {pretty_name} ({label_a} vs {label_b}): missing rows"
            )
            return
        x_a = (
            df_a[metric_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        x_b = (
            df_b[metric_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if x_a.size == 0 or x_b.size == 0:
            print(
                f"Skipping violin for {pretty_name} ({label_a} vs {label_b}): no valid values"
            )
            return

        t_stat, p_val = ttest_ind(x_a, x_b, equal_var=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        parts = ax.violinplot([x_a, x_b], showmeans=True, showextrema=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#87CEEB")
            pc.set_edgecolor("black")
            pc.set_alpha(0.4)

        rng = np.random.default_rng(0)
        jitter_a = 1 + (rng.random(x_a.size) - 0.5) * 0.15
        jitter_b = 2 + (rng.random(x_b.size) - 0.5) * 0.15
        ax.scatter(jitter_a, x_a, s=12, color="#1f77b4", alpha=0.6, linewidths=0)
        ax.scatter(jitter_b, x_b, s=12, color="#d62728", alpha=0.6, linewidths=0)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([label_a, label_b])
        ax.set_ylabel(pretty_name)
        ax.set_title(
            f"{pretty_name}: {label_a} vs {label_b} (t={t_stat:.2f}, p={p_val:.2e})"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out_png = out_dir / f"violin_{out_suffix}_{metric_col}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        stats_txt = out_dir / f"violin_{out_suffix}_{metric_col}_ttest.txt"
        stats_txt.write_text(
            (
                f"Metric: {metric_col}\n"
                f"{label_a} n={x_a.size}, mean={np.mean(x_a):.4g}, std={np.std(x_a, ddof=1):.4g}\n"
                f"{label_b} n={x_b.size}, mean={np.mean(x_b):.4g}, std={np.std(x_b, ddof=1):.4g}\n"
                f"Welch t-test: t={t_stat:.6g}, p={p_val:.6g}\n"
            )
        )
        print(
            f"Saved violin and t-test for {pretty_name} ({label_a} vs {label_b}) to {out_png}"
        )

    df_ntc, df_rpl, df_mrpl = _select_groups(gdf)
    # Additional groups: 'PSM' (contains) and 'SRP' (startswith), case-insensitive
    names_all = gdf["gene_name"].astype(str).str.strip()
    names_all_lower = names_all.str.lower()
    is_psm = names_all_lower.str.contains("psm")
    is_srp = names_all_lower.str.startswith("srp")
    is_snr = names_all_lower.str.startswith("snr")
    df_psm = gdf[is_psm]
    df_srp = gdf[is_srp]
    df_snr = gdf[is_snr]
    # Per user request, skip saving individual per-metric violins; only generate side-by-side grids below

    # Side-by-side grid with all available metrics
    def _t_test_and_plot_grid_pair(
        metric_list: list[tuple[str, str]],
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        label_a: str,
        label_b: str,
        out_suffix: str,
    ):
        metrics_present = [
            (col, name) for col, name in metric_list if col and col in gdf.columns
        ]
        if not metrics_present:
            print(
                f"No metrics available for grid violin plot ({label_a} vs {label_b})."
            )
            return
        if df_a.empty or df_b.empty:
            print(f"Skipping grid violin ({label_a} vs {label_b}): missing rows")
            return

        n = len(metrics_present)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        axes = axes[0]

        stats_lines = []

        # Compute guide and gene counts per group for x-axis annotations
        def _counts(df_grp: pd.DataFrame) -> tuple[int, int]:
            guides = (
                int(df_grp["barcode"].nunique())
                if "barcode" in df_grp.columns
                else int(len(df_grp))
            )
            genes = (
                int(df_grp["gene_name"].nunique())
                if "gene_name" in df_grp.columns
                else 0
            )
            return guides, genes

        guides_a, genes_a = _counts(df_a)
        guides_b, genes_b = _counts(df_b)

        for ax, (metric_col, pretty_name) in zip(axes, metrics_present):
            x_a = (
                df_a[metric_col]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            x_b = (
                df_b[metric_col]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            if x_a.size == 0 or x_b.size == 0:
                ax.set_visible(False)
                continue

            # Welch's t-test and Cohen's d
            t_stat, p_val = ttest_ind(x_a, x_b, equal_var=False)
            mean_a = float(np.mean(x_a))
            mean_b = float(np.mean(x_b))
            std_a = float(np.std(x_a, ddof=1)) if x_a.size > 1 else np.nan
            std_b = float(np.std(x_b, ddof=1)) if x_b.size > 1 else np.nan
            pooled_num = (
                (x_a.size - 1) * (std_a**2) + (x_b.size - 1) * (std_b**2)
                if x_a.size > 1 and x_b.size > 1
                else np.nan
            )
            pooled_den = (
                (x_a.size + x_b.size - 2) if x_a.size > 1 and x_b.size > 1 else np.nan
            )
            pooled_std = (
                float(np.sqrt(pooled_num / pooled_den))
                if pooled_den and pooled_den > 0
                else np.nan
            )
            cohens_d = (
                (mean_a - mean_b) / pooled_std
                if pooled_std and pooled_std > 0
                else np.nan
            )

            parts = ax.violinplot(
                [x_a, x_b], showmeans=True, showextrema=True, widths=0.8
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#87CEEB")
                pc.set_edgecolor("black")
                pc.set_alpha(0.4)

            rng = np.random.default_rng(0)
            jitter_a = 1 + (rng.random(x_a.size) - 0.5) * 0.15
            jitter_b = 2 + (rng.random(x_b.size) - 0.5) * 0.15
            ax.scatter(jitter_a, x_a, s=12, color="#1f77b4", alpha=0.6, linewidths=0)
            ax.scatter(jitter_b, x_b, s=12, color="#d62728", alpha=0.6, linewidths=0)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [
                    f"{label_a}\nGuides: {guides_a}  Genes: {genes_a}",
                    f"{label_b}\nGuides: {guides_b}  Genes: {genes_b}",
                ]
            )
            ax.set_ylabel(pretty_name)
            ax.set_title(f"Cohen's d = {cohens_d:.2f}, p = {p_val:.2e}")
            ax.grid(True, axis="y", alpha=0.3)

            stats_lines.append(
                (
                    f"{pretty_name} [{metric_col}]:\n"
                    f"  {label_a} n={x_a.size}, mean={mean_a:.4g}, std={std_a:.4g}\n"
                    f"  {label_b} n={x_b.size}, mean={mean_b:.4g}, std={std_b:.4g}\n"
                    f"  Guides: {label_a}={guides_a}, {label_b}={guides_b}; Genes: {label_a}={genes_a}, {label_b}={genes_b}\n"
                    f"  Welch t-test: t={t_stat:.6g}, p={p_val:.6g}; Cohen's d: {cohens_d:.6g}\n"
                )
            )

        # Add a single legend listing all gene names for the comparison group (label_b)
        try:
            gene_list = sorted(
                [
                    str(n)
                    for n in df_b.get("gene_name", pd.Series([], dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                    if str(n).strip().lower() != "0"
                ]
            )
            if gene_list:
                legend_text = "Genes (" + label_b + "): " + ", ".join(gene_list)
                legend_text = textwrap.fill(legend_text, width=90)
                # Use an empty handle to host the text in the figure-level legend
                axes[0].plot([], [], " ", label=legend_text)
                fig.legend(loc="upper right", frameon=True, fontsize=8)
        except Exception:
            pass

        fig.tight_layout()
        out_png = out_dir / f"violin_{out_suffix}_grid.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        stats_txt = out_dir / f"violin_{out_suffix}_grid_ttest.txt"
        stats_txt.write_text("\n".join(stats_lines))
        print(f"Saved grid violin plot to {out_png}")

    metric_list_mean = [
        (cell_mean_col, "Cell area (mean)"),
        (nuclei_mean_col, "Nuclei area (mean)"),
        (mito_mean_col, "Mitochondria area (mean)"),
    ]
    metric_list_median = [
        (cell_median_col, "Cell area (median)"),
        (nuclei_median_col, "Nuclei area (median)"),
        (mito_median_col, "Mitochondria area (median)"),
    ]
    _t_test_and_plot_grid_pair(
        metric_list_mean, df_ntc, df_rpl, "NTC", "RPL KOs", "ntc_vs_rpl_mean"
    )
    _t_test_and_plot_grid_pair(
        metric_list_median, df_ntc, df_rpl, "NTC", "RPL KOs", "ntc_vs_rpl_median"
    )
    _t_test_and_plot_grid_pair(
        metric_list_mean, df_ntc, df_mrpl, "NTC", "MRPL KOs", "ntc_vs_mrpl_mean"
    )
    _t_test_and_plot_grid_pair(
        metric_list_median, df_ntc, df_mrpl, "NTC", "MRPL KOs", "ntc_vs_mrpl_median"
    )

    # NTC vs PSM group
    _t_test_and_plot_grid_pair(
        metric_list_mean, df_ntc, df_psm, "NTC", "PSM KOs", "ntc_vs_psm_mean"
    )
    _t_test_and_plot_grid_pair(
        metric_list_median, df_ntc, df_psm, "NTC", "PSM KOs", "ntc_vs_psm_median"
    )

    # NTC vs SRP group
    _t_test_and_plot_grid_pair(
        metric_list_mean, df_ntc, df_srp, "NTC", "SRP KOs", "ntc_vs_srp_mean"
    )
    _t_test_and_plot_grid_pair(
        metric_list_median, df_ntc, df_srp, "NTC", "SRP KOs", "ntc_vs_srp_median"
    )

    # NTC vs SNR group
    _t_test_and_plot_grid_pair(
        metric_list_mean, df_ntc, df_snr, "NTC", "SNR KOs", "ntc_vs_snr_mean"
    )
    _t_test_and_plot_grid_pair(
        metric_list_median, df_ntc, df_snr, "NTC", "SNR KOs", "ntc_vs_snr_median"
    )

    # Run the three requested comparisons
    run_one_pair(
        nuclei_col,
        membrane_or_cell_col,
        title="Nuclei area vs Membrane/Cell area (Gene medians)",
    )
    if mito_col is not None:
        run_one_pair(
            mito_col,
            membrane_or_cell_col,
            title="Mitochondria area vs Cell area (Gene medians)",
        )
        run_one_pair(
            mito_col,
            nuclei_col,
            title="Mitochondria area vs Nuclei area (Gene medians)",
        )


if __name__ == "__main__":
    main()
    # usage: python nuclei_membrane_correlation.py ops0033_20250429
