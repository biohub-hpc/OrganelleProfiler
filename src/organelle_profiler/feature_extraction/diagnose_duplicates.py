"""
Diagnose duplicate rows in per-organelle feature subsets.

Usage:
    python -m organelle_profiler.feature_extraction.diagnose_duplicates -e 94
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ad

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name


def diagnose_duplicates(experiment: str, sample_size: int = 50000):
    """Diagnose why per-organelle feature subsets have many duplicates."""

    # Load data
    dataset = OpsDataset(experiment)

    # Try to load AnnData first (has organelle grouping in var)
    # Check for both naming conventions
    adata_path = dataset.analysis_path / f"{experiment}_cell_features.h5ad"
    if not adata_path.exists():
        adata_path = dataset.analysis_path / "features.h5ad"
    csv_path = dataset.analysis_path / "features.csv"

    if adata_path.exists():
        print(f"Loading AnnData from {adata_path}...")
        adata = ad.read_h5ad(adata_path)
        df = adata.to_df()

        # Get organelle groups from adata.var['organelle']
        if 'organelle' in adata.var.columns:
            organelle_groups = {}
            for feature_name, row in adata.var.iterrows():
                org = row.get('organelle')
                if org is None or pd.isna(org):
                    org = "_unassigned"
                org = str(org)
                if org not in organelle_groups:
                    organelle_groups[org] = []
                organelle_groups[org].append(feature_name)
            print(f"Found {len(organelle_groups)} organelle groups from adata.var['organelle']")
        else:
            print("ERROR: No 'organelle' column in adata.var")
            return None
    elif csv_path.exists():
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        print("ERROR: CSV loaded but need AnnData for organelle grouping")
        return None
    else:
        print(f"No features file found at {adata_path} or {csv_path}")
        return None

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {len(df):,} rows for analysis")

    print(f"\nTotal feature columns: {len(df.columns)}")
    print(f"Organelle groups: {len(organelle_groups)}")

    # Analyze each organelle group
    print("\n" + "="*80)
    print("PER-ORGANELLE DUPLICATE ANALYSIS")
    print("="*80)

    results = []

    for organelle, cols in sorted(organelle_groups.items()):
        # Filter to columns that exist in df
        cols = [c for c in cols if c in df.columns]

        if len(cols) < 2:
            print(f"\n{organelle}: Only {len(cols)} columns - skipped")
            continue

        # Get subset
        org_features = df[cols].copy()

        # Count duplicates
        n_total = len(org_features)
        n_unique = len(org_features.drop_duplicates())
        n_duplicates = n_total - n_unique
        dup_pct = n_duplicates / n_total * 100

        # Analyze why duplicates exist
        # Check 1: How many unique values per column?
        unique_per_col = org_features.nunique()
        avg_unique = unique_per_col.mean()
        min_unique = unique_per_col.min()
        max_unique = unique_per_col.max()

        # Check 2: Are values discretized/quantized?
        sample_col = cols[0]
        sample_values = org_features[sample_col].dropna()
        if len(sample_values) > 0:
            # Check if values are integers or have limited decimal places
            decimals = sample_values.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            avg_decimals = decimals.mean()
        else:
            avg_decimals = 0

        # Check 3: Value range
        value_range = org_features.max().max() - org_features.min().min()

        # Check 4: Are there many zeros?
        zero_pct = (org_features == 0).sum().sum() / org_features.size * 100

        # Check 5: Constant columns
        n_constant = (org_features.nunique() <= 1).sum()

        results.append({
            'organelle': organelle,
            'n_cols': len(cols),
            'n_total': n_total,
            'n_unique': n_unique,
            'dup_pct': dup_pct,
            'avg_unique_per_col': avg_unique,
            'min_unique_per_col': min_unique,
            'max_unique_per_col': max_unique,
            'avg_decimals': avg_decimals,
            'value_range': value_range,
            'zero_pct': zero_pct,
            'n_constant_cols': n_constant,
        })

        print(f"\n{organelle}:")
        print(f"  Columns: {len(cols)}")
        print(f"  Duplicates: {n_duplicates:,} / {n_total:,} ({dup_pct:.1f}%)")
        print(f"  Unique values per column: min={min_unique}, avg={avg_unique:.0f}, max={max_unique}")
        print(f"  Constant columns (<=1 unique): {n_constant}")
        print(f"  Zero values: {zero_pct:.1f}%")
        print(f"  Avg decimal places: {avg_decimals:.1f}")

        # Show the columns with fewest unique values (likely culprits)
        low_unique = unique_per_col.nsmallest(3)
        if low_unique.iloc[0] < 100:
            print(f"  Low-cardinality columns:")
            for col, nuniq in low_unique.items():
                print(f"    - {col}: {nuniq} unique values")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        high_dup = results_df[results_df['dup_pct'] > 20]

        print(f"\nOrganelles with >20% duplicates: {len(high_dup)} / {len(results_df)}")

        # Correlation analysis
        if len(results_df) > 3:
            print("\nCorrelation with duplicate %:")
            for col in ['n_cols', 'avg_unique_per_col', 'zero_pct', 'n_constant_cols']:
                corr = results_df['dup_pct'].corr(results_df[col])
                print(f"  {col}: {corr:.3f}")

        # Diagnosis
        print("\n" + "-"*40)
        print("DIAGNOSIS:")
        print("-"*40)

        avg_cols = results_df['n_cols'].mean()
        avg_unique = results_df['avg_unique_per_col'].mean()
        avg_constant = results_df['n_constant_cols'].mean()

        if avg_cols < 10:
            print("- FEW FEATURES: Organelle groups have few columns on average ({:.0f})".format(avg_cols))
            print("  → With fewer dimensions, identical rows are more likely")

        if avg_unique < 1000:
            print("- LOW CARDINALITY: Features have low unique value counts (avg {:.0f})".format(avg_unique))
            print("  → Suggests quantization or discretization in feature extraction")

        if avg_constant > 0:
            print("- CONSTANT COLUMNS: {:.1f} constant columns per organelle on average".format(avg_constant))
            print("  → These don't contribute to uniqueness")

        # Check if it's a bug or expected behavior
        print("\n" + "-"*40)
        print("IS THIS A BUG?")
        print("-"*40)

        if avg_unique < 100:
            print("LIKELY BUG: Very low cardinality suggests features may be")
            print("  incorrectly computed or stored as integers/categories")
        elif avg_cols < 5:
            print("EXPECTED: With only {:.0f} features per organelle,".format(avg_cols))
            print("  duplicates are mathematically likely in large datasets")
        else:
            print("UNCERTAIN: Moderate duplicates may be normal for this data")
            print("  Consider checking if feature extraction is working correctly")

        # Show sample duplicated rows for inspection
        print("\n" + "-"*40)
        print("SAMPLE DUPLICATED ROWS (first organelle with >10% dups):")
        print("-"*40)

        for _, row in results_df.iterrows():
            if row['dup_pct'] > 10:
                org = row['organelle']
                cols = organelle_groups[org]
                cols = [c for c in cols if c in df.columns]

                org_features = df[cols]
                dup_mask = org_features.duplicated(keep=False)

                if dup_mask.sum() > 0:
                    print(f"\nOrganelle: {org}")
                    print(f"Sample duplicated feature vectors:")

                    # Get first group of duplicates
                    dup_rows = org_features[dup_mask]
                    first_dup_row = dup_rows.iloc[0]
                    matching = (dup_rows == first_dup_row).all(axis=1)
                    n_matching = matching.sum()

                    print(f"  {n_matching} rows have identical values:")
                    print(f"  {dict(first_dup_row.iloc[:5])}...")  # Show first 5 features
                    break

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose duplicate rows in feature data")
    parser.add_argument("-e", "--experiment", required=True, help="Experiment name or shorthand")
    parser.add_argument("--sample", type=int, default=50000, help="Sample size for analysis")
    args = parser.parse_args()

    experiment = resolve_experiment_name(args.experiment, allow_interactive=False, autoselect=True)
    diagnose_duplicates(experiment, sample_size=args.sample)
