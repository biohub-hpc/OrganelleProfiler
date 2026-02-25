"""
Feature Graph Generation Pipeline v3 - Stage-Based Architecture.

This is the launcher for the refactored fe_graphs package, which provides
a modular, hierarchical framework for phenotypic analysis.

Architecture:
-------------
- Level-based: Analysis flows cell → guide → gene with result chaining
- Stage-based: Each level runs QC → Embedding → Clustering → Differential → Summary
- Consistent: Same analytical suite runs at every level

Usage:
------
# Generate graphs for an experiment (shorthand resolves to full name)
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e 94

# Run with full experiment name
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e ops0094_20251217

# Debug mode with 10% of cells
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e 94 --debug 0.1

# Run only specific levels
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e 94 --levels cell gene

# Skip specific stages
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e 94 --skip-stages qc clustering

# Force CPU usage (no cuML GPU acceleration)
python -m organelle_profiler.feature_extraction.fe_graphs_v2 -e 94 --no-cuml

Output:
-------
Graphs are saved to: {experiment}/4-features/graphs/
    1_cell_level/
    2_guide_level/
    3_gene_level/
    4_integrated/

See fe_graphs/FE_GRAPHS_ARCHITECTURE.md for full documentation.
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.getcwd())


def list_available_options():
    """Print available levels and stages."""
    print("\n" + "=" * 60)
    print("AVAILABLE OPTIONS")
    print("=" * 60)
    
    print("\nLevels (--levels):")
    print("  cell    - Individual cell analysis (~millions of items)")
    print("  guide   - sgRNA guide aggregated analysis (~thousands)")
    print("  gene    - Gene aggregated analysis (~hundreds)")
    
    print("\nStages (--stages or --skip-stages) - in execution order:")
    print("  1. qc                        - Basic quality control (batch effects, feature quality)")
    print("  2. spatial_drift             - Comprehensive radial drift analysis (cell level)")
    print("  3. embedding                 - Dimensionality reduction (PCA, UMAP)")
    print("  4. embedding_visualization   - Comprehensive annotated UMAP plots")
    print("  5. positive_controls         - Validate using known gene clusters (all levels) [early!]")
    print("  6. clustering                - Unsupervised clustering (HDBSCAN, KMeans, Leiden)")
    print("  7. differential              - NTC comparison (z-scores, volcano plots)")
    print("  8. organelle_discrimination  - Compare organelle discrimination power")
    print("  9. gene_relationships        - PHATE clustering & hierarchical analysis (gene level)")
    print(" 10. cp_comparison             - Cell Painting vs non-CP comparison (with --just-cp)")
    print(" 11. cp_challenge              - CP vs live-cell mAP head-to-head per organelle")
    print(" 12. summary                   - Result aggregation and hit lists")
    
    print("\nCache Options:")
    print("  --no-cache     - Ignore cached results and recompute everything")
    print("  --clear-cache  - Clear cache before running (then use cache normally)")
    print("  (default)      - Use cached embeddings/clustering if available")
    
    print("\nExamples:")
    print("  # Full pipeline")
    print("  fe_graphs_v2.py -e 94")
    print("")
    print("  # Gene-level only, skip QC")
    print("  fe_graphs_v2.py -e 94 --levels gene --skip-stages qc")
    print("")
    print("  # Cell and gene levels, embedding and clustering only")
    print("  fe_graphs_v2.py -e 94 --levels cell gene --stages embedding clustering")
    print("")
    print("  # Force recompute everything (ignore cache)")
    print("  fe_graphs_v2.py -e 94 --no-cache")
    print("")
    print("  # Clear cache first, then run normally")
    print("  fe_graphs_v2.py -e 94 --clear-cache")
    print()


def main():
    """Main execution function."""
    from ops_utils.data.filesystem import resolve_experiment_name
    from .fe_graphs import AnalysisOrchestrator
    
    parser = argparse.ArgumentParser(
        description="Generate graphs from morphological feature data (v3 - stage-based architecture).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -e 94                              # Full pipeline (cell → guide → gene)
  %(prog)s -e 94 --debug 0.1                  # Debug mode (10%% of cells)
  %(prog)s -e 94 --levels gene                # Gene-level only
  %(prog)s -e 94 --skip-stages qc clustering  # Skip QC and clustering
  %(prog)s --list-options                     # Show available levels and stages
        """
    )
    
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        help="Name or shorthand for the experiment (e.g., '94', 'ops94', 'ops0094_20251217').",
    )
    parser.add_argument(
        "--debug",
        type=float,
        default=None,
        help="Fraction of cells to sample for debug mode (e.g., 0.1 for 10%%).",
    )
    parser.add_argument(
        "--skip_object_features",
        action="store_true",
        help="If set, do not load object-level feature CSVs to speed up execution.",
    )
    parser.add_argument(
        "--just-cp",
        action="store_true",
        help="If set, only include Cell Painting organelles (exclude nuclear_seg, cell_seg, phase2d, focus3d groups).",
    )
    parser.add_argument(
        "--no-cuml",
        action="store_true",
        help="If set, disable GPU acceleration with cuML and force CPU usage.",
    )
    parser.add_argument(
        "--cluster-algo",
        type=str,
        default="all",
        choices=["all", "hdbscan", "kmeans", "leiden"],
        help="Clustering algorithm(s) to use. 'all' runs all three methods.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If set, generate data for interactive dashboards (skipped by default).",
    )
    parser.add_argument(
        "--drift-method",
        type=str,
        default="three_segment_regression",
        choices=["three_segment_regression", "segmented_regression", "second_derivative"],
        help="Method to determine breakpoint in radial drift plots.",
    )
    
    # Cache control
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="If set, ignore cached embeddings/clustering and recompute everything.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached results before running (then use cache normally).",
    )
    
    # Level and stage control (new in v3)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=str,
        choices=["cell", "guide", "gene"],
        help="Levels to run (default: all). Example: --levels cell gene",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=str,
        choices=["qc", "spatial_drift", "embedding", "embedding_visualization", "clustering", "differential", "organelle_discrimination", "positive_controls", "gene_relationships", "cp_comparison", "cp_challenge", "summary"],
        help="Specific stages to run. Example: --stages embedding clustering gene_relationships",
    )
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        type=str,
        choices=["qc", "spatial_drift", "embedding", "embedding_visualization", "clustering", "differential", "organelle_discrimination", "positive_controls", "gene_relationships", "cp_comparison", "cp_challenge", "summary"],
        help="Stages to skip. Example: --skip-stages qc",
    )
    
    # Legacy compatibility
    parser.add_argument(
        "--guide-analysis",
        action="store_true",
        help="[DEPRECATED] Use --levels guide instead. Only run guide-level analysis.",
    )
    parser.add_argument(
        "--gene-analysis",
        action="store_true",
        help="[DEPRECATED] Use --levels gene instead. Only run gene-level analysis.",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        type=str,
        help="[DEPRECATED] Use --stages instead. Specific analyses to run.",
    )
    
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List all available levels and stages and exit.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip generating plots/images that already exist on disk.",
    )

    args = parser.parse_args()
    
    # Handle --list-options
    if args.list_options:
        list_available_options()
        return
    
    # Require experiment if not listing
    if not args.experiment:
        parser.error("the following arguments are required: -e/--experiment")
    
    # Resolve experiment name (e.g., "94" -> "ops0094_20251217")
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True, autoselect=True)
    
    # Determine analysis mode from legacy args
    analysis_mode = "all"
    if args.guide_analysis and args.gene_analysis:
        analysis_mode = "guide_and_gene"
    elif args.guide_analysis:
        analysis_mode = "guide"
    elif args.gene_analysis:
        analysis_mode = "gene"
    
    # New levels arg takes precedence
    levels = args.levels
    if levels is None and analysis_mode != "all":
        # Convert legacy mode to levels
        if analysis_mode == "guide":
            levels = ["guide"]
        elif analysis_mode == "gene":
            levels = ["gene"]
        elif analysis_mode == "guide_and_gene":
            levels = ["guide", "gene"]
    
    try:
        # Create orchestrator
        orchestrator = AnalysisOrchestrator(
            experiment,
            debug_cell_fraction=args.debug,
            skip_object_features=args.skip_object_features,
            just_cell_painting=args.just_cp,
            use_cuml=not args.no_cuml,
            cluster_algo=args.cluster_algo,
            skip_interactive_plots=not args.interactive,
            drift_method=args.drift_method,
            analysis_mode=analysis_mode,
            use_cache=not args.no_cache,
            skip_complete=args.skip_complete,
        )
        
        # Handle --clear-cache
        if args.clear_cache:
            from fe_graphs.core.fe_graphs_cache import EmbeddingCache
            cache = EmbeddingCache(orchestrator.data.cache_dir)
            n_cleared = cache.clear_all()
            print(f"Cleared {n_cleared} cached files from {orchestrator.data.cache_dir}")
        
        # Run analyses
        result = orchestrator.run(
            levels=levels,
            stages=args.stages,
            skip_stages=args.skip_stages,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        for level, level_result in result.level_results.items():
            status = "OK" if level_result.success else "ERRORS"
            n_files = len(level_result.get_all_files())
            print(f"\n  {level.upper()} Level ({n_files} files) [{status}]")
            
            for stage_name, stage_result in level_result.stages.items():
                stage_status = "✓" if stage_result.success else "✗"
                stage_files = len(stage_result.output_files)
                print(f"    {stage_status} {stage_name}: {stage_files} files")
                
                if stage_result.errors:
                    for error in stage_result.errors:
                        print(f"        Error: {error}")
        
        print(f"\nTotal output files: {len(result.all_output_files)}")
        print(f"Output directory: {orchestrator.graph_output_path}")
        
        if result.errors:
            print(f"\nWarning: {len(result.errors)} errors occurred during execution")
            sys.exit(1)
        else:
            print("\nAll analyses completed successfully")
        
    except FileNotFoundError as e:
        print(f"Execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


def run_graphs(experiment: str, **kwargs):
    """
    Programmatic entry point for running the graph generation pipeline.

    Called by the orchestrator to run feature graphs for an experiment.

    Args:
        experiment: Experiment name (e.g., 'ops0094_20251217')
        **kwargs: Additional parameters passed to AnalysisOrchestrator
            - debug_cell_fraction: Fraction of cells to sample for debug mode
            - skip_object_features: Skip loading object-level features
            - use_cuml: Use GPU acceleration with cuML (default True)
            - cluster_algo: Clustering algorithm ('all', 'hdbscan', 'kmeans', 'leiden')
            - skip_interactive_plots: Skip interactive dashboard generation
            - drift_method: Method for radial drift breakpoint detection
            - use_cache: Use cached embeddings/clustering (default True)
            - levels: List of levels to run ('cell', 'guide', 'gene')
            - stages: List of specific stages to run
            - skip_stages: List of stages to skip
    """
    from ops_utils.data.filesystem import resolve_experiment_name
    from .fe_graphs import AnalysisOrchestrator

    # Resolve experiment name
    resolved_experiment = resolve_experiment_name(
        experiment, allow_interactive=False, autoselect=True
    )

    # Extract run-specific params
    levels = kwargs.pop("levels", None)
    stages = kwargs.pop("stages", None)
    skip_stages = kwargs.pop("skip_stages", None)

    # Create orchestrator with remaining kwargs
    orchestrator = AnalysisOrchestrator(resolved_experiment, **kwargs)

    # Run analysis
    result = orchestrator.run(
        levels=levels,
        stages=stages,
        skip_stages=skip_stages,
    )

    # Log summary
    total_files = len(result.all_output_files)
    print(f"[fe_graphs_v2] Completed: {total_files} output files generated")
    print(f"[fe_graphs_v2] Output directory: {orchestrator.graph_output_path}")

    if result.errors:
        print(f"[fe_graphs_v2] Warning: {len(result.errors)} errors occurred")
        for error in result.errors:
            print(f"  - {error}")

    return result


if __name__ == "__main__":
    main()
