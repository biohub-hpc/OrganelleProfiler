import os
import sys
import subprocess
from pathlib import Path
import argparse


def find_interactive_data_path(experiment_name: str = None) -> Path:
    """
    Finds the interactive data directory based on the project structure.

    If an experiment_name is provided, it builds the path directly.
    Otherwise, it finds the most recently modified experiment folder.
    """
    base_path = Path("/hpc/projects/intracellular_dashboard/fast_ops")
    if not base_path.is_dir():
        print(f"Error: The base project directory was not found at '{base_path}'.")
        return None

    experiment_path = None
    if experiment_name:
        experiment_path = base_path / experiment_name
        if not experiment_path.is_dir():
            print(
                f"Error: The specified experiment directory does not exist: {experiment_path}"
            )
            return None
        print(f"Targeting specified experiment: {experiment_name}")
    else:
        print("No experiment specified. Searching for the latest experiment...")
        # Get all subdirectories in the base path
        experiments = [d for d in base_path.iterdir() if d.is_dir()]
        if not experiments:
            print(f"Error: No experiment folders found in '{base_path}'.")
            return None

        try:
            # Find the most recently modified experiment directory
            experiment_path = max(experiments, key=lambda p: p.stat().st_mtime)
            print(f"Found latest experiment: {experiment_path.name}")
        except FileNotFoundError:
            print(
                "Error: A file was not found while checking modification times. Check for broken symbolic links."
            )
            return None

    # Construct the final path using the logic from experiment.py
    return (
        experiment_path
        / "3-assembly"
        / "feature_extraction"
        / "graphs"
        / "interactive_umaps"
    )


def main():
    """
    Finds available interactive plot data and presents a menu to the user
    for launching the Dash dashboard.
    """
    parser = argparse.ArgumentParser(
        description="Launch an interactive UMAP dashboard for an experiment.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        help="Specify the experiment name (e.g., 'ops0049_20250626').\nIf not provided, the most recently modified experiment will be used.",
    )
    args = parser.parse_args()

    # 1. Find the directory containing the parquet files
    interactive_dir = find_interactive_data_path(args.experiment)

    if not interactive_dir or not interactive_dir.exists():
        print(
            f"\nError: Could not find an 'interactive_umaps' data directory for the target experiment."
        )
        print(
            "Please run the fe_graphs.py analysis script first to generate the data.\n"
        )
        sys.exit(1)

    # 2. Find all .parquet files and sort them by modification time (newest first)
    try:
        parquet_files = sorted(
            list(interactive_dir.glob("interactive_data_*.parquet")),
            key=os.path.getmtime,
            reverse=True,
        )
    except FileNotFoundError:
        print(
            f"Error: Search for .parquet files failed in '{interactive_dir}'. Directory may have been removed."
        )
        sys.exit(1)

    if not parquet_files:
        print(f"Error: No interactive data (.parquet) files found in {interactive_dir}")
        print("Please run the fe_graphs.py analysis script first.")
        sys.exit(1)

    # 3. Present choices to the user
    print("\n--- Interactive UMAP Dashboard Launcher ---\n")
    print("Please choose a dashboard to launch:")
    for i, f in enumerate(parquet_files):
        # Make the filename more readable for the menu
        name = (
            f.stem.replace("interactive_data_", "")
            .replace("_", " ")
            .replace("original", "All Features")
        )
        print(f"  [{i+1}] {name.title()}  ({f.name})")

    print(f"  [0] Exit")

    # 4. Get user input
    while True:
        try:
            choice = int(input("\nEnter your choice: "))
            if 0 <= choice <= len(parquet_files):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 0:
        print("Exiting.")
        sys.exit(0)

    selected_file = parquet_files[choice - 1]

    # 5. Launch the dashboard script as a subprocess
    print(f"\nLaunching dashboard for: {selected_file.name}")

    # Use '-m' to run the dashboard as a module, which is robust
    command = [
        sys.executable,  # Use the same python interpreter that is running this script
        "-m",
        "organelle_profiler.extra.interactive_dashboard",
        str(selected_file),
    ]

    try:
        # The dashboard will run until the user stops it with Ctrl+C
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError launching dashboard: {e}\n")
    except FileNotFoundError:
        print(
            "\nError: 'python' command not found. Make sure your environment is set up correctly.\n"
        )
    except KeyboardInterrupt:
        print("\nDashboard closed by user. Exiting launcher.")


if __name__ == "__main__":
    main()
