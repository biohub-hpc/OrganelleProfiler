#!/usr/bin/env python
"""
Quick script to submit a single feature extraction batch directly to SLURM.
Skips all preprocessing since batch metadata already exists.

Usage:
    # Combined mode (default): GPU + CPU in one job
    python submit_single_batch.py -e 94 -b A_1_0_0000

    # GPU-only mode: morphology + localization, saves network task metadata
    python submit_single_batch.py -e 94 -b A_1_0_0000 --mode gpu

    # CPU-only mode: network analysis using GPU phase output (single node)
    python submit_single_batch.py -e 94 -b A_1_0_0000 --mode cpu

    # Both: submit GPU job, then CPU job with SLURM dependency
    python submit_single_batch.py -e 94 -b A_1_0_0000 --mode both

    # SPMD mode: multi-node network analysis (256 MPI ranks + merge job)
    python submit_single_batch.py -e 94 -b A_1_0_9999 --mode spmd --ntasks 256
"""

import argparse
import subprocess
import sys
import submitit
from pathlib import Path

from organelle_profiler.feature_extraction.feature_extraction_slurm import (
    feature_extraction_batch_worker,
    gpu_batch_worker,
    cpu_network_batch_worker,
    FE_SLURM_PARAMS_BASE,
    FE_GPU_SLURM_PARAMS,
    FE_CPU_SLURM_PARAMS,
)


def _parse_batch_id(batch_id: str):
    """Parse batch_id to get well and batch_idx."""
    parts = batch_id.rsplit('_', 1)
    well_safe = parts[0]  # A_1_0
    batch_idx = int(parts[1])  # 0000 -> 0
    well = well_safe.replace('_', '/')  # A/1/0
    return well, well_safe, batch_idx


def submit_combined(experiment: str, batch_id: str, output_dir: Path, log_dir: Path):
    """Submit a single combined (GPU+CPU) batch job to SLURM."""
    well, well_safe, batch_idx = _parse_batch_id(batch_id)

    batch_meta_path = output_dir / "_batch_metadata" / f"batch_{batch_id}_meta.parquet"
    if not batch_meta_path.exists():
        print(f"ERROR: Batch metadata not found: {batch_meta_path}")
        return None

    print(f"Submitting COMBINED batch {batch_id}")
    print(f"  Well: {well}")
    print(f"  Batch index: {batch_idx}")
    print(f"  Metadata: {batch_meta_path}")
    print(f"  Output: {output_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)
    params = {**FE_SLURM_PARAMS_BASE}
    params["slurm_constraint"] = "a100_80|h100|h200"
    executor.update_parameters(**params)

    job = executor.submit(
        feature_extraction_batch_worker,
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        batch_cells_path=str(batch_meta_path),
        output_dir=str(output_dir),
        full_features=True,
        sequential=False,
    )

    print(f"  Job ID: {job.job_id}")
    return job


def submit_gpu_only(experiment: str, batch_id: str, output_dir: Path, log_dir: Path,
                    n_workers: int = None, io_threads: int = None, max_cells: int = None):
    """Submit GPU-only phase to SLURM."""
    well, well_safe, batch_idx = _parse_batch_id(batch_id)

    batch_meta_path = output_dir / "_batch_metadata" / f"batch_{batch_id}_meta.parquet"
    if not batch_meta_path.exists():
        print(f"ERROR: Batch metadata not found: {batch_meta_path}")
        return None

    print(f"Submitting GPU-ONLY batch {batch_id}")
    print(f"  Well: {well}")
    print(f"  Batch index: {batch_idx}")
    print(f"  Output: {output_dir}")
    if n_workers:
        print(f"  Workers: {n_workers}")
    if io_threads:
        print(f"  I/O threads: {io_threads}")
    if max_cells:
        print(f"  Max cells: {max_cells}")

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(**FE_GPU_SLURM_PARAMS)

    job = executor.submit(
        gpu_batch_worker,
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        batch_cells_path=str(batch_meta_path),
        output_dir=str(output_dir),
        full_features=True,
        n_workers=n_workers,
        io_threads=io_threads,
        max_cells=max_cells,
    )

    print(f"  GPU Job ID: {job.job_id}")
    return job


def submit_cpu_only(experiment: str, batch_id: str, output_dir: Path, log_dir: Path):
    """Submit CPU-only network analysis phase to SLURM."""
    well, well_safe, batch_idx = _parse_batch_id(batch_id)

    # Check that GPU outputs exist
    batch_results_dir = output_dir / "_batch_results"
    gpu_features_path = batch_results_dir / f"batch_{batch_id}_gpu_features.parquet"
    network_tasks_path = batch_results_dir / f"batch_{batch_id}_network_tasks.parquet"

    if not gpu_features_path.exists():
        print(f"ERROR: GPU features not found: {gpu_features_path}")
        print(f"  Run GPU phase first: --mode gpu")
        return None

    if not network_tasks_path.exists():
        print(f"ERROR: Network tasks not found: {network_tasks_path}")
        print(f"  Run GPU phase first: --mode gpu")
        return None

    print(f"Submitting CPU-ONLY batch {batch_id}")
    print(f"  Well: {well}")
    print(f"  Batch index: {batch_idx}")
    print(f"  Output: {output_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(**FE_CPU_SLURM_PARAMS)

    job = executor.submit(
        cpu_network_batch_worker,
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        output_dir=str(output_dir),
    )

    print(f"  CPU Job ID: {job.job_id}")
    return job


def submit_cpu_spmd(experiment: str, batch_id: str, output_dir: Path, log_dir: Path, ntasks: int = 256):
    """Submit SPMD multi-node CPU job: N MPI ranks + separate merge job."""
    batch_results_dir = output_dir / "_batch_results"
    gpu_features_path = batch_results_dir / f"batch_{batch_id}_gpu_features.parquet"
    network_tasks_path = batch_results_dir / f"batch_{batch_id}_network_tasks.parquet"

    if not gpu_features_path.exists():
        print(f"ERROR: GPU features not found: {gpu_features_path}")
        return None
    if not network_tasks_path.exists():
        print(f"ERROR: Network tasks not found: {network_tasks_path}")
        return None

    log_dir.mkdir(parents=True, exist_ok=True)

    # Use current conda environment's Python and codebase path
    conda_python = sys.executable
    ops_process_main = Path(__file__).resolve().parents[3]

    # Job 1: SPMD worker (N ranks, 1 CPU each)
    partials_dir = f"{output_dir}/_batch_results/_partials_{batch_id}"
    spmd_script = f"""#!/bin/bash
#SBATCH --job-name=fe_spmd_{batch_id}
#SBATCH --partition=preempted
#SBATCH --requeue
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output={log_dir}/spmd_{batch_id}_%j.out
#SBATCH --error={log_dir}/spmd_{batch_id}_%j.err

cd {ops_process_main}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun {conda_python} -u -m organelle_profiler.feature_extraction.spmd_cpu_worker \\
    --experiment {experiment} \\
    --batch-id {batch_id} \\
    --output-dir '{output_dir}' \\
    --restart
"""
    spmd_script_path = log_dir / f"spmd_{batch_id}.sh"
    spmd_script_path.write_text(spmd_script)

    result = subprocess.run(["sbatch", str(spmd_script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: sbatch failed for SPMD worker: {result.stderr}")
        return None

    spmd_job_id = result.stdout.strip().split()[-1]

    # Job 2: Merge (single process, 32 CPUs for parallel reads + aggregation, depends on SPMD)
    merge_script = f"""#!/bin/bash
#SBATCH --job-name=fe_merge_{batch_id}
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --dependency=afterok:{spmd_job_id}
#SBATCH --output={log_dir}/merge_{batch_id}_%j.out
#SBATCH --error={log_dir}/merge_{batch_id}_%j.err

cd {ops_process_main}

{conda_python} -m organelle_profiler.feature_extraction.spmd_cpu_merge \\
    --experiment {experiment} \\
    --batch-id {batch_id} \\
    --output-dir '{output_dir}'
"""
    merge_script_path = log_dir / f"merge_{batch_id}.sh"
    merge_script_path.write_text(merge_script)

    result = subprocess.run(["sbatch", str(merge_script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: sbatch failed for merge job: {result.stderr}")
        return None

    merge_job_id = result.stdout.strip().split()[-1]

    print(f"Submitting SPMD batch {batch_id}")
    print(f"  Tasks: {ntasks} (SLURM scatters across nodes)")
    print(f"  Output: {output_dir}")
    print(f"  SPMD Job ID: {spmd_job_id}")
    print(f"  Merge Job ID: {merge_job_id} (depends on {spmd_job_id})")
    return spmd_job_id, merge_job_id


def submit_both(experiment: str, batch_id: str, output_dir: Path, log_dir: Path):
    """Submit GPU job then CPU job with SLURM dependency."""
    well, well_safe, batch_idx = _parse_batch_id(batch_id)

    batch_meta_path = output_dir / "_batch_metadata" / f"batch_{batch_id}_meta.parquet"
    if not batch_meta_path.exists():
        print(f"ERROR: Batch metadata not found: {batch_meta_path}")
        return None

    print(f"Submitting BOTH phases for batch {batch_id}")
    print(f"  Well: {well}")
    print(f"  Batch index: {batch_idx}")
    print(f"  Output: {output_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Submit GPU job first
    gpu_executor = submitit.AutoExecutor(folder=log_dir)
    gpu_executor.update_parameters(**FE_GPU_SLURM_PARAMS)

    gpu_job = gpu_executor.submit(
        gpu_batch_worker,
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        batch_cells_path=str(batch_meta_path),
        output_dir=str(output_dir),
        full_features=True,
    )
    print(f"  GPU Job ID: {gpu_job.job_id}")

    # Submit CPU job with dependency on GPU job
    cpu_executor = submitit.AutoExecutor(folder=log_dir)
    cpu_params = {**FE_CPU_SLURM_PARAMS}
    cpu_params["slurm_additional_parameters"] = {"dependency": f"afterok:{gpu_job.job_id}"}
    cpu_executor.update_parameters(**cpu_params)

    cpu_job = cpu_executor.submit(
        cpu_network_batch_worker,
        experiment=experiment,
        well=well,
        batch_idx=batch_idx,
        output_dir=str(output_dir),
    )
    print(f"  CPU Job ID: {cpu_job.job_id} (depends on {gpu_job.job_id})")

    return gpu_job, cpu_job


def main():
    parser = argparse.ArgumentParser(description="Submit single feature extraction batch")
    parser.add_argument("-e", "--experiment", type=int, required=True, help="Experiment number (e.g., 94)")
    parser.add_argument("-b", "--batch", type=str, required=True, help="Batch ID (e.g., A_1_0_0000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: auto)")
    parser.add_argument(
        "--mode", type=str, default="combined",
        choices=["combined", "gpu", "cpu", "both", "spmd"],
        help="Execution mode: combined (default), gpu (GPU only), cpu (CPU only, single node), "
             "both (GPU then CPU with dependency), spmd (multi-node SPMD network analysis)",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override Dask worker count (benchmark, Dask mode only)")
    parser.add_argument("--io-threads", type=int, default=None, help="Override I/O threads (benchmark)")
    parser.add_argument("--max-cells", type=int, default=None, help="Limit cell count for quick benchmarking")
    parser.add_argument("--ntasks", type=int, default=256, help="Number of MPI ranks for SPMD mode (default: 256)")
    args = parser.parse_args()

    # Resolve experiment name
    experiment = f"ops{args.experiment:04d}_20251217_mark"

    # Default output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"/hpc/projects/intracellular_dashboard/fastops/{experiment}/3-assembly/feature_extraction/_preview")

    # Log directory - use current codebase path
    ops_process_main = Path(__file__).resolve().parents[3]
    log_dir = ops_process_main / f"slurm_logs/slurm_feature_extraction_logs/{experiment}"

    if args.mode == "combined":
        job = submit_combined(experiment, args.batch, output_dir, log_dir)
        if job:
            print(f"\nJob submitted! Monitor with: tail -f {log_dir}/{job.job_id}/{job.job_id}_0_log.out")

    elif args.mode == "gpu":
        job = submit_gpu_only(experiment, args.batch, output_dir, log_dir,
                              n_workers=args.workers, io_threads=args.io_threads,
                              max_cells=args.max_cells)
        if job:
            print(f"\nGPU job submitted! Monitor with: tail -f {log_dir}/{job.job_id}/{job.job_id}_0_log.out")

    elif args.mode == "cpu":
        job = submit_cpu_only(experiment, args.batch, output_dir, log_dir)
        if job:
            print(f"\nCPU job submitted! Monitor with: tail -f {log_dir}/{job.job_id}/{job.job_id}_0_log.out")

    elif args.mode == "spmd":
        result = submit_cpu_spmd(experiment, args.batch, output_dir, log_dir, ntasks=args.ntasks)
        if result:
            spmd_job_id, merge_job_id = result
            print(f"\nSPMD jobs submitted!")
            print(f"  SPMD: tail -f {log_dir}/spmd_{args.batch}_{spmd_job_id}.out")
            print(f"  Merge: tail -f {log_dir}/merge_{args.batch}_{merge_job_id}.out")

    elif args.mode == "both":
        result = submit_both(experiment, args.batch, output_dir, log_dir)
        if result:
            gpu_job, cpu_job = result
            print(f"\nBoth jobs submitted!")
            print(f"  GPU: tail -f {log_dir}/{gpu_job.job_id}/{gpu_job.job_id}_0_log.out")
            print(f"  CPU: tail -f {log_dir}/{cpu_job.job_id}/{cpu_job.job_id}_0_log.out")
            print(f"  CPU job {cpu_job.job_id} will start after GPU job {gpu_job.job_id} completes")


if __name__ == "__main__":
    main()
