"""
Persistent Homology Analysis
Maps the stability of connected topological components (Betti-0)
using 2D projection for accurate connectivity mapping.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.ndimage import label
from pathlib import Path
import logging
import sys
import argparse

def run_topology_audit():
    parser = argparse.ArgumentParser(description="Persistent Homology Audit v3.2")
    parser.add_argument("--nside", type=int, default=64, help="NSIDE resolution")
    parser.add_argument("--xsize", type=int, default=1024, help="Cartesian projection width in pixels")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data folder")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    np.random.seed(42)

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    map_path = data_dir / "planck_smica.fits"
    logging.info("CMB-AUDIT: Initializing Persistent Homology Audit (Betti-0 v3.2)...")

    if not map_path.exists():
        logging.critical(f"ERROR: Target dataset not found at {map_path}")
        return

    # Data Preparation and Normalization
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64)
        data = hp.ud_grade(cmb_map, args.nside)
        data = (data - np.mean(data)) / np.std(data)
    except Exception as e:
        logging.exception(f"FAILED to process FITS data: {e}")
        return

    # 2D Projection
    logging.info(f"Projecting spherical map to 2D Cartesian plane ({args.xsize}px)...")
    proj_map = hp.cartview(data, xsize=args.xsize, return_projected_map=True)
    plt.close()

    proj_map = np.nan_to_num(proj_map, nan=0.0)

    # Filtration Analysis (Betti-0)
    thresholds = np.linspace(-3.0, 3.0, 60)
    persistence = []

    logging.info("Mapping topological persistence across 60 filtration levels...")

    for t in thresholds:
        binary_map = proj_map > t
        _, num_features = label(binary_map)
        persistence.append(num_features)

    persistence = np.array(persistence)

    # Theoretical Gaussian Baseline
    theoretical_baseline = (1 - thresholds**2) * np.exp(-thresholds**2 / 2)
    scale_factor = np.max(persistence) / np.max(np.abs(theoretical_baseline))
    theoretical_baseline *= scale_factor

    # Save raw data
    np.save(results_dir / "topology_thresholds_v3.npy", thresholds)
    np.save(results_dir / "topology_persistence_v3.npy", persistence)
    np.save(results_dir / "topology_theoretical_v3.npy", theoretical_baseline)
    np.save(results_dir / "projected_map_v3.npy", np.asarray(proj_map))

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, persistence, color='lime', linewidth=2.2, label='Empirical Persistence (CMB)')
    plt.plot(thresholds, theoretical_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
             label='Stochastic Baseline (Gaussian Random Field)')

    plt.title("Persistent Homology Audit: Structural Stability (Betti-0)")
    plt.xlabel("Energy Threshold (σ)")
    plt.ylabel("Number of Connected Components (Betti-0)")
    plt.legend()
    plt.grid(axis='both', alpha=0.1)

    output_plot = results_dir / "topology_persistence_audit_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Verdict (Area Under Curve Ratio)
    area_actual = np.trapezoid(persistence)
    area_baseline = np.trapezoid(theoretical_baseline)
    persistence_ratio = area_actual / area_baseline

    logging.info("\n" + "="*65)
    logging.info("PERSISTENT HOMOLOGY VERDICT")
    logging.info(f"Structural Persistence Index: {persistence_ratio:.4f}")
    logging.info("="*65)

    if persistence_ratio > 1.25:
        logging.info("VERDICT: PERSISTENT GEOMETRIC ARCHITECTURE DETECTED.")
    else:
        logging.info("RESULT: Consistent with Gaussian Random Field expectations.")

    logging.info(f"\nAudit plot archived at: {output_plot}")

if __name__ == "__main__":
    run_topology_audit()