"""
Minkowski Geometric Audit
Euler Characteristic across energy thresholds.
Uses accurate 2D Cartesian projection and 8-connectivity labeling.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.ndimage import label
from pathlib import Path
import logging
import sys
import argparse

def run_minkowski_audit():
    parser = argparse.ArgumentParser(description="Minkowski Geometric Audit v3.2")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
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
    logging.info("CMB-AUDIT: Initializing Minkowski Geometric Audit (Topology Check v3.2)...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    # Load and condition dataset
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64)
        data = hp.ud_grade(cmb_map, args.nside)
        data = (data - np.mean(data)) / np.std(data)
    except Exception as e:
        logging.exception(f"FAILED to process map data: {e}")
        return

    # 2D Projection
    logging.info(f"Projecting spherical map to 2D Cartesian plane ({args.xsize}px)...")
    proj_map = hp.cartview(data, xsize=args.xsize, return_projected_map=True)
    plt.close()

    proj_map = np.nan_to_num(proj_map, nan=0.0)

    # Topological calculation
    thresholds = np.linspace(-3.0, 3.0, 60)
    euler_chars = []

    conn_8 = np.ones((3, 3))   # 8-connectivity

    logging.info("Calculating Euler Characteristic using 8-connectivity...")

    for t in thresholds:
        binary_islands = proj_map > t
        _, num_islands = label(binary_islands, structure=conn_8)
        binary_voids = proj_map < t
        _, num_voids = label(binary_voids, structure=conn_8)
        euler_chars.append(num_islands - num_voids)

    euler_chars = np.array(euler_chars)

    # Theoretical baseline
    theoretical = (1 - thresholds**2) * np.exp(-thresholds**2 / 2)
    scale = np.max(np.abs(euler_chars)) / np.max(np.abs(theoretical))
    theoretical *= scale

    # Save data
    np.save(results_dir / "minkowski_thresholds_v3.npy", thresholds)
    np.save(results_dir / "minkowski_euler_chars_v3.npy", euler_chars)
    np.save(results_dir / "minkowski_theoretical_v3.npy", theoretical)
    np.save(results_dir / "projected_map_v3.npy", np.asarray(proj_map))

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, euler_chars, 'o-', color='cyan', linewidth=2, label='Empirical Topology (CMB)')
    plt.plot(thresholds, theoretical, '--', color='red', linewidth=2, alpha=0.7,
             label='Gaussian Random Field (Adler Formula)')

    plt.title("Minkowski Audit v3.2: Euler Characteristic and Topological Consistency")
    plt.xlabel("Temperature Threshold (σ)")
    plt.ylabel("Euler Characteristic (χ)")
    plt.legend()
    plt.grid(True, alpha=0.1)

    output_plot = results_dir / "minkowski_topology_audit_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Deviation analysis
    diff = np.abs(euler_chars - theoretical)
    max_deviation = np.max(diff) / np.max(np.abs(euler_chars))

    logging.info("\n" + "="*70)
    logging.info("MINKOWSKI TOPOLOGICAL VERDICT")
    logging.info(f"Maximum relative deviation: {max_deviation:.4%}")
    logging.info("="*70)

    if max_deviation > 0.15:
        logging.info("VERDICT: SIGNIFICANT TOPOLOGICAL ANOMALY DETECTED.")
    else:
        logging.info("RESULT: Consistent with standard Gaussian Random Field theory.")

    logging.info(f"\nDiagnostic plot saved to: {output_plot}")

if __name__ == "__main__":
    run_minkowski_audit()