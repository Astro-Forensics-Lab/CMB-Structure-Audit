"""
Geometric Scaling Audit
Minkowski-Bouligand dimension using multi-scale box-counting.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def run_fractal_audit():
    parser = argparse.ArgumentParser(description="Geometric Scaling Audit v3.2")
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
    logging.info("CMB-AUDIT: Initializing Fractal Dimension Analysis (Box-Counting Method)...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    # Load map
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64)
    except Exception as e:
        logging.exception(f"FAILED to load FITS file: {e}")
        return

    # Cartesian Projection
    logging.info(f"Projecting spherical data into Cartesian plane ({args.xsize}px)...")
    data = hp.cartview(cmb_map, xsize=args.xsize, return_projected_map=True)
    plt.close()

    # Binarization
    data = np.nan_to_num(data, nan=0.0)
    binary = (data > np.median(data)).astype(np.int32)

    def box_count(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1
        )
        return np.sum(S > 0)

    # Multi-Scale Measurement
    scales = 2**np.arange(3, 10)
    counts = []
    for s in scales:
        try:
            c = box_count(binary, s)
            counts.append(c)
        except Exception as e:
            logging.error(f"Box-counting failed at scale {s}: {e}")
            continue
    counts = np.array(counts)

    # Linear Regression
    valid = counts > 0
    if np.sum(valid) < 2:
        logging.critical("Not enough valid boxes for regression.")
        return

    coeffs = np.polyfit(np.log(scales[valid]), np.log(counts[valid]), 1)
    df = -coeffs[0]

    # Save data (convert to normal array to avoid MaskedArray error)
    np.save(results_dir / "fractal_scales_v3.npy", scales)
    np.save(results_dir / "fractal_counts_v3.npy", counts)
    np.save(results_dir / "binary_map_v3.npy", np.asarray(binary))

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.loglog(scales, counts, 'o-', color='orange', label=f'Calculated D = {df:.4f}')
    plt.plot(scales, np.exp(np.polyval(coeffs, np.log(scales))), 'k--', alpha=0.4, label='Linear Regression Fit')

    plt.title("Fractal Audit: Scale-Invariance and Structural Complexity")
    plt.xlabel("Box Size (Log Scale)")
    plt.ylabel("Number of Occupied Boxes (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    output_plot = results_dir / "fractal_dimension_audit_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("\n" + "="*60)
    logging.info(f"DETECTED FRACTAL DIMENSION (D): {df:.4f}")
    logging.info("="*60)

    if 1.7 < df < 2.1:
        logging.info("VERDICT: HIGH-ORDER SCALE-INVARIANCE CONFIRMED.")
    elif 1.4 < df <= 1.7:
        logging.info("VERDICT: MODERATE STRUCTURAL COMPLEXITY.")
    else:
        logging.info("RESULT: Inconclusive structural recurrence.")

    logging.info(f"\nAudit plot archived at: {output_plot}")

if __name__ == "__main__":
    run_fractal_audit()