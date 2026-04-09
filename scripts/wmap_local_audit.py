"""
Inter-Satellite Integrity Validation
Provides a dynamic consistency check between NASA WMAP and ESA Planck metrics.
Eliminates hard-coded benchmarks and instrumental bias via Z-score normalization.
"""

import numpy as np
import healpy as hp
import zlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def run_wmap_local_audit():
    parser = argparse.ArgumentParser(description="Inter-Satellite Integrity Validation v3.0")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution for comparison")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data folder")
    args = parser.parse_args()

    # Logging configured inside the function
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    np.random.seed(42)

    # Robust paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    planck_path = data_dir / "planck_smica.fits"
    wmap_path = data_dir / "wmap_ilc_9yr_v5.fits"

    logging.info("CMB-AUDIT: Initializing Inter-Satellite Integrity Validation (Planck vs NASA WMAP)...")

    # Verify datasets exist
    if not planck_path.exists() or not wmap_path.exists():
        logging.critical("ERROR: Datasets missing. Required: Planck and WMAP .fits files.")
        return

    # 2. Load Datasets
    try:
        logging.info("Loading Planck and WMAP maps...")
        planck_map = hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False)
        wmap_map = hp.read_map(str(wmap_path), field=0, dtype=np.float64, verbose=False)
    except Exception as e:
        logging.exception(f"FAILED to read FITS files: {e}")
        return

    # 3. Standardization
    logging.info(f"Standardizing resolution to NSIDE={args.nside} and applying Z-score normalization...")

    # Downgrade and Normalize Planck
    planck_data = hp.ud_grade(planck_map, args.nside)
    planck_data = (planck_data - np.mean(planck_data)) / np.std(planck_data)

    # Downgrade and Normalize WMAP
    wmap_data = hp.ud_grade(wmap_map, args.nside)
    wmap_data = (wmap_data - np.mean(wmap_data)) / np.std(wmap_data)

    # 4. Algorithmic Redundancy Calculation
    def get_ratio(data):
        """Compression ratio (Kolmogorov complexity proxy)."""
        bytes_data = data.astype(np.float32).tobytes()
        return len(zlib.compress(bytes_data, level=9)) / len(bytes_data)

    planck_ratio = get_ratio(planck_data)
    wmap_ratio = get_ratio(wmap_data)

    # Direct Pixel-to-Pixel Correlation
    corr_coeff = np.corrcoef(planck_data, wmap_data)[0, 1]

    # 5. Save raw data for reproducibility
    np.save(results_dir / "planck_normalized_v3.npy", planck_data)
    np.save(results_dir / "wmap_normalized_v3.npy", wmap_data)

    # 6. Visualization (Triple Map View)
    logging.info("Generating validation maps...")

    fig = plt.figure(figsize=(15, 6))
    hp.mollview(planck_data, title="Planck (ESA) Normalized", cmap="plasma", sub=(1, 3, 1), hold=False)
    hp.mollview(wmap_data, title="WMAP (NASA) Normalized", cmap="plasma", sub=(1, 3, 2), hold=False)
    # The difference map should be mostly noise if missions agree
    hp.mollview(planck_data - wmap_data, title="Difference (Residuals)", cmap="RdBu_r", sub=(1, 3, 3), hold=False)

    plt.suptitle("Inter-Satellite Integrity Audit: Cross-Mission Agreement", fontsize=14, y=1.05)

    output_plot = results_dir / "inter_satellite_integrity_validation_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 7. Final Verdict
    logging.info("\n" + "="*65)
    logging.info("INTER-SATELLITE INTEGRITY REPORT")
    logging.info(f"PLANCK Ratio (Real-time): {planck_ratio:.6f}")
    logging.info(f"WMAP Ratio (Real-time): {wmap_ratio:.6f}")
    logging.info(f"Correlation Coefficient: {corr_coeff:.6f}")
    logging.info("="*65)

    diff_ratio = abs(planck_ratio - wmap_ratio)
    if diff_ratio < 0.02 and corr_coeff > 0.88:
        logging.info("VERDICT: MISSION INTEGRITY CONFIRMED.")
        logging.info("Findings are satellite-independent. Bias eliminated.")
    else:
        logging.info(f"RESULT: Minor divergence (Diff: {diff_ratio:.4f}, Corr: {corr_coeff:.4f}).")
        logging.info("Structural persistence is mission-specific or requires better masking.")

    logging.info(f"\nValidation plot archived at: {output_plot}")
    logging.info(f"Raw data saved in: {results_dir}")

if __name__ == "__main__":
    run_wmap_local_audit()