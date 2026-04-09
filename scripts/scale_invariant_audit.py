"""
Cross-Mission Scale Invariance Audit
Compares low-multipole data between independent satellite missions 
(Planck vs. WMAP) using normalized redundancy and direct pixel-to-pixel correlation.
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
def run_scale_audit():
    parser = argparse.ArgumentParser(description="Cross-Mission Scale Invariance Audit v3.0")
    parser.add_argument("--lmax", type=int, default=30, help="Maximum multipole for low-l analysis")
    parser.add_argument("--nside-clean", type=int, default=32, help="NSIDE for cleaned low-multipole maps")
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

    logging.info("CMB-AUDIT: Initializing Cross-Mission Scale Invariance Audit...")

    # Verification of datasets
    if not planck_path.exists() or not wmap_path.exists():
        logging.critical("ERROR: One or more datasets missing in the 'data' directory.")
        return

    # 2. Load Maps
    try:
        logging.info("Loading datasets (Planck PR3 & WMAP 9-Year)...")
        planck_map = hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False)
        wmap_map = hp.read_map(str(wmap_path), field=0, dtype=np.float64, verbose=False)
    except Exception as e:
        logging.exception(f"FAILED to process datasets: {e}")
        return

    # 3. Filtering and Normalization
    logging.info(f"Applying Harmonic Filter (Lmax={args.lmax}) and Z-Score Normalization...")

    def get_clean_map(m, lmax, nside_clean):
        """Extracts low-multipole components and normalizes the signal."""
        alm = hp.map2alm(m, lmax=lmax)
        clean_map = hp.alm2map(alm, nside=nside_clean, verbose=False)
        # CRITICAL: Normalize to zero mean and unit variance
        clean_map = (clean_map - np.mean(clean_map)) / np.std(clean_map)
        return clean_map

    clean_planck = get_clean_map(planck_map, args.lmax, args.nside_clean)
    clean_wmap = get_clean_map(wmap_map, args.lmax, args.nside_clean)

    # 4. Invariance Metrics
    def get_ratio(m):
        """Compression ratio (Kolmogorov complexity proxy)."""
        bytes_data = m.astype(np.float32).tobytes()
        return len(zlib.compress(bytes_data, level=9)) / len(bytes_data)

    r_planck = get_ratio(clean_planck)
    r_wmap = get_ratio(clean_wmap)

    # Direct Pixel Correlation (Pearson r)
    corr_coeff = np.corrcoef(clean_planck, clean_wmap)[0, 1]

    # 5. Save raw data for reproducibility
    np.save(results_dir / "clean_planck_v3.npy", clean_planck)
    np.save(results_dir / "clean_wmap_v3.npy", clean_wmap)

    # 6. Visualization
    logging.info("Generating mission comparison maps...")

    fig = plt.figure(figsize=(15, 5))
    hp.mollview(clean_planck, title="Planck (Low-l)", cmap="viridis", sub=(1, 3, 1), hold=False)
    hp.mollview(clean_wmap, title="WMAP (Low-l)", cmap="viridis", sub=(1, 3, 2), hold=False)
    # Difference map shows what doesn't match
    hp.mollview(clean_planck - clean_wmap, title="Residuals (Difference)", cmap="RdBu_r", sub=(1, 3, 3), hold=False)

    plt.suptitle(f"Cross-Mission Audit: Scale Invariance (L ≤ {args.lmax})", fontsize=14, y=1.05)

    output_plot = results_dir / "cross_mission_scale_invariance_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 7. Final Verdict
    logging.info("\n" + "="*65)
    logging.info("CROSS-MISSION INVARIANCE VERDICT")
    logging.info(f"PLANCK Redundancy Ratio : {r_planck:.6f}")
    logging.info(f"WMAP Redundancy Ratio : {r_wmap:.6f}")
    logging.info(f"Pixel Correlation (r) : {corr_coeff:.6f}")
    logging.info("="*65)

    if corr_coeff > 0.85:
        logging.info("VERDICT: MASTER STRUCTURE INVARIANCE CONFIRMED.")
        logging.info("The structural web is an objective reality, independent of the observer.")
    else:
        logging.info("RESULT: Partial divergence detected. Instrumental bias likely.")

    logging.info(f"\nAudit plot archived at: {output_plot}")
    logging.info(f"Raw data saved in: {results_dir}")

if __name__ == "__main__":
    run_scale_audit()