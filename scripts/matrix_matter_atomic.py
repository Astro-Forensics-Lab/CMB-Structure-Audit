"""
Vacuum-Matter Correlation Audit
Analyzes the internal correlation between the CMB temperature field
and its Laplacian (proxy for potential wells).
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse

def run_matter_sync_audit():
    parser = argparse.ArgumentParser(description="Vacuum-Matter Correlation Audit v3.1")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
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
    logging.info("CMB-AUDIT: Initializing Vacuum-Matter Internal Correlation Audit...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    # Load and condition map
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64)
        matrix_data = hp.ud_grade(cmb_map, args.nside)
        matrix_data = hp.remove_dipole(matrix_data, gal_cut=20.0)
    except Exception as e:
        logging.exception(f"FAILED to process map data: {e}")
        return

    # Apply spherical Laplacian
    logging.info("Applying spherical Laplacian operator (proxy for potential wells)...")
    lmax = 3 * args.nside - 1
    alm = hp.map2alm(matrix_data, lmax=lmax)
    ls, _ = hp.Alm.getlm(lmax)
    alm *= -ls * (ls + 1)
    matter_proxy = hp.alm2map(alm, args.nside)

    # Z-Score Normalization
    m1 = (matrix_data - np.mean(matrix_data)) / (np.std(matrix_data) + 1e-10)
    m2 = (matter_proxy - np.mean(matter_proxy)) / (np.std(matter_proxy) + 1e-10)

    # Correlation
    r_coeff, p_value = pearsonr(m1, m2)

    # Save data (convert to normal array to avoid MaskedArray error)
    np.save(results_dir / "cmb_field_v3.npy", np.asarray(matrix_data))
    np.save(results_dir / "laplacian_proxy_v3.npy", np.asarray(matter_proxy))

    # Visualization
    fig = plt.figure(figsize=(14, 7))
    hp.mollview(m1, title="Vacuum Potential Field (CMB)", cmap="viridis", sub=(1, 2, 1), hold=False)
    hp.mollview(m2, title="Laplacian Proxy (Curvature / Wells)", cmap="magma", sub=(1, 2, 2), hold=False)
    plt.suptitle("Vacuum-Matter Geometric Correlation Audit (Internal)", fontsize=14, y=1.02)

    output_plot = results_dir / "vacuum_matter_correlation_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Verdict
    logging.info("\n" + "="*65)
    logging.info("VACUUM–MATTER INTERNAL CORRELATION VERDICT")
    logging.info(f"Correlation Coefficient (r) : {r_coeff:.6f}")
    logging.info(f"Statistical Significance (p) : {p_value:.6e}")
    logging.info("="*65)
    logging.info("ANALYSIS: This measures how the primordial potential generates its own")
    logging.info("geometric 'scaffolding' for future matter distribution.")

    if abs(r_coeff) > 0.7:
        logging.info("VERDICT: VERY STRONG GEOMETRIC ALIGNMENT.")
    else:
        logging.info("RESULT: Weak or complex geometric coupling detected.")

    logging.info(f"\nDiagnostic plot archived at: {output_plot}")

if __name__ == "__main__":
    run_matter_sync_audit()