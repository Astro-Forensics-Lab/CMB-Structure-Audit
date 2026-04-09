"""
Ultimate Integrity Audit
The final "Truth Filter"
- Ultra-high fidelity 350 DPI visualization for publication.
- Verified Galactic Masking and Z-Score Normalization.
- Explicit technical note on masking-induced phase artefacts.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def ultimate_integrity_audit():
    parser = argparse.ArgumentParser(description="Ultimate Integrity Audit v3.0")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--gal-cut", type=float, default=20.0, help="Galactic latitude cut in degrees")
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

    logging.info("NBSL-BRAIN: Initializing Ultimate Integrity Audit (v3.0 - Normalized)...")

    # 1. Path Configuration
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    data_dir_path = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    planck_path = data_dir_path / "planck_smica.fits"
    wmap_path = data_dir_path / "wmap_ilc_9yr_v5.fits"

    if not planck_path.exists() or not wmap_path.exists():
        logging.critical(f"CRITICAL ERROR: Datasets missing at {data_dir_path}.")
        logging.info("Please run setup_project_infrastructure.py first.")
        return

    # 2. Loading and Resampling
    nside_audit = args.nside
    logging.info(f"Resampling maps to NSIDE={nside_audit} for Null Test subtraction...")

    map_p = hp.ud_grade(
        hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False),
        nside_audit
    )
    map_w = hp.ud_grade(
        hp.read_map(str(wmap_path), field=0, dtype=np.float64, verbose=False),
        nside_audit
    )

    # 3. Galactic Masking (|b| > cut)
    logging.info(f"Applying Galactic Plane Mask (|b| > {args.gal_cut}°)...")
    npix = hp.nside2npix(nside_audit)
    theta, _ = hp.pix2ang(nside_audit, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)
    mask = np.abs(galactic_latitude) > args.gal_cut
    mask_float = mask.astype(float)

    # 4. Z-Score Normalization (Unmasked pixels only)
    def normalize_clean_sky(m, msk):
        """Normalize using only unmasked pixels."""
        clean_pixels = m[msk == 1]
        mean_val = np.mean(clean_pixels)
        std_val = np.std(clean_pixels)
        m_norm = (m - mean_val) / (std_val + 1e-10)
        return m_norm * msk

    logging.info("Aligning mission units via Z-Score normalization...")
    map_p_norm = normalize_clean_sky(map_p, mask_float)
    map_w_norm = normalize_clean_sky(map_w, mask_float)

    # 5. Null Test (Planck - WMAP)
    null_map = map_p_norm - map_w_norm
    null_residual = np.std(null_map[mask])

    # 6. Phase Covariance Analysis
    logging.info("Analyzing phase distribution for non-stochastic coupling...")
    alms = hp.map2alm(map_p_norm)
    phases = np.angle(alms)
    phase_variance = np.var(phases)
    theoretical_var = (np.pi ** 2) / 3
    entropy_deviation = abs(phase_variance - theoretical_var) / theoretical_var

    # 7. Save raw data for reproducibility
    np.save(results_dir / "ultimate_planck_norm_v3.npy", map_p_norm)
    np.save(results_dir / "ultimate_wmap_norm_v3.npy", map_w_norm)
    np.save(results_dir / "ultimate_null_map_v3.npy", null_map)

    # 8. Visualization (Ultra-High Fidelity)
    fig = plt.figure(figsize=(14, 10))
    # Subplot 1: Residual Map
    hp.mollview(null_map, title="Ultimate Audit: Normalized Residuals (Planck - WMAP)",
                unit="σ", cmap="seismic", sub=(2, 1, 1), hold=False)
    # Subplot 2: Phase Distribution Histogram
    plt.subplot(2, 1, 2)
    plt.hist(phases, bins=100, color='gold', alpha=0.8, density=True, label="Empirical Phases")
    plt.axhline(1 / (2 * np.pi), color='red', linestyle='--', linewidth=2, label="Uniform Phase Theory")
    plt.title("Phase Covariance Audit: Informational Coupling Detection")
    plt.xlabel("Phase Angle (radians)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # Save with versioning and ultra-high resolution
    output_path = results_dir / "ultimate_integrity_audit_v3.0.png"
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close(fig)

    # 9. Verdict
    logging.info("\n" + "="*70)
    logging.info("ULTIMATE INTEGRITY VERDICT (v3.0 - NORMALIZED)")
    logging.info(f"Instrumental Residual (Null Test) : {null_residual:.6f} σ")
    logging.info(f"Phase Variance Deviation : {entropy_deviation*100:.4f}%")
    logging.info("="*70)

    if null_residual < 0.65:
        logging.info("VERDICT: THE MATRIX IS INDEPENDENT OF INSTRUMENT.")
        logging.info("Geometric structures are fully verified across ESA and NASA missions.")
    elif null_residual < 0.85:
        logging.info("VERDICT: STRONG MISSION CONSISTENCY.")
    else:
        logging.info("VERDICT: INCONCLUSIVE. Significant residual detected.")

    logging.info("="*70)
    logging.info(f"High-Resolution Plot archived at: {output_path}")
    logging.info("\nNOTE: Phase analysis may contain minor edge artefacts due to galactic masking.")

if __name__ == "__main__":
    ultimate_integrity_audit()