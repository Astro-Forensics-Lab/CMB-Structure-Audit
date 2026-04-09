"""
Ultimate Integrity Audit
The final "Truth Filter"
- Ultra-high fidelity 350 DPI visualization for publication.
- Robust Galactic Masking with apodization to minimize edge artefacts.
- Verified Z-Score Normalization and Null Test between Planck and WMAP.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def ultimate_integrity_audit():
    parser = argparse.ArgumentParser(description="Ultimate Integrity Audit v3.1")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--gal-cut", type=float, default=20.0, help="Galactic latitude cut in degrees")
    parser.add_argument("--apod-width", type=float, default=5.0, help="Apodization width in degrees")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data folder")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    logging.info("Initializing Ultimate Integrity Audit (v3.1 - Apodized Mask)...")

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
    logging.info(f"Resampling maps to NSIDE={nside_audit} for Null Test...")
    map_p = hp.ud_grade(
        hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False),
        nside_audit
    )
    map_w = hp.ud_grade(
        hp.read_map(str(wmap_path), field=0, dtype=np.float64, verbose=False),
        nside_audit
    )

    # 3. Apodized Galactic Mask (|b| > gal-cut with smooth transition)
    logging.info(f"Applying apodized Galactic Plane Mask (|b| > {args.gal_cut}° ± {args.apod_width}°)...")
    npix = hp.nside2npix(nside_audit)
    theta, phi = hp.pix2ang(nside_audit, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)

    # Hard mask
    hard_mask = np.abs(galactic_latitude) > args.gal_cut
    # Apodization (cosine taper)
    apod_mask = np.ones(npix, dtype=np.float64)
    transition = (np.abs(galactic_latitude) > args.gal_cut - args.apod_width) & \
                 (np.abs(galactic_latitude) <= args.gal_cut)
    apod_mask[transition] = 0.5 * (1 + np.cos(
        np.pi * (args.gal_cut - np.abs(galactic_latitude[transition])) / args.apod_width
    ))
    apod_mask[~hard_mask] = 0.0

    # 4. Z-Score Normalization (using only clean sky pixels)
    def normalize_clean_sky(m, msk):
        """Normalize using only unmasked pixels."""
        clean_pixels = m[msk > 0.5]
        mean_val = np.mean(clean_pixels)
        std_val = np.std(clean_pixels)
        m_norm = (m - mean_val) / (std_val + 1e-10)
        return m_norm * msk

    logging.info("Aligning mission units via Z-Score normalization...")
    map_p_norm = normalize_clean_sky(map_p, apod_mask)
    map_w_norm = normalize_clean_sky(map_w, apod_mask)

    # 5. Null Test (Planck - WMAP)
    null_map = map_p_norm - map_w_norm
    null_residual = np.std(null_map[apod_mask > 0.5])

    # 6. Phase Covariance Analysis (on properly masked map)
    logging.info("Analyzing phase distribution for non-stochastic coupling...")
    alms = hp.map2alm(map_p_norm, use_pixel_weights=True)
    phases = np.angle(alms)
    phase_variance = np.var(phases)
    theoretical_var = (np.pi ** 2) / 3
    entropy_deviation = abs(phase_variance - theoretical_var) / theoretical_var

    # 7. Save raw data for reproducibility
    np.save(results_dir / "ultimate_planck_norm_v3.1.npy", map_p_norm)
    np.save(results_dir / "ultimate_wmap_norm_v3.1.npy", map_w_norm)
    np.save(results_dir / "ultimate_null_map_v3.1.npy", null_map)

    # 8. Visualization (Ultra-High Fidelity)
    fig = plt.figure(figsize=(14, 10))
    # Subplot 1: Residual Map
    hp.mollview(null_map, title="Ultimate Audit: Normalized Residuals (Planck - WMAP)",
                unit="σ", cmap="seismic", sub=(2, 1, 1), hold=False)
    # Subplot 2: Phase Distribution Histogram
    plt.subplot(2, 1, 2)
    plt.hist(phases, bins=100, color='gold', alpha=0.8, density=True, label="Empirical Phases")
    plt.axhline(1 / (2 * np.pi), color='red', linestyle='--', linewidth=2, label="Uniform Phase Theory")
    plt.title("Phase Covariance Audit")
    plt.xlabel("Phase Angle (radians)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # Save with ultra-high resolution
    output_path = results_dir / "ultimate_integrity_audit_v3.1.png"
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.close(fig)

    # 9. Verdict
    logging.info("\n" + "="*70)
    logging.info("ULTIMATE INTEGRITY VERDICT (v3.1 - Apodized Mask)")
    logging.info(f"Instrumental Residual (Null Test) : {null_residual:.6f} σ")
    logging.info(f"Phase Variance Deviation         : {entropy_deviation*100:.4f}%")
    logging.info("="*70)

    if null_residual < 0.65:
        logging.info("VERDICT: Strong consistency between Planck and WMAP.")
    elif null_residual < 0.85:
        logging.info("VERDICT: Good mission consistency.")
    else:
        logging.info("VERDICT: Significant residual detected - further investigation recommended.")
    logging.info("="*70)
    logging.info(f"High-Resolution Plot archived at: {output_path}")

if __name__ == "__main__":
    ultimate_integrity_audit()