"""
Cross-Mission Scale Invariance Audit
Compares low-multipole data between independent satellite missions
(Planck vs. WMAP) using normalized redundancy and direct pixel-to-pixel correlation.
- Uses apodized Galactic mask to minimize edge artefacts.
- Z-Score normalization on unmasked pixels only.
- Low-l harmonic filtering (l ≤ lmax).
"""
import numpy as np
import healpy as hp
import zlib
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import argparse

def run_scale_audit():
    parser = argparse.ArgumentParser(description="Cross-Mission Scale Invariance Audit v3.3")
    parser.add_argument("--lmax", type=int, default=30, help="Maximum multipole for low-l analysis")
    parser.add_argument("--nside-clean", type=int, default=32, help="NSIDE for cleaned low-multipole maps")
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

    logging.info("Initializing Cross-Mission Scale Invariance Audit (v3.3 - Apodized Mask)...")

    # 1. Path Configuration
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    planck_path = data_dir / "planck_smica.fits"
    wmap_path = data_dir / "wmap_ilc_9yr_v5.fits"

    if not planck_path.exists() or not wmap_path.exists():
        logging.critical("CRITICAL ERROR: One or more datasets missing in the 'data' directory.")
        logging.info("Please run setup_project_infrastructure.py first.")
        return

    # 2. Load Maps
    try:
        logging.info("Loading datasets (Planck PR3 & WMAP 9-Year)...")
        planck_map = hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False)
        wmap_map = hp.read_map(str(wmap_path), field=0, dtype=np.float64, verbose=False)
    except Exception as e:
        logging.exception(f"FAILED to process datasets: {e}")
        return

    # 3. Apodized Galactic Mask (same mask for both missions)
    logging.info(f"Applying apodized Galactic Plane Mask (|b| > {args.gal_cut}° ± {args.apod_width}°)...")
    npix = hp.nside2npix(args.nside_clean)
    theta, _ = hp.pix2ang(args.nside_clean, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)

    hard_mask = np.abs(galactic_latitude) > args.gal_cut
    apod_mask = np.ones(npix, dtype=np.float64)
    transition = (np.abs(galactic_latitude) > args.gal_cut - args.apod_width) & \
                 (np.abs(galactic_latitude) <= args.gal_cut)
    apod_mask[transition] = 0.5 * (1 + np.cos(
        np.pi * (args.gal_cut - np.abs(galactic_latitude[transition])) / args.apod_width
    ))
    apod_mask[~hard_mask] = 0.0

    # 4. Filtering and Normalization
    logging.info(f"Applying Harmonic Filter (Lmax={args.lmax}) and Z-Score Normalization...")

    def get_clean_map(m, lmax, nside_clean, mask):
        """Extract low-multipole components on masked map and normalize."""
        # Apply mask before alm
        m_masked = m * mask
        alm = hp.map2alm(m_masked, lmax=lmax, use_pixel_weights=True)
        clean_map = hp.alm2map(alm, nside=nside_clean, verbose=False)
        # Z-Score on unmasked pixels only
        clean_pixels = clean_map[mask > 0.5]
        mean_val = np.mean(clean_pixels)
        std_val = np.std(clean_pixels)
        clean_map = (clean_map - mean_val) / (std_val + 1e-10)
        return clean_map * mask   # keep mask for later correlation

    # Resample both maps to clean NSIDE
    planck_resampled = hp.ud_grade(planck_map, args.nside_clean)
    wmap_resampled = hp.ud_grade(wmap_map, args.nside_clean)

    clean_planck = get_clean_map(planck_resampled, args.lmax, args.nside_clean, apod_mask)
    clean_wmap = get_clean_map(wmap_resampled, args.lmax, args.nside_clean, apod_mask)

    # 5. Invariance Metrics (only on clean sky pixels)
    def get_ratio(m, mask):
        """Compression ratio (Kolmogorov complexity proxy) on unmasked pixels."""
        clean_data = m[mask > 0.5].astype(np.float32)
        bytes_data = clean_data.tobytes()
        return len(zlib.compress(bytes_data, level=9)) / len(bytes_data)

    r_planck = get_ratio(clean_planck, apod_mask)
    r_wmap = get_ratio(clean_wmap, apod_mask)

    # Direct Pixel Correlation (Pearson r on clean sky)
    clean_mask = apod_mask > 0.5
    corr_coeff = np.corrcoef(clean_planck[clean_mask], clean_wmap[clean_mask])[0, 1]

    # 6. Save raw data for reproducibility
    np.save(results_dir / "clean_planck_v3.3.npy", clean_planck)
    np.save(results_dir / "clean_wmap_v3.3.npy", clean_wmap)

    # 7. Visualization
    logging.info("Generating mission comparison maps...")
    fig = plt.figure(figsize=(15, 5))
    hp.mollview(clean_planck, title="Planck (Low-l)", cmap="viridis", sub=(1, 3, 1), hold=False)
    hp.mollview(clean_wmap, title="WMAP (Low-l)", cmap="viridis", sub=(1, 3, 2), hold=False)
    hp.mollview(clean_planck - clean_wmap, title="Residuals (Difference)", cmap="RdBu_r", sub=(1, 3, 3), hold=False)
    plt.suptitle(f"Cross-Mission Audit: Scale Invariance (L ≤ {args.lmax})", fontsize=14, y=1.05)
    output_plot = results_dir / "cross_mission_scale_invariance_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close(fig)

    # 8. Final Verdict
    logging.info("\n" + "="*70)
    logging.info("CROSS-MISSION INVARIANCE VERDICT (v3.3)")
    logging.info(f"PLANCK Redundancy Ratio : {r_planck:.6f}")
    logging.info(f"WMAP Redundancy Ratio   : {r_wmap:.6f}")
    logging.info(f"Pixel Correlation (r)   : {corr_coeff:.6f}")
    logging.info("="*70)

    if corr_coeff > 0.85:
        logging.info("VERDICT: Strong scale invariance between Planck and WMAP.")
    elif corr_coeff > 0.70:
        logging.info("VERDICT: Good consistency between missions.")
    else:
        logging.info("RESULT: Moderate divergence detected - possible residual systematics.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")

if __name__ == "__main__":
    run_scale_audit()