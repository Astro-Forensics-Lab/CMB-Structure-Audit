"""
Vacuum-Matter Correlation Audit
Analyzes the internal correlation between the CMB temperature field
and its Laplacian (proxy for potential wells / curvature).
- Uses apodized Galactic mask to minimize edge artefacts.
- Z-Score normalization on unmasked pixels only.
- Pearson correlation computed only on clean sky.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import logging
import sys
import argparse

def run_matter_sync_audit():
    parser = argparse.ArgumentParser(description="Vacuum-Matter Correlation Audit v3.3")
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

    logging.info("Initializing Vacuum-Matter Correlation Audit (v3.3 - Apodized Mask)...")

    # 1. Path Configuration
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    planck_path = data_dir / "planck_smica.fits"

    if not planck_path.exists():
        logging.critical(f"CRITICAL ERROR: Data file not found at {planck_path}")
        logging.info("Please run setup_project_infrastructure.py first.")
        return

    # 2. Load and condition dataset
    try:
        logging.info(f"Loading and resampling map to NSIDE={args.nside}...")
        cmb_map = hp.read_map(str(planck_path), field=0, dtype=np.float64, verbose=False)
        data = hp.ud_grade(cmb_map, args.nside)
    except Exception as e:
        logging.exception(f"FAILED to process FITS data: {e}")
        return

    # 3. Apodized Galactic Mask
    logging.info(f"Applying apodized Galactic Plane Mask (|b| > {args.gal_cut}° ± {args.apod_width}°)...")
    npix = hp.nside2npix(args.nside)
    theta, _ = hp.pix2ang(args.nside, np.arange(npix))
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

    # Apply mask (set masked pixels to mean of unmasked)
    clean_pixels = data[apod_mask > 0.5]
    data = data * apod_mask + np.mean(clean_pixels) * (1 - apod_mask)

    # 4. Z-Score Normalization (unmasked pixels only)
    def normalize_clean_sky(m, msk):
        clean_pixels = m[msk > 0.5]
        mean_val = np.mean(clean_pixels)
        std_val = np.std(clean_pixels)
        m_norm = (m - mean_val) / (std_val + 1e-10)
        return m_norm * msk

    data_norm = normalize_clean_sky(data, apod_mask)

    # 5. Remove dipole (on masked map)
    logging.info("Removing dipole component...")
    data_dipole_free = hp.remove_dipole(data_norm, gal_cut=args.gal_cut)

    # 6. Spherical Laplacian (proxy for potential wells)
    logging.info("Applying spherical Laplacian operator (proxy for potential wells)...")
    lmax = 3 * args.nside - 1
    alm = hp.map2alm(data_dipole_free, lmax=lmax, use_pixel_weights=True)
    ls, _ = hp.Alm.getlm(lmax)
    alm *= -ls * (ls + 1)          # Laplacian in harmonic space
    matter_proxy = hp.alm2map(alm, args.nside)

    # 7. Z-Score on Laplacian proxy (unmasked pixels)
    matter_norm = normalize_clean_sky(matter_proxy, apod_mask)

    # 8. Pearson Correlation (only on clean sky pixels)
    clean_mask = apod_mask > 0.5
    r_coeff, p_value = pearsonr(data_norm[clean_mask], matter_norm[clean_mask])

    # Save data for reproducibility
    np.save(results_dir / "vacuum_field_v3.3.npy", data_norm)
    np.save(results_dir / "laplacian_proxy_v3.3.npy", matter_norm)

    # 9. Visualization
    fig = plt.figure(figsize=(14, 7))
    hp.mollview(data_norm, title="CMB Temperature Field (Normalized)", cmap="viridis", sub=(1, 2, 1), hold=False)
    hp.mollview(matter_norm, title="Laplacian Proxy (Potential Wells)", cmap="magma", sub=(1, 2, 2), hold=False)
    plt.suptitle("Vacuum-Matter Correlation Audit (Internal)", fontsize=14, y=1.02)
    output_plot = results_dir / "vacuum_matter_correlation_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close(fig)

    # 10. Verdict
    logging.info("\n" + "="*70)
    logging.info("VACUUM–MATTER CORRELATION VERDICT (v3.3)")
    logging.info(f"Correlation Coefficient (r)      : {r_coeff:.6f}")
    logging.info(f"Statistical Significance (p-value): {p_value:.6e}")
    logging.info("="*70)

    if abs(r_coeff) > 0.7:
        logging.info("VERDICT: Strong internal geometric correlation detected.")
    elif abs(r_coeff) > 0.4:
        logging.info("VERDICT: Moderate correlation between temperature and curvature.")
    else:
        logging.info("RESULT: Weak or no significant internal correlation.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")

if __name__ == "__main__":
    run_matter_sync_audit()