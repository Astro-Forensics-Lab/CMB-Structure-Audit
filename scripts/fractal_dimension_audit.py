"""
Geometric Scaling Audit
Minkowski-Bouligand dimension using multi-scale box-counting.
- Uses apodized Galactic mask to minimize edge artefacts.
- 2D Cartesian projection (approximation - polar distortions still present).
- Box-counting on normalized binary map.
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
    parser = argparse.ArgumentParser(description="Geometric Scaling Audit v3.3")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--xsize", type=int, default=2048, help="Cartesian projection width in pixels (higher = less pixelation)")
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

    logging.info("Initializing Geometric Scaling Audit (Fractal Dimension v3.3 - Apodized Mask)...")

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

    # Apply mask (set masked pixels to mean of unmasked for cleaner projection)
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

    # 5. 2D Projection (Cartesian - still an approximation)
    logging.info(f"Projecting spherical map to 2D Cartesian plane ({args.xsize} px)...")
    proj_map = hp.cartview(data_norm, xsize=args.xsize, return_projected_map=True)
    plt.close()
    proj_map = np.nan_to_num(proj_map, nan=0.0)

    # 6. Binarization (using median threshold)
    binary = (proj_map > np.median(proj_map)).astype(np.int32)
    # CRITICAL FIX: Force regular ndarray (hp.cartview can return MaskedArray)
    binary = np.asarray(binary, dtype=np.int32)

    # Box-counting function
    def box_count(img, k):
        """Count occupied boxes at scale k."""
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1
        )
        return np.sum(S > 0)

    # Multi-Scale Measurement
    scales = 2**np.arange(3, 10)
    counts = []
    logging.info("Performing multi-scale box-counting...")

    for s in scales:
        try:
            c = box_count(binary, s)
            counts.append(c)
        except Exception as e:
            logging.error(f"Box-counting failed at scale {s}: {e}")
            continue

    counts = np.array(counts)

    # Linear Regression for fractal dimension
    valid = counts > 0
    if np.sum(valid) < 2:
        logging.critical("Not enough valid boxes for regression.")
        return

    coeffs = np.polyfit(np.log(scales[valid]), np.log(counts[valid]), 1)
    df = -coeffs[0]

    # Save data for reproducibility
    np.save(results_dir / "fractal_scales_v3.3.npy", scales)
    np.save(results_dir / "fractal_counts_v3.3.npy", counts)
    np.save(results_dir / "fractal_binary_map_v3.3.npy", binary)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.loglog(scales, counts, 'o-', color='orange', label=f'Calculated D = {df:.4f}')
    plt.plot(scales, np.exp(np.polyval(coeffs, np.log(scales))), 'k--', alpha=0.4, label='Linear Regression Fit')
    plt.title("Fractal Audit v3.3: Scale-Invariance and Structural Complexity")
    plt.xlabel("Box Size (Log Scale)")
    plt.ylabel("Number of Occupied Boxes (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)
    output_plot = results_dir / "fractal_dimension_audit_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close()

    # Verdict
    logging.info("\n" + "="*70)
    logging.info("FRACTAL DIMENSION VERDICT (v3.3)")
    logging.info(f"Detected Fractal Dimension (D): {df:.4f}")
    logging.info("="*70)

    if 1.7 < df < 2.1:
        logging.info("VERDICT: High-order scale-invariance consistent with expected CMB topology.")
    elif 1.4 < df <= 1.7:
        logging.info("VERDICT: Moderate structural complexity detected.")
    else:
        logging.info("RESULT: Inconclusive - further investigation recommended.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")
    logging.info("NOTE: This uses 2D Cartesian projection (approximation). Polar regions may still contain minor distortions.")

if __name__ == "__main__":
    run_fractal_audit()