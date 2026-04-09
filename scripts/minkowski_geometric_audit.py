"""
Minkowski Geometric Audit
Euler Characteristic across energy thresholds.
- Uses apodized Galactic mask to minimize edge artefacts.
- 2D Cartesian projection (approximation - polar distortions still present).
- 8-connectivity labeling on projected map.
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
    parser = argparse.ArgumentParser(description="Minkowski Geometric Audit v3.3")
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

    logging.info("Initializing Minkowski Geometric Audit (v3.3 - Apodized Mask)...")

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
        logging.exception(f"FAILED to process map data: {e}")
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

    # Apply mask to data
    clean_pixels = data[apod_mask > 0.5]
    data = data * apod_mask + np.mean(clean_pixels) * (1 - apod_mask)

    # 4. Z-Score Normalization
    def normalize_clean_sky(m, msk):
        clean_pixels = m[msk > 0.5]
        mean_val = np.mean(clean_pixels)
        std_val = np.std(clean_pixels)
        m_norm = (m - mean_val) / (std_val + 1e-10)
        return m_norm * msk

    data_norm = normalize_clean_sky(data, apod_mask)

    # 5. 2D Projection
    logging.info(f"Projecting spherical map to 2D Cartesian plane ({args.xsize} px)...")
    proj_map = hp.cartview(data_norm, xsize=args.xsize, return_projected_map=True)
    plt.close()
    proj_map = np.nan_to_num(proj_map, nan=0.0)
    # CRITICAL FIX: Force regular ndarray (hp.cartview returns MaskedArray)
    proj_map = np.asarray(proj_map, dtype=np.float64)

    # 6. Topological calculation (Euler Characteristic)
    thresholds = np.linspace(-3.0, 3.0, 60)
    euler_chars = []
    conn_8 = np.ones((3, 3))  # 8-connectivity
    logging.info("Calculating Euler Characteristic using 8-connectivity on projected map...")

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
    np.save(results_dir / "minkowski_thresholds_v3.3.npy", thresholds)
    np.save(results_dir / "minkowski_euler_chars_v3.3.npy", euler_chars)
    np.save(results_dir / "minkowski_theoretical_v3.3.npy", theoretical)
    np.save(results_dir / "minkowski_projected_map_v3.3.npy", proj_map)

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, euler_chars, 'o-', color='cyan', linewidth=2, label='Empirical Topology (CMB)')
    plt.plot(thresholds, theoretical, '--', color='red', linewidth=2, alpha=0.7,
             label='Gaussian Random Field (Adler Formula)')
    plt.title("Minkowski Audit v3.3: Euler Characteristic and Topological Consistency")
    plt.xlabel("Temperature Threshold (σ)")
    plt.ylabel("Euler Characteristic (χ)")
    plt.legend()
    plt.grid(True, alpha=0.1)
    output_plot = results_dir / "minkowski_topology_audit_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close()

    # 8. Deviation analysis
    diff = np.abs(euler_chars - theoretical)
    max_deviation = np.max(diff) / np.max(np.abs(euler_chars))

    logging.info("\n" + "="*70)
    logging.info("MINKOWSKI TOPOLOGICAL VERDICT (v3.3)")
    logging.info(f"Maximum relative deviation: {max_deviation:.4%}")
    logging.info("="*70)

    if max_deviation > 0.15:
        logging.info("VERDICT: Significant topological anomaly detected.")
    else:
        logging.info("RESULT: Consistent with standard Gaussian Random Field theory.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")
    logging.info("NOTE: This uses 2D Cartesian projection (approximation). Polar regions may still contain minor distortions.")

if __name__ == "__main__":
    run_minkowski_audit()