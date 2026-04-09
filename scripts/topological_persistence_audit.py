"""
Persistent Homology Audit (Betti-0 Approximation)
Maps the stability of connected topological components (Betti-0)
using 2D Cartesian projection for connectivity mapping.
- Uses apodized Galactic mask to minimize edge artefacts.
- Note: This is a simplified 2D approximation of Betti-0 (not full persistent homology on sphere).
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.ndimage import label
from pathlib import Path
import logging
import sys
import argparse

def run_topology_audit():
    parser = argparse.ArgumentParser(description="Persistent Homology Audit v3.3")
    parser.add_argument("--nside", type=int, default=64, help="NSIDE resolution")
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

    logging.info("Initializing Persistent Homology Audit (Betti-0 v3.3 - Apodized Mask)...")

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

    # Apply mask
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

    # 6. Filtration Analysis (Betti-0)
    thresholds = np.linspace(-3.0, 3.0, 60)
    persistence = []
    logging.info("Mapping topological persistence (Betti-0) across 60 filtration levels...")

    for t in thresholds:
        binary_map = proj_map > t
        _, num_features = label(binary_map)
        persistence.append(num_features)

    persistence = np.array(persistence)

    # Theoretical Gaussian Baseline
    theoretical_baseline = (1 - thresholds**2) * np.exp(-thresholds**2 / 2)
    scale_factor = np.max(persistence) / np.max(np.abs(theoretical_baseline))
    theoretical_baseline *= scale_factor

    # Save raw data
    np.save(results_dir / "topology_thresholds_v3.3.npy", thresholds)
    np.save(results_dir / "topology_persistence_v3.3.npy", persistence)
    np.save(results_dir / "topology_theoretical_v3.3.npy", theoretical_baseline)
    np.save(results_dir / "topology_projected_map_v3.3.npy", proj_map)

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, persistence, color='lime', linewidth=2.2, label='Empirical Persistence (CMB)')
    plt.plot(thresholds, theoretical_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
             label='Stochastic Baseline (Gaussian Random Field)')
    plt.title("Persistent Homology Audit v3.3: Structural Stability (Betti-0)")
    plt.xlabel("Energy Threshold (σ)")
    plt.ylabel("Number of Connected Components (Betti-0)")
    plt.legend()
    plt.grid(axis='both', alpha=0.1)
    output_plot = results_dir / "topology_persistence_audit_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close()

    # 8. Verdict
    area_actual = np.trapezoid(persistence)
    area_baseline = np.trapezoid(theoretical_baseline)
    persistence_ratio = area_actual / area_baseline

    logging.info("\n" + "="*70)
    logging.info("PERSISTENT HOMOLOGY VERDICT (v3.3)")
    logging.info(f"Structural Persistence Index: {persistence_ratio:.4f}")
    logging.info("="*70)

    if persistence_ratio > 1.25:
        logging.info("VERDICT: Significant deviation from Gaussian expectations.")
    else:
        logging.info("RESULT: Consistent with Gaussian Random Field expectations.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")
    logging.info("NOTE: This is a simplified 2D Betti-0 approximation.")

if __name__ == "__main__":
    run_topology_audit()