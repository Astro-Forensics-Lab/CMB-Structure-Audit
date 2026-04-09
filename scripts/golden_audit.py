"""
Phase Synchronization Audit
Investigates harmonic coupling between multipole scales.
- Uses apodized Galactic mask to minimize edge artefacts in alm computation.
- Z-Score normalization on unmasked pixels only.
- Monte Carlo significance testing (exploratory analysis).
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def run_golden_audit():
    parser = argparse.ArgumentParser(description="Phase Synchronization Audit v3.3")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--lmax", type=int, default=1000, help="Maximum multipole for map2alm")
    parser.add_argument("--l-base", type=int, default=50, help="Base multipole for scale analysis")
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

    logging.info("Initializing Phase Synchronization Audit (v3.3 - Apodized Mask)...")

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

    # Apply mask (set masked pixels to mean of unmasked for cleaner alm)
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

    # 5. Extract Spherical Harmonics (properly masked map)
    logging.info(f"Computing Spherical Harmonic Coefficients (alm) up to lmax={args.lmax}...")
    try:
        alm = hp.map2alm(data_norm, lmax=args.lmax, use_pixel_weights=True)
    except Exception as e:
        logging.exception(f"FAILED to compute spherical harmonics: {e}")
        return

    # 6. Golden Ratio Scales (Phi) - exploratory analysis
    phi = (1 + 5**0.5) / 2
    l1 = args.l_base
    l2 = int(l1 * phi)
    l3 = int(l1 * phi**2)
    logging.info(f"Testing phase relations at multipoles: l = {l1}, {l2}, {l3} (Phi-based)")

    # 7. Phase Extraction (m=0 only)
    def get_phase(l_val, m=0):
        idx = hp.Alm.getidx(args.lmax, l_val, m)
        return np.angle(alm[idx])

    phase1 = get_phase(l1)
    phase2 = get_phase(l2)
    phase3 = get_phase(l3)

    # Golden Phase Difference
    golden_diff = (phase1 + phase2 - phase3) % (2 * np.pi)
    if golden_diff > np.pi:
        golden_diff -= 2 * np.pi

    # 8. MONTE CARLO SIGNIFICANCE TEST (null hypothesis: random phases)
    logging.info("Running Monte Carlo simulation (1000 random universes) for P-Value...")
    n_sim = 1000
    random_diffs = []
    for _ in range(n_sim):
        r_ph1, r_ph2, r_ph3 = np.random.uniform(-np.pi, np.pi, 3)
        diff = (r_ph1 + r_ph2 - r_ph3) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        random_diffs.append(abs(diff))

    p_value = np.mean(np.array(random_diffs) <= abs(golden_diff))

    # Save raw data for reproducibility
    np.save(results_dir / "golden_alm_v3.3.npy", alm)
    np.save(results_dir / "golden_phases_v3.3.npy", np.array([phase1, phase2, phase3]))
    np.save(results_dir / "golden_diff_v3.3.npy", golden_diff)
    np.save(results_dir / "golden_mc_random_diffs_v3.3.npy", random_diffs)

    # 9. Visualization
    all_phases = np.angle(alm)
    plt.figure(figsize=(10, 6))
    plt.hist(all_phases, bins=100, color='gold', alpha=0.7, density=True, label='Empirical CMB Phases (All)')
    plt.axvline(golden_diff, color='red', linestyle='--', linewidth=2,
                label=f'Phi-coupling Error: {abs(golden_diff):.4f} rad')
    plt.title("Phase Synchronization Audit v3.3: Exploratory Phi-Based Analysis")
    plt.xlabel("Phase Angle (radians)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    output_plot = results_dir / "golden_ratio_resonance_v3.3.png"
    plt.savefig(output_plot, dpi=350, bbox_inches='tight')
    plt.close()

    # 10. FINAL VERDICT
    logging.info("\n" + "="*70)
    logging.info("PHASE SYNCHRONIZATION VERDICT (v3.3)")
    logging.info(f"Phase Synchronization Error : {abs(golden_diff):.6f} rad")
    logging.info(f"P-Value (Monte Carlo)       : {p_value:.6f}")
    logging.info("="*70)

    if p_value < 0.05:
        logging.info("VERDICT: Statistically significant deviation detected (p < 0.05).")
    else:
        logging.info("RESULT: No statistically significant Phi-based synchronization.")
        logging.info("Phase distribution is consistent with random expectation.")

    logging.info(f"\nHigh-resolution diagnostic plot saved to: {output_plot}")
    logging.info("NOTE: This analysis is exploratory. Golden-ratio multipole scaling is not part of standard CMB theory.")

if __name__ == "__main__":
    run_golden_audit()