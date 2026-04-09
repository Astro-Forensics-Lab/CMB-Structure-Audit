"""
Fortress Monte Carlo Audit
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse
from scipy.stats import kurtosis

def run_fortress_monte_carlo():
    parser = argparse.ArgumentParser(description="Fortress Monte Carlo Audit v4.0")
    parser.add_argument("--n-sim", type=int, default=1000, help="Number of LCDM simulations")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--gal-cut", type=float, default=20.0, help="Galactic latitude cut in degrees")
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
    base_dir = script_dir.parent.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    planck_path = data_dir / "planck_smica.fits"
    cl_txt_path = data_dir / "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

    logging.info("Initializing Fortress Monte Carlo Audit v4.0...")

    if not planck_path.exists():
        logging.critical(f"ERROR: Planck map not found at {planck_path}")
        return

    if not cl_txt_path.exists():
        logging.critical(f"ERROR: Cl file not found at {cl_txt_path}")
        return

    # Galactic mask
    npix = hp.nside2npix(args.nside)
    theta, _ = hp.pix2ang(args.nside, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)
    mask = np.abs(galactic_latitude) > args.gal_cut
    mask_idx = np.where(mask)[0]

    sky_frac = len(mask_idx) / npix
    logging.info(f"Valid sky coverage: {sky_frac*100:.1f}%")

    # Real map
    try:
        real_map = hp.read_map(str(planck_path), field=0, dtype=np.float64)
        real_map_low = hp.ud_grade(real_map, args.nside)
        real_map_low = hp.remove_dipole(real_map_low, gal_cut=args.gal_cut)
        real_kurt = kurtosis(real_map_low[mask_idx], bias=False)
        logging.info(f"Real Kurtosis (masked): {real_kurt:.8f}")
    except Exception as e:
        logging.exception(f"Failed to process real map: {e}")
        return

    # Load the .txt file you downloaded
    data = np.loadtxt(cl_txt_path, comments='#', usecols=(0,1))
    ell = data[:, 0]
    dl_tt = data[:, 1]
    cl_theoretical = np.zeros(3 * args.nside)
    max_ell = min(len(ell), len(cl_theoretical))
    cl_theoretical[2:max_ell] = dl_tt[:max_ell-2] / (ell[:max_ell-2] * (ell[:max_ell-2] + 1)) * 2 * np.pi

    # Simulations
    sim_values = []
    logging.info(f"Running {args.n_sim} Monte Carlo simulations...")

    for i in range(args.n_sim):
        try:
            sim_map = hp.synfast(cl_theoretical, args.nside)
            sim_map = hp.remove_dipole(sim_map, gal_cut=args.gal_cut)
            sim_val = kurtosis(sim_map[mask_idx], bias=False)
            sim_values.append(sim_val)

            if (i + 1) % 200 == 0:
                logging.info(f"Progress: {i+1}/{args.n_sim}")
        except Exception as e:
            logging.error(f"Simulation {i} failed: {e}")
            continue

    if len(sim_values) == 0:
        logging.critical("No simulations completed.")
        return

    sim_values = np.array(sim_values)
    hits = np.sum(np.abs(sim_values) >= np.abs(real_kurt))
    p_value = hits / len(sim_values) if hits > 0 else 1.0 / (len(sim_values) + 1)

    # Verdict
    logging.info("\n" + "="*60)
    logging.info("FORTRESS MONTE CARLO VERDICT")
    logging.info(f"Real Kurtosis : {real_kurt:.8f}")
    logging.info(f"LCDM Mean     : {np.mean(sim_values):.8f}")
    logging.info(f"P-Value       : {p_value:.6f}")
    logging.info("="*60)

    if p_value < 0.05:
        logging.info("VERDICT: STATISTICALLY SIGNIFICANT ANOMALY DETECTED.")
    else:
        logging.info("VERDICT: Consistent with standard ΛCDM Gaussianity.")

if __name__ == "__main__":
    run_fortress_monte_carlo()
