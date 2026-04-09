"""
Algorithmic Information Theory (AIT)
Measures Kolmogorov complexity using zlib compression and validates
against LCDM simulations for statistical significance (P-Value).
"""

import numpy as np
import healpy as hp
import zlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse

# ========================= CONFIGURATION =========================
def run_kolmogorov_audit():
    parser = argparse.ArgumentParser(description="AIT Kolmogorov Audit v3.0")
    parser.add_argument("--n-sim", type=int, default=100, help="Number of LCDM simulations")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data folder")
    args = parser.parse_args()

    # Logging configured inside the function for safety
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    np.random.seed(42)

    # Robust paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    map_path = data_dir / "planck_smica.fits"
    logging.info("CMB-AUDIT: Initializing Kolmogorov Complexity Audit (Monte Carlo Reinforced)...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    # 1. Load and Prepare Real Data
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64, verbose=False)
        real_data = hp.ud_grade(cmb_map, args.nside)
    except Exception as e:
        logging.exception(f"FAILED: {e}")
        return

    # 2. Calculate Real Complexity
    def calculate_complexity(data_array):
        """Calculates the compression ratio of a data array (float16)."""
        data_bytes = data_array.astype(np.float16).tobytes()
        compressed = zlib.compress(data_bytes, level=9)
        return len(compressed) / len(data_bytes)

    real_ratio = calculate_complexity(real_data)
    logging.info(f"\nREAL MAP RATIO: {real_ratio:.8f}")

    # 3. MONTE CARLO
    logging.info(f"Generating {args.n_sim} synthetic LCDM universes for P-Value audit...")
    cl_real = hp.anafast(real_data, lmax=3 * args.nside - 1)

    sim_ratios = []
    for i in range(args.n_sim):
        try:
            sim_map = hp.synfast(cl_real, args.nside, verbose=False)
            sim_ratios.append(calculate_complexity(sim_map))
            if (i + 1) % 20 == 0:
                logging.info(f"Simulations: {i+1}/{args.n_sim}")
        except Exception as e:
            logging.error(f"Simulation {i} failed: {e}")
            continue

    if len(sim_ratios) == 0:
        logging.critical("No successful simulations.")
        return

    sim_ratios = np.array(sim_ratios)
    p_value = np.sum(sim_ratios <= real_ratio) / len(sim_ratios)

    # 4. Redundancy Mapping (sliding window - kept exactly as original)
    npix = hp.nside2npix(args.nside)
    redundancy_map = np.zeros(npix)
    step = 128
    for i in range(0, npix, step):
        chunk = real_data[i:i + step]
        if len(chunk) > 0:
            redundancy_map[i:i + len(chunk)] = calculate_complexity(chunk)

    # 5. Save raw data for reproducibility
    np.save(results_dir / "real_ratio_v3.npy", real_ratio)
    np.save(results_dir / "sim_ratios_v3.npy", sim_ratios)
    np.save(results_dir / "redundancy_map_v3.npy", redundancy_map)
    np.save(results_dir / "cl_real_v3.npy", cl_real)

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    hp.mollview(redundancy_map, title="Kolmogorov Complexity Map (Local Entropy)",
                cmap="plasma", unit="Ratio", hold=False)

    output_plot = results_dir / "algorithmic_complexity_map_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Final Verdict
    logging.info("\n" + "="*55)
    logging.info("AIT AUDIT VERDICT")
    logging.info(f"Real Map Ratio: {real_ratio:.6f}")
    logging.info(f"LCDM Mean Ratio: {np.mean(sim_ratios):.6f}")
    logging.info(f"P-Value: {p_value:.10f}")
    logging.info("="*55)

    if p_value < 0.05:
        logging.info("RESULT: STATISTICALLY SIGNIFICANT. Structure detected beyond random noise.")
    else:
        logging.info("RESULT: Consistent with standard Gaussian fluctuations.")

    logging.info(f"\nAudit plot archived at: {output_plot}")
    logging.info(f"Raw data saved in: {results_dir}")

if __name__ == "__main__":
    run_kolmogorov_audit()