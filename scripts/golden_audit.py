"""
Phase Synchronization Audit
Investigates harmonic coupling between multipole scales based on the Golden Ratio (Phi)
with Monte Carlo significance testing.
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
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Phase Synchronization Audit v3.0")
    parser.add_argument("--lmax", type=int, default=1000, help="Lmax máximo para map2alm")
    parser.add_argument("--l-base", type=int, default=50, help="Multipolo base para escalas Phi")
    parser.add_argument("--data-dir", type=Path, default=None, help="Pasta de dados")
    return parser.parse_args()

def run_golden_audit():
    args = parse_args()
    np.random.seed(42)

    # Paths robustos
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    map_path = data_dir / "planck_smica.fits"
    logging.info("CMB-AUDIT: Initializing Golden Ratio Phase Synchronization Audit (v3)...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    # 2. Extract Spherical Harmonics
    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64, verbose=False)
        logging.info(f"Computing Spherical Harmonic Coefficients (alm) up to lmax={args.lmax}...")
        alm = hp.map2alm(cmb_map, lmax=args.lmax)
    except Exception as e:
        logging.exception(f"FAILED to process spherical harmonics: {e}")
        return

    # 3. Golden Ratio Scales (Phi)
    phi = (1 + 5**0.5) / 2
    l1 = args.l_base
    l2 = int(l1 * phi)
    l3 = int(l1 * phi**2)
    logging.info(f"Testing Phase Synchrony at Golden Scales: l = {l1}, {l2}, {l3}")

    # 4. Phase Extraction (m=0)
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

    # 5. MONTE CARLO SIGNIFICANCE TEST
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
    np.save(results_dir / "alm_v3.npy", alm)
    np.save(results_dir / "golden_phases_v3.npy", np.array([phase1, phase2, phase3]))
    np.save(results_dir / "golden_diff_v3.npy", golden_diff)
    np.save(results_dir / "mc_random_diffs_v3.npy", random_diffs)

    # 6. Visualization
    all_phases = np.angle(alm)

    plt.figure(figsize=(10, 6))
    plt.hist(all_phases, bins=100, color='gold', alpha=0.7, density=True, label='Empirical CMB Phases (All)')
    plt.axvline(golden_diff, color='red', linestyle='--', linewidth=2,
                label=f'Phi-Coupling Error: {abs(golden_diff):.4f} rad')

    plt.title("Phase Signature Audit: Golden Ratio Resonance")
    plt.xlabel("Phase Angle (radians)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.2)

    output_plot = results_dir / "golden_ratio_resonance_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 7. FINAL VERDICT
    logging.info("\n" + "="*65)
    logging.info("GOLDEN RATIO PHASE SYNCHRONIZATION VERDICT")
    logging.info(f"Phase Synchronization Error : {abs(golden_diff):.6f} rad")
    logging.info(f"P-Value (Probability of Fluke): {p_value:.6f}")
    logging.info("="*65)

    if p_value < 0.05:
        logging.info("SIGNIFICANT ALIGNMENT: Phi-coupling is statistically anomalous.")
        logging.info("This suggests a non-random geometric resonance in the vacuum state.")
    else:
        logging.info("RESULT: No significant Golden Ratio synchronization.")
        logging.info("Phase distribution is consistent with stochastic Gaussian noise.")
    logging.info(f"\nAudit plot archived at: {output_plot}")
    logging.info(f"Raw data saved in: {results_dir}")

if __name__ == "__main__":
    run_golden_audit()