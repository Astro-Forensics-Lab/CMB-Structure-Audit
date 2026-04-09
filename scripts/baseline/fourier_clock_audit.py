"""
Fourier Clock Audit (v3.0 – Clean & Robust)
Análise de ressonâncias periódicas no espectro de potência CMB via FFT.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse

# ========================= CONFIGURAÇÃO =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fourier Clock Audit v3.0")
    parser.add_argument("--lmax", type=int, default=2000, help="Lmax máximo do espectro")
    parser.add_argument("--data-dir", type=Path, default=None, help="Pasta de dados")
    return parser.parse_args()

def run_fourier_clock_audit():
    args = parse_args()
    np.random.seed(42)  # reprodutibilidade

    # Paths robustos
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    map_path = data_dir / "planck_smica.fits"
    logging.info("CMB-AUDIT: Initializing Resonance Frequency Analysis...")

    if not map_path.exists():
        logging.critical(f"ERROR: Data file not found at {map_path}")
        return

    try:
        cmb_map = hp.read_map(str(map_path), field=0, dtype=np.float64, verbose=False)
        logging.info(f"Calculating Angular Power Spectrum up to lmax={args.lmax}...")
        cl = hp.anafast(cmb_map, lmax=args.lmax)
    except Exception as e:
        logging.exception(f"FAILED to process map: {e}")
        return

    # === PRÉ-PROCESSAMENTO ===
    l = np.arange(len(cl))
    # Detrending (mantido exatamente como no original)
    cl_detrended = cl / (l**2 + 1e-8)

    # FFT
    N = len(cl_detrended)
    yf = fft(cl_detrended)
    powers = 2.0 / N * np.abs(yf[0:N//2])
    xf = np.linspace(0.0, 1.0, N//2)

    # === DETEÇÃO DE PICOS ===
    peaks, _ = find_peaks(powers[1:], height=np.mean(powers[1:]) * 3, distance=5)

    max_peak = np.max(powers[1:])
    mean_power = np.mean(powers[1:])
    snr = max_peak / mean_power

    # === SALVAR DADOS PARA REPRODUTIBILIDADE ===
    np.save(results_dir / "cl_raw_v3.npy", cl)
    np.save(results_dir / "cl_detrended_v3.npy", cl_detrended)
    np.save(results_dir / "fft_powers_v3.npy", powers)
    np.save(results_dir / "fft_xf_v3.npy", xf)

    # === VISUALIZAÇÃO ===
    plt.figure(figsize=(12, 6))
    plt.plot(xf[1:], powers[1:], color='gold', linewidth=1.2, label='FFT (detrended)')
    if len(peaks) > 0:
        plt.plot(xf[1:][peaks], powers[1:][peaks], 'ro', label=f'Peaks ({len(peaks)} detected)')
    plt.title("Frequency Domain Audit: CMB Power Spectrum Resonances (detrended)")
    plt.xlabel("Normalized Oscillation Frequency")
    plt.ylabel("Signal Intensity")
    plt.yscale('log')
    plt.grid(True, alpha=0.2)
    plt.legend()

    output_plot = results_dir / "fourier_clock_resonance_v3.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # === VERDICT ===
    logging.info("\n" + "="*60)
    logging.info(f"RESONANCE SIGNAL STRENGTH (SNR): {snr:.4f}")
    logging.info(f"Number of significant peaks: {len(peaks)}")
    logging.info("="*60)

    if snr > 5.0:
        logging.info("SIGNIFICANT ANOMALY DETECTED: Periodic component found in vacuum fluctuations.")
    else:
        logging.info("RESULT: No significant periodic signal above noise level.")
    logging.info(f"Plot saved: {output_plot}")
    logging.info(f"Raw data saved in {results_dir}")

if __name__ == "__main__":
    run_fourier_clock_audit()