"""
Multi-Channel Consistency Audit
Validates structural persistence across independent physical frequency detectors
(100 GHz and 143 GHz Bolometers). Implements Galactic Masking to eliminate
foreground bias. Focuses on spatial form (Z-score + pixel correlation).
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse
from scipy.stats import pearsonr

# ========================= CONFIGURATION =========================
def run_multichannel_audit():
    parser = argparse.ArgumentParser(description="Multi-Channel Consistency Audit v3.0")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--gal-cut", type=float, default=20.0, help="Galactic latitude cut in degrees")
    parser.add_argument("--corr-threshold", type=float, default=0.92, help="Minimum correlation for high consistency")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data folder")
    args = parser.parse_args()

    # Logging configured inside the function
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    np.random.seed(42)

    version = "3.0"
    start_time = datetime.now()
    logging.info(f"Initializing Multi-Channel Consistency Audit v{version} [{start_time.strftime('%Y-%m-%d %H:%M:%S')}]")

    # ====================== 1. PATH CONFIGURATION ======================
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir_path = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ====================== 2. PLANCK HFI CHANNELS ======================
    channels = [
        "HFI_SkyMap_100_2048_R3.01_full.fits",
        "HFI_SkyMap_143_2048_R3.01_full.fits",
    ]

    # ====================== 3. GALACTIC MASK ======================
    logging.info(f"Applying Galactic mask (|b| > {args.gal_cut}°) to isolate CMB signal...")
    npix = hp.nside2npix(args.nside)
    theta, _ = hp.pix2ang(args.nside, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)
    mask = np.abs(galactic_latitude) > args.gal_cut
    mask_idx = np.where(mask)[0]

    if len(mask_idx) == 0:
        logging.critical("ERROR: Invalid mask parameters.")
        return None

    sky_coverage = (len(mask_idx) / npix) * 100
    logging.info(f"Mask active: {len(mask_idx):,} pixels ({sky_coverage:.1f}% of sky)")

    normalized_maps = []
    channel_names = []

    # ====================== 4. MAP PROCESSING ======================
    for channel in channels:
        file_path = data_dir_path / channel
        if not file_path.exists():
            logging.warning(f"File missing → {file_path}")
            continue
        try:
            logging.info(f"Processing channel: {channel}")
            m = hp.read_map(str(file_path), field=0, dtype=np.float64, verbose=False)
            m_low = hp.ud_grade(m, args.nside)
            clean_data = m_low[mask_idx]
            mean_val = np.nanmean(clean_data)
            std_val = np.nanstd(clean_data)
            if std_val == 0 or np.isnan(std_val):
                logging.error(f"Zero variance in {channel}. Skipping.")
                continue
            normalized_maps.append((m_low - mean_val) / std_val)
            channel_names.append(channel)
        except Exception as e:
            logging.exception(f"ERROR processing {channel}: {e}")
            continue

    # ====================== 5. STATISTICAL CORRELATION ======================
    if len(normalized_maps) < 2:
        logging.critical("ERROR: CROSS-CHANNEL AUDIT FAILED. Need at least 2 valid channels.")
        return None

    correlation, p_value = pearsonr(
        normalized_maps[0][mask_idx], normalized_maps[1][mask_idx]
    )

    logging.info("\n" + "=" * 70)
    logging.info(f"INTER-CHANNEL CORRELATION (r): {correlation:.6f}")
    logging.info(f"STATISTICAL SIGNIFICANCE (p-value): {p_value:.2e}")
    logging.info("=" * 70)

    # ====================== 6. SAVE RAW DATA ======================
    np.save(results_dir / "channel_100_normalized_v3.npy", normalized_maps[0])
    np.save(results_dir / "channel_143_normalized_v3.npy", normalized_maps[1])

    # ====================== 7. DETAILED REPORT ======================
    report_file = results_dir / "multichannel_audit_report_v3.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"--- AUDIT SESSION v{version}: {datetime.now().isoformat()} ---\n")
        f.write(f"Channels Compared : {', '.join(channel_names)}\n")
        f.write(f"Nside / Cut : {args.nside} / |b| > {args.gal_cut}°\n")
        f.write(f"Sky Coverage : {sky_coverage:.1f}%\n")
        f.write(f"Correlation (r) : {correlation:.6f}\n")
        f.write(f"P-Value : {p_value:.2e}\n")
        f.write(f"Duration : {datetime.now() - start_time}\n")
        f.write("-" * 60 + "\n")

    # ====================== 8. DIAGNOSTIC PLOT ======================
    try:
        plot_file = results_dir / f"multichannel_v{version.replace('.', '_')}_comparison.png"
        fig, axs = plt.subplots(1, 3, figsize=(20, 7))
        m1_plot = normalized_maps[0].copy()
        m2_plot = normalized_maps[1].copy()
        m1_plot[~mask] = np.nan
        m2_plot[~mask] = np.nan

        hp.mollview(m1_plot, title=f"{channel_names[0]}\n(Z-score, masked)", cmap="RdYlBu_r", ax=axs[0], hold=False)
        hp.mollview(m2_plot, title=f"{channel_names[1]}\n(Z-score, masked)", cmap="RdYlBu_r", ax=axs[1], hold=False)
        hp.mollview(m1_plot - m2_plot, title=f"Residuals\n(r = {correlation:.4f})", cmap="RdBu_r", ax=axs[2], hold=False)

        plt.suptitle(f"Multi-Channel Consistency Audit v{version}", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"High-resolution plot saved → {plot_file}")
    except Exception as e:
        logging.error(f"Visualization Error: {e}")

    # ====================== 9. VERDICT ======================
    if correlation > args.corr_threshold:
        logging.info("VERDICT: High structural consistency detected between independent channels.")
        if p_value is not None and p_value < 0.001:
            logging.info("CONFIDENCE: Result is statistically highly significant (p < 0.001).")
    else:
        logging.info("VERDICT: Correlation below threshold. Instrumental noise or foregrounds likely.")

    logging.info(f"\nFull audit report saved in: {report_file}")
    logging.info(f"Total execution time: {datetime.now() - start_time}")
    return correlation

if __name__ == "__main__":
    run_multichannel_audit()
