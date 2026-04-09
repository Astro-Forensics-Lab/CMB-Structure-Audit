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
import logging
import sys
import argparse
from scipy.stats import pearsonr

def run_multichannel_audit():
    parser = argparse.ArgumentParser(description="Multi-Channel Consistency Audit v3.3")
    parser.add_argument("--nside", type=int, default=128, help="NSIDE resolution")
    parser.add_argument("--gal-cut", type=float, default=20.0, help="Galactic latitude cut in degrees")
    parser.add_argument("--apod-width", type=float, default=5.0, help="Apodization width in degrees")
    parser.add_argument("--corr-threshold", type=float, default=0.92, help="Minimum correlation for high consistency")
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

    logging.info("Initializing Multi-Channel Consistency Audit (v3.3 - Apodized Mask)...")

    # 1. Path Configuration
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir_path = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Planck HFI Channels (high-frequency bolometers)
    channels = [
        "HFI_SkyMap_100_2048_R3.01_full.fits",
        "HFI_SkyMap_143_2048_R3.01_full.fits",
    ]

    # 3. Apodized Galactic Mask
    logging.info(f"Applying apodized Galactic Plane Mask (|b| > {args.gal_cut}° ± {args.apod_width}°)...")
    npix = hp.nside2npix(args.nside)
    theta, _ = hp.pix2ang(args.nside, np.arange(npix))
    galactic_latitude = 90.0 - np.degrees(theta)

    hard_mask = np.abs(galactic_latitude) > args.gal_cut
    apod_mask = np.ones(npix, dtype=np.float64)
    transition = (np.abs(galactic_latitude) > args.gal_cut - args.apod_width) & \
                 (np.abs(galactic_latitude) <= args.gal_cut)
    apod_mask[transition] = 0.5 * (1 + np.cos(
        np.pi * (args.gal_cut - np.abs(galactic_latitude[transition])) / args.apod_width
    ))
    apod_mask[~hard_mask] = 0.0

    normalized_maps = []
    channel_names = []

    # 4. Map Processing
    for channel in channels:
        file_path = data_dir_path / channel
        if not file_path.exists():
            logging.warning(f"File missing → {file_path}")
            continue

        try:
            logging.info(f"Processing channel: {channel}")
            m = hp.read_map(str(file_path), field=0, dtype=np.float64, verbose=False)
            m_low = hp.ud_grade(m, args.nside)

            # Apply mask before normalization
            clean_pixels = m_low[apod_mask > 0.5]
            mean_val = np.nanmean(clean_pixels)
            std_val = np.nanstd(clean_pixels)

            if std_val == 0 or np.isnan(std_val):
                logging.error(f"Zero variance in {channel}. Skipping.")
                continue

            m_norm = (m_low - mean_val) / std_val
            m_norm = m_norm * apod_mask  # keep masked area at 0

            normalized_maps.append(m_norm)
            channel_names.append(channel.split("_")[2])  # e.g. "100" or "143"

        except Exception as e:
            logging.exception(f"ERROR processing {channel}: {e}")
            continue

    if len(normalized_maps) < 2:
        logging.critical("ERROR: CROSS-CHANNEL AUDIT FAILED. Need at least 2 valid channels.")
        return

    # 5. Statistical Correlation (only on clean sky)
    clean_mask = apod_mask > 0.5
    correlation, p_value = pearsonr(
        normalized_maps[0][clean_mask],
        normalized_maps[1][clean_mask]
    )

    logging.info("\n" + "=" * 70)
    logging.info(f"INTER-CHANNEL CORRELATION (r): {correlation:.6f}")
    logging.info(f"STATISTICAL SIGNIFICANCE (p-value): {p_value:.2e}")
    logging.info("=" * 70)

    # 6. Save raw data
    np.save(results_dir / "channel_100_normalized_v3.3.npy", normalized_maps[0])
    np.save(results_dir / "channel_143_normalized_v3.3.npy", normalized_maps[1])

    # 7. Diagnostic Plot (fixed mollview usage)
    try:
        plot_file = results_dir / "multichannel_comparison_v3.3.png"
        fig = plt.figure(figsize=(20, 7))

        # Use healpy's sub= parameter instead of ax=
        hp.mollview(normalized_maps[0], title=f"{channel_names[0]} GHz\n(Z-score, masked)",
                    cmap="RdYlBu_r", sub=(1, 3, 1), hold=False)
        hp.mollview(normalized_maps[1], title=f"{channel_names[1]} GHz\n(Z-score, masked)",
                    cmap="RdYlBu_r", sub=(1, 3, 2), hold=False)
        hp.mollview(normalized_maps[0] - normalized_maps[1], title=f"Residuals\n(r = {correlation:.4f})",
                    cmap="RdBu_r", sub=(1, 3, 3), hold=False)

        plt.suptitle("Multi-Channel Consistency Audit v3.3", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=350, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"High-resolution plot saved → {plot_file}")

    except Exception as e:
        logging.error(f"Visualization Error: {e}")

    # 8. Verdict
    if correlation > args.corr_threshold:
        logging.info("VERDICT: High structural consistency detected between independent channels.")
        if p_value < 0.001:
            logging.info("CONFIDENCE: Result is statistically highly significant (p < 0.001).")
    else:
        logging.info("VERDICT: Correlation below threshold. Instrumental noise or foregrounds likely.")

    logging.info(f"\nAudit completed successfully.")

if __name__ == "__main__":
    run_multichannel_audit()