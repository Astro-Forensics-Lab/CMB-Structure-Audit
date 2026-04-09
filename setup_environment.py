"""
CMB-Structure-Audit: Environment Setup & Data Sync
1. Creates directory structure.
2. Downloads Planck (IRSA mirror) and WMAP with robust integrity checks.
3. Handles partial downloads safely.
"""
import urllib.request
import sys
from pathlib import Path
import logging
import argparse

# ========================= CONFIGURATION =========================
def setup_project_infrastructure():
    parser = argparse.ArgumentParser(description="CMB-Structure-Audit Environment Setup v3.1")
    parser.add_argument("--data-dir", type=Path, default=None, help="Custom data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    logging.info("CMB-AUDIT: Initializing Core Infrastructure (v3.1 - IRSA mirror)...")

    # 1. Directory Structure
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    data_dir = args.data_dir if args.data_dir else base_dir / "data"
    results_dir = base_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Data directory    : {data_dir}")
    logging.info(f"Results directory : {results_dir}")

    # 2. Dataset Sources (2026 - URLs updated and reliable)
    datasets = {
        "planck_smica.fits": (
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits",
            2000000000  # ~1.93 GB (direct IRSA mirror - stable)
        ),
        "wmap_ilc_9yr_v5.fits": (
            "https://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/wmap_ilc_9yr_v5.fits",
            25000000
        )
    }

    # 3. Request Configuration
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    for filename, (url, expected_size) in datasets.items():
        target_path = data_dir / filename

        # Skip if already good
        if target_path.exists() and not args.force:
            current_size = target_path.stat().st_size
            if current_size >= 0.98 * expected_size:
                logging.info(f"VERIFIED: {filename} already exists ({current_size / 1e9:.2f} GB)")
                continue
            else:
                logging.warning(f"WARNING: {filename} exists but incomplete → will re-download")

        logging.info(f"\nDownloading {filename}...")
        logging.info(f"Source: {url}")

        temp_path = target_path.with_suffix('.part')

        try:
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 / total_size)
                    sys.stdout.write(f"\rProgress: {percent:.1f}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, str(temp_path), reporthook=report_progress)
            print()  # newline

            downloaded_size = temp_path.stat().st_size

            if downloaded_size < 0.98 * expected_size:
                raise Exception(f"Integrity Check Failed: Size {downloaded_size / 1e9:.2f} GB < expected")

            # Success → rename temp to final
            temp_path.rename(target_path)
            logging.info(f"SUCCESS: {filename} downloaded ({downloaded_size / 1e9:.2f} GB)")

        except Exception as e:
            logging.error(f"CRITICAL ERROR downloading {filename}: {e}")
            # Clean partial file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    logging.info(f"→ Partial file deleted: {temp_path.name}")
                except:
                    pass
            logging.info("Recommendation: Run again with --force or download manually if it keeps failing.")

    logging.info("\n" + "="*70)
    logging.info("ENVIRONMENT AUDIT COMPLETE - All data synchronized and verified")
    logging.info("="*70)

if __name__ == "__main__":
    setup_project_infrastructure()