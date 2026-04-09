"""
CMB-Structure-Audit: Extra Channels Downloader
Downloads the 100 GHz and 143 GHz maps for Multichannel Audit.
"""

import urllib.request
import sys
from pathlib import Path


def report_progress(block_num: int, block_size: int, total_size: int, filename: str):
    """Clean and user-friendly progress bar."""
    if total_size > 0:
        percent = min(100, block_num * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading {filename}: {percent:.1f}% ({total_size/1e9:.1f} GB)")
        sys.stdout.flush()


def main():
    # Location of the data folder (relative to this script)
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)   # ← FIXED: creates the folder automatically

    extra_datasets = {
        "HFI_SkyMap_100_2048_R3.01_full.fits": 
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits",
        "HFI_SkyMap_143_2048_R3.01_full.fits": 
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits",
    }

    print("=== Extra Channels Download (≈ 4 GB Total) ===")
    print("Using official IRSA (NASA) mirror - links updated 2026\n")

    for filename, url in extra_datasets.items():
        target = data_dir / filename
        part_file = target.with_suffix(".fits.part")

        # If the final file already exists and is large enough → skip
        if target.exists() and target.stat().st_size > 1_500_000_000:  # > 1.5 GB
            print(f"[SKIP] {filename} already exists and is complete.")
            continue

        # If a partial file exists → inform the user
        if part_file.exists():
            print(f"[INFO] Partial file found for {filename}. Resuming download...")

        print(f"\nStarting download: {filename} (≈2.0 GB)")

        try:
            # Download to .part first (protection against interruptions)
            urllib.request.urlretrieve(
                url,
                str(part_file),
                reporthook=lambda nb, bs, ts: report_progress(nb, bs, ts, filename)
            )

            # If we reach here, the download was successful
            part_file.rename(target)
            print(f"\n[DONE] {filename} saved successfully! ({target.stat().st_size / 1e9:.2f} GB)\n")

        except urllib.error.HTTPError as e:
            print(f"\n[HTTP ERROR {e.code}] Failed to download {filename}")
            print(f"   URL: {url}")
            if part_file.exists():
                part_file.unlink()  # remove corrupted partial file
        except urllib.error.URLError as e:
            print(f"\n[NETWORK ERROR] Please check your internet connection: {e.reason}")
            if part_file.exists():
                part_file.unlink()
        except KeyboardInterrupt:
            print("\n\n[DOWNLOAD INTERRUPTED by user]")
            if part_file.exists():
                print(f"   Partial file kept: {part_file}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] {type(e).__name__}: {e}")
            if part_file.exists():
                part_file.unlink()

    print("=== Extra channels download completed! ===")


if __name__ == "__main__":
    main()