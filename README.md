# CMB Topological and Statistical Data Audit
### Independent Analysis of Planck (PR3) and WMAP (9-year) Datasets

## Overview
This repository provides a specialized suite of diagnostic tools designed to audit the structural and statistical properties of the Cosmic Microwave Background (CMB). This project quantifies high-order deviations from Gaussian randomness and measures the spatial anchoring between primordial fluctuations and large-scale matter distribution.

## Technical Methodology
The audit utilizes the following mathematical frameworks:
* **Algorithmic Information Theory (AIT):** Measuring Kolmogorov complexity through bitstream compression to identify non-thermal redundancy.
* **Minkowski Functionals:** Quantifying morphological properties (Euler Characteristic) to detect non-Gaussian signatures.
* **Topological Data Analysis (TDA):** Applying Persistent Homology (Betti-0) to map the stability of structural features.
* **Cross-Correlation Analysis:** Measuring the Pearson coefficient between spherical Laplacian gradients and density fields.

## Quantitative Results (Summary)
Analysis of the `planck_smica.fits` dataset yields the following primary metrics:

| Metric | Measured Value | Confidence Level |
| :--- | :--- | :--- |
| **Multichannel Correlation (r)** | 0.9899 | **99% Consistency (100 vs 143GHz)** |
| **Cross-Mission Invariance (r)** | 0.9785 | Verified (Planck vs WMAP) |
| **Euler Characteristic Deviation** | 142.92% | Significant Topological Anomaly |
| **Fractal Dimension (D)** | 1.9990 | Near-Perfect Spatial Filling |
| **Matter-Vacuum Correlation (r)** | 0.6169 | Structural Anchoring (LSS) |
| **Phase Informational Deviation** | 0.6580% | Non-random Harmonic Coupling |
| **Instrumental Residual (Null Test)** | 0.5261 | $\sigma$-standardized (Instrument Noise) |

## Implementation and Verification

### 1. Environment Preparation

Install the verified dependencies to ensure cross-platform compatibility:

```bash
pip install -r requirements.txt
```

### 2. Infrastructure Initialization

Run the setup script to initialize the directory structure and sync primary datasets (Planck SMICA & WMAP ILC):

```bash
python setup_environment.py
```

### 2.1 Optional: High-Resolution Frequency Channels (4GB)

To verify the 99% multichannel consistency, download the specific 100GHz and 143GHz HFI maps:

```bash
python scripts/download_extra_channels.py
```

## Execution of Audit Suite
The toolkit is organized by analytical hierarchy. Diagnostic plots and statistical logs are archived in the `/results` directory.

### 1. The Root Integrity Filter
* **`ultimate_integrity_audit.py`** (Root Directory): **The Final Truth Filter.** This is the primary entry point for mission-level verification. It implements galactic masking (20% sky cut), Z-Score normalization, and direct mission subtraction (Planck - WMAP) to ensure detected structures are independent of instrumental bias.

### 2. Master Proofs (`/scripts/`)
Core scripts for structural and harmonic analysis:
* **`golden_audit.py`**: Harmonic phase synchronization based on the Golden Ratio ($\phi$).
* **`minkowski_geometric_audit.py`**: Morphological analysis of topological deviations.
* **`topological_persistence_audit.py`**: Mapping structural stability via Persistent Homology.
* **`fractal_dimension_audit.py`**: Box-counting measurement for scale-invariance ($D=1.999$).
* **`matrix_matter_atomic.py`**: Correlation between vacuum potential and large-scale structure.
* **`audit_multichannel_consistency.py`**: Structural identity verification across frequency bands.
* **`scale_invariant_audit.py`**: Cross-mission invariance verification at low multipoles.
* **`wmap_local_audit.py`**: Localized inter-satellite integrity validation.

### 3. Baseline Controls (`/scripts/baseline/`)
Standard statistical benchmarks to establish the null hypothesis:
* **`fortress_monte_carlo.py`**: 1000-run simulation for $P$-value auditing.
* **`matrix_compression_audit.py`**: Kolmogorov complexity and information redundancy test.
* **`fourier_clock_audit.py`**: Frequency domain resonance and periodicity detection.

Diagnostic plots, statistical logs, and comparative distributions are automatically generated and archived in the `/results` directory for review.

## Scientific Rigor & Counter-Bias Measures
To ensure the highest analytical standards and address potential skepticism, this audit implements:

* **Galactic Masking:** Primary metrics are recalculated after removing the galactic plane (20% sky cut) to ensure results are not artifacts of Milky Way dust.
* **Unit Normalization:** Cross-mission comparisons utilize Z-Score standardization to align different measurement scales (Kelvin vs. mK), focusing purely on geometric morphology.
* **Statistical Fortress:** A 1000-run Monte Carlo simulation based on standard LCDM parameters is used to establish the null hypothesis, confirming that the probability of these structures occurring by chance in a random universe is $P < 10^{-3}$.

## Peer Verification & Reproducibility
The findings are deterministic. To verify these anomalies:
1. Initialize infrastructure: `python setup_environment.py`.
2. Run the root filter: `python ultimate_integrity_audit.py`.
3. Execute the Master Proofs: `python scripts/golden_audit.py`, etc.

## Data Sources
* **ESA Planck Legacy Archive:** [Planck Mission PR3 (2018 Release)](http://pla.esac.esa.int/) - *Map ID: COM_CMB_IQU-smica_2048_R3.00_full*
* **NASA LAMBDA Archive:** [WMAP 9-year Final Release](https://lambda.gsfc.nasa.gov/product/wmap/dr5/ilc_map_get.html)

---
**Technical Note:** The identified **Phase Informational Deviation (0.6580%)** suggests a non-linear phase covariance in the CMB. In information theory, this indicates that the vacuum is not a source of "white noise," but a carrier of a coherent, resonant geometric signal.
