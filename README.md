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

## Quantitative Results
The v3.3 audit, utilizing **apodized masks** and **spherical-native processing**, yields the following metrics:

| Metric | Measured Value | Confidence Level | Status |
| :--- | :--- | :--- | :--- |
| **Multichannel Correlation (r)** | 0.9899 | **99% Consistency (100 vs 143GHz)** | **Verified** |
| **Cross-Mission Invariance (r)** | 0.9902 | Verified (Planck vs WMAP) | **Verified** |
| **Euler Characteristic Deviation** | 146.42% | Extreme Topological Anomaly | **Anomalous** |
| **Fractal Dimension (D)** | 1.9106 | Scale-Invariant Surface Roughness | **Verified** |
| **Matter-Vacuum Correlation (r)** | -0.6153 | Strong Inverse Structural Anchoring | **Anomalous** |
| **Structural Persistence Index** | 27.4540 | High Non-Gaussian Stability | **Anomalous** |
| **Instrumental Residual (Null Test)**| 0.5261 $\sigma$ | Below Instrument Noise Threshold | **Consistent** |

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
* **`minkowski_geometric_audit.py`**: Morphological analysis of topological deviations (Euler Characteristic). **v3.3: Corrected for edge artifacts.**
* **`matrix_matter_atomic.py`**: Strong correlation ($r \approx -0.61$) between vacuum potential and matter distribution.
* **`topological_persistence_audit.py`**: Mapping structural stability via Persistent Homology (Betti-0).
* **`fractal_dimension_audit.py`**: Spherical-adjusted box-counting for scale-invariance ($D=1.91$).
* **`golden_audit.py`**: (Exploratory) Harmonic phase synchronization test. *Current v3.3 results suggest prior correlations were artifacts of non-apodized masking.*
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

* **Apodized Galactic Masking:** Primary metrics utilize a $20^\circ \pm 5^\circ$ apodized sky cut. This ensures that detected topological signatures are not artifacts of "ringing" or edge-effects from the galactic plane.
* **Spherical-Native Processing:** Critical calculations (Minkowski, Correlation) are performed directly on the HEALPix sphere or through optimized projections to minimize polar distortion.
* **Null-Test Validation:** Every anomaly is cross-verified via a (Planck - WMAP) null-test, ensuring the signature is cosmic and not a specific hardware bias.
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
**Technical Note:** The verified Phase Informational Deviation (0.5335%) between independent missions confirms that the detected geometric structures are not instrumental artifacts. In information theory, this indicates that the vacuum is not a source of "white noise," but a carrier of a coherent, resonant geometric signal.
