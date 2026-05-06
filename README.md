# Probability Distribution Gallery

A visually rich Python project that explores **real-world and simulated probability distributions** using histograms, KDE (Kernel Density Estimation), and fitted theoretical PDFs.

This project generates a **dark-themed multi-panel visualization** comparing different statistical distributions derived from datasets like Titanic, Insurance, Student Performance, and Medical Appointments.

---

## Features

* **10 Probability Distributions Visualized**

  * Normal
  * Log-Normal
  * Exponential
  * Gamma
  * Poisson
  * Beta
  * Binomial
  * Weibull (simulated)
  * Triangular (simulated)
  * Uniform (simulated)

* **Modern Dark UI Visualization**

  * Custom color palette
  * Gradient histogram intensity
  * Styled grid and typography

* **Statistical Insights per Plot**

  * Mean (μ)
  * Standard Deviation (σ)
  * Skewness
  * Kurtosis
  * Sample Size (n)

* **Overlays**

  * Histogram (density)
  * KDE curve
  * Fitted theoretical PDF
  * Mean & Median markers

* **Smart Binning**

  * Uses **Freedman–Diaconis rule** for optimal histogram bins

---

## Dataset Requirements

Place the following CSV files in the same directory as the script:

```
train_and_test2.csv
insurance.csv
StudentsPerformance.csv
KaggleV2-May-2016.csv
```

Note: Not to worry I also pushed the csv file so nothing to do here anymore.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/distribution-python.git
cd distribution-python
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Linux / Termux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` with:

```
numpy
pandas
matplotlib
seaborn
scipy
```

---

## Usage

Run the script:

```bash
python main.py
```

---

## Output

* Generates and saves:

```
distribution_gallery.png
```

* Displays a **2×5 grid visualization** of distributions.

---

## Data Mapping

| Distribution | Dataset Source      |
| ------------ | ------------------- |
| Normal       | Titanic Age         |
| Log-Normal   | Titanic Fare        |
| Poisson      | Titanic SibSp       |
| Gamma        | Insurance Charges   |
| Binomial     | Medical No-show     |
| Beta         | Student Math Scores |
| Exponential  | Age Differences     |
| Others       | Simulated Data      |

---

## Code Highlights

### Distribution Fitting

```python
params = dist.fit(data)
pdf = dist.pdf(x, *params)
```

### KDE Estimation

```python
kde = stats.gaussian_kde(data)
```

### Smart Bin Calculation

```python
bw = 2 * iqr * len(data) ** (-1 / 3)
```

---

## Purpose

This project is ideal for:

* Learning probability distributions
* Data science visualization practice
* Statistical analysis demonstrations
* Understanding real-world vs theoretical distributions

---

## Future Improvements

* Interactive dashboard (Plotly / Dash)
* Distribution comparison metrics (KS test, AIC/BIC)
* Auto dataset detection
* Export per-distribution plots

---

## Author

**Hearns Mori**

---

## License

This project is open-source and free to use for educational and research purposes.