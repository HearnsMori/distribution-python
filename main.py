import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
from scipy.stats import (
    norm, lognorm, poisson, gamma, binom, beta,
    expon, weibull_min, triang, uniform
)
import warnings
warnings.filterwarnings("ignore")

BACKGROUND   = "#0F1117"
PANEL_BG     = "#1A1D27"
GRID_COLOR   = "#2A2D3A"
TEXT_PRIMARY = "#F0F2F8"
TEXT_MUTED   = "#8890AA"
ACCENT       = "#5B8CFF"

PALETTE = [
    "#5B8CFF", "#FF6B9D", "#FFB347", "#4ECDC4",
    "#A78BFA", "#34D399", "#F87171", "#FBBF24",
    "#60A5FA", "#E879F9",
]

plt.rcParams.update({
    "figure.facecolor"      : BACKGROUND,
    "axes.facecolor"        : PANEL_BG,
    "axes.edgecolor"        : GRID_COLOR,
    "axes.labelcolor"       : TEXT_MUTED,
    "axes.titlecolor"       : TEXT_PRIMARY,
    "axes.grid"             : True,
    "grid.color"            : GRID_COLOR,
    "grid.linewidth"        : 0.6,
    "xtick.color"           : TEXT_MUTED,
    "ytick.color"           : TEXT_MUTED,
    "xtick.labelsize"       : 7.5,
    "ytick.labelsize"       : 7.5,
    "text.color"            : TEXT_PRIMARY,
    "font.family"           : "DejaVu Sans",
    "lines.linewidth"       : 2.2,
})


titanic      = pd.read_csv("train_and_test2.csv")
insurance    = pd.read_csv("insurance.csv")
students     = pd.read_csv("StudentsPerformance.csv")
appointments = pd.read_csv("KaggleV2-May-2016.csv")

normal_data    = titanic["Age"].dropna().values
lognormal_data = titanic["Fare"].dropna().values
poisson_data   = titanic["sibsp"].dropna().values
gamma_data     = insurance["charges"].values
binomial_data  = appointments["No-show"].map({"Yes": 1, "No": 0}).values
beta_data      = (students["math score"] / 100).clip(1e-4, 1 - 1e-4).values
exp_data       = np.abs(np.diff(np.sort(normal_data)))

weibull_data   = np.random.weibull(1.5, 1000)
triangular_data= np.random.triangular(0, 5, 10, 1000)
uniform_data   = np.random.uniform(0, 1, 1000)


def fit_and_pdf(dist, data, x):
    """Return fitted PDF values for `x`; None on failure."""
    try:
        params = dist.fit(data)
        return dist.pdf(x, *params)
    except Exception:
        return None


DISTRIBUTIONS = [
    ("Normal",      "Age",           normal_data,     norm,       {}),
    ("Log-Normal",  "Fare",          lognormal_data,  lognorm,    {}),
    ("Exponential", "Age Diffs",     exp_data,        expon,      {}),
    ("Gamma",       "Charges",       gamma_data,      gamma,      {}),
    ("Poisson",     "Siblings",      poisson_data,    None,       {}),
    ("Beta",        "Math Score/100",beta_data,       beta,       {}),
    ("Binomial",    "No-show",       binomial_data,   None,       {}),
    ("Weibull",     "Simulated",     weibull_data,    weibull_min,{}),
    ("Triangular",  "Simulated",     triangular_data, triang,     {}),
    ("Uniform",     "Simulated",     uniform_data,    uniform,    {}),
]

N_COLS = 5
N_ROWS = 2
fig = plt.figure(figsize=(22, 10), dpi=150, facecolor=BACKGROUND)

# Header
fig.text(0.5, 0.97,
         "Probability Distribution Gallery",
         ha="center", va="top",
         fontsize=22, fontweight="bold", color=TEXT_PRIMARY)
fig.text(0.5, 0.935,
         "Real-world & simulated datasets  ·  Empirical histogram  +  KDE  +  Fitted theoretical PDF",
         ha="center", va="top",
         fontsize=10, color=TEXT_MUTED, style="italic")

# Leave space for header
gs = gridspec.GridSpec(N_ROWS, N_COLS,
                       top=0.905, bottom=0.07,
                       left=0.04, right=0.98,
                       hspace=0.52, wspace=0.32)

for idx, (dist_name, data_label, data, scipy_dist, _) in enumerate(DISTRIBUTIONS):
    row, col = divmod(idx, N_COLS)
    ax = fig.add_subplot(gs[row, col])
    color = PALETTE[idx]

    # ── compute bins (Freedman–Diaconis rule) ──
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bw = 2 * iqr * len(data) ** (-1 / 3) if iqr > 0 else 0.1
    n_bins = max(10, min(60, int((data.max() - data.min()) / bw) if bw > 0 else 20))

    # ── histogram (density) ──
    n, bin_edges, patches = ax.hist(
        data, bins=n_bins, density=True, alpha=0.35,
        color=color, edgecolor="none", zorder=2
    )

    # gradient fill intensity by bar height
    max_n = n.max() if n.max() > 0 else 1
    for patch, val in zip(patches, n):
        patch.set_alpha(0.18 + 0.45 * (val / max_n))

    x_range = np.linspace(data.min(), data.max(), 400)

    # ── KDE ──
    try:
        kde = stats.gaussian_kde(data, bw_method="scott")
        kde_y = kde(x_range)
        ax.plot(x_range, kde_y, color=color, lw=2.2, alpha=0.9,
                label="KDE", zorder=4)
        ax.fill_between(x_range, kde_y, alpha=0.08, color=color, zorder=3)
    except Exception:
        pass

    # ── fitted theoretical PDF ──
    if scipy_dist is not None:
        pdf_y = fit_and_pdf(scipy_dist, data, x_range)
        if pdf_y is not None:
            ax.plot(x_range, pdf_y, color="white", lw=1.4,
                    linestyle="--", alpha=0.65, label="Fitted PDF", zorder=5)

    # ── vertical lines: mean & median ──
    mean_v   = np.mean(data)
    median_v = np.median(data)
    ylim_top = ax.get_ylim()[1]
    ax.axvline(mean_v,   color="#FFD700", lw=1.3, linestyle="-",  alpha=0.8, zorder=6)
    ax.axvline(median_v, color="#FF6B9D", lw=1.3, linestyle=":",  alpha=0.8, zorder=6)

    # ── stats badge (upper right corner) ──
    skew_v = stats.skew(data)
    kurt_v = stats.kurtosis(data)
    std_v  = np.std(data)

    badge_lines = [
        f"μ = {mean_v:.2f}",
        f"σ = {std_v:.2f}",
        f"skew = {skew_v:.2f}",
        f"kurt = {kurt_v:.2f}",
        f"n = {len(data):,}",
    ]
    badge_text = "\n".join(badge_lines)
    ax.text(0.97, 0.97, badge_text,
            transform=ax.transAxes,
            fontsize=6.2, color=TEXT_MUTED,
            verticalalignment="top", horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BACKGROUND,
                      edgecolor=GRID_COLOR, alpha=0.85))

    # ── title ──
    ax.set_title(f"{dist_name}", fontsize=11, fontweight="bold",
                 color=TEXT_PRIMARY, pad=6)
    ax.set_xlabel(data_label, fontsize=7.5, color=TEXT_MUTED, labelpad=3)
    ax.set_ylabel("Density", fontsize=7.5, color=TEXT_MUTED, labelpad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    # ── spine styling ──
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        spine.set_linewidth(0.8)

legend_handles = [
    mpatches.Patch(color=PALETTE[0], alpha=0.5, label="Histogram"),
    plt.Line2D([0], [0], color=PALETTE[0], lw=2.2,  label="KDE"),
    plt.Line2D([0], [0], color="white",    lw=1.4, linestyle="--", alpha=0.7, label="Fitted Theoretical PDF"),
    plt.Line2D([0], [0], color="#FFD700",  lw=1.3,  label="Mean"),
    plt.Line2D([0], [0], color="#FF6B9D",  lw=1.3, linestyle=":", label="Median"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=5,
           frameon=True,
           facecolor=PANEL_BG, edgecolor=GRID_COLOR,
           fontsize=8.5, labelcolor=TEXT_PRIMARY,
           bbox_to_anchor=(0.5, 0.01))

plt.savefig("distribution_gallery.png", dpi=180,
            bbox_inches="tight", facecolor=BACKGROUND)
plt.show()
print("Saved → distribution_gallery.png")