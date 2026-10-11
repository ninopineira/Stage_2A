# Schematic illustration for the introduction of the report: why predicting the next cell
# matters for mobility management (reactive handover today vs predictive handover, the
# direction studied for 6G).
#
# ILLUSTRATIVE ONLY: the two signal curves come from a textbook log-distance path-loss
# model between two masts, not from the dataset (which contains no signal measurement).
# No prerequisite: nothing is read from Database/ or results/.
#
# Output:
#   results/plots/handover_reactive_vs_predictive.png
# (copied to rapport/figures/handover_predictif.png by make_report_figures.py)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MAIN_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Scene (arbitrary units) ──────────────────────────────────────────────────
MAST_A, MAST_B = 0.0, 1.0      # positions of the two masts along the route
MAST_HEIGHT = 0.12             # keeps the distance strictly positive under a mast
PATH_LOSS_EXPONENT = 3.5
MIN_USABLE = -22.0             # level under which the link is considered degraded

HYSTERESIS = 4.0               # the neighbour must be better than the serving cell by this margin...
TRIGGER_DELAY = 0.12           # ...for this long (in route units) before a reactive handover fires

Y_MIN, Y_MAX = -33.0, 3.5      # vertical extent of the panels

COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
COLOR_SERVED = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_DEGRADED = "#d03b3b"


def signal(x, mast):
    """Received signal strength (dB, arbitrary reference) at position x from a mast."""
    distance = np.sqrt((x - mast) ** 2 + MAST_HEIGHT ** 2)
    return -10 * PATH_LOSS_EXPONENT * np.log10(distance) - 30


def draw_panel(ax, x, sig_a, sig_b, x_switch, title, switch_label, show_degraded):
    served = np.where(x < x_switch, sig_a, sig_b)

    # The served signal is a wide grey band drawn UNDER the two curves, so their colours stay visible.
    ax.plot(x, served, color=COLOR_SERVED, lw=9, alpha=0.22, solid_capstyle="round",
            label="signal of the cell serving the phone")
    ax.plot(x, sig_a, color=COLOR_A, lw=1.8, label="signal of cell A")
    ax.plot(x, sig_b, color=COLOR_B, lw=1.8, label="signal of cell B")

    ax.axhline(MIN_USABLE, color=COLOR_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.012, MIN_USABLE + 0.6, "minimum level for a good link", color=COLOR_MUTED,
            fontsize=8.5, va="bottom")

    if show_degraded:
        degraded = x[(x < x_switch) & (sig_a < MIN_USABLE)]
        ax.axvspan(degraded.min(), x_switch, color=COLOR_DEGRADED, alpha=0.16, lw=0,
                   label="degraded link")
        ax.annotate("the link degrades\nbefore the switch", xy=(degraded.mean(), Y_MIN + 4.5),
                    xytext=(0.27, Y_MIN + 3.2), fontsize=9, color=COLOR_DEGRADED, ha="center",
                    va="center", arrowprops=dict(arrowstyle="-", color=COLOR_DEGRADED, lw=0.9))

    ax.axvline(x_switch, color=COLOR_SERVED, lw=1.0, ls=(0, (2, 2)))
    ax.text(x_switch, Y_MAX + 0.5, switch_label, fontsize=9, ha="center", va="bottom", color=COLOR_SERVED)

    ax.set_title(title, fontsize=11, loc="left", pad=26, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([MAST_A + 0.02, MAST_B - 0.02])
    ax.set_xticklabels(["mast of cell A", "mast of cell B"], fontsize=9)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[1].set_ha("right")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    ax.set_xlabel("position of the user along the route  →", fontsize=9, color=COLOR_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_MUTED)
        ax.spines[side].set_linewidth(0.8)


if __name__ == "__main__":
    x = np.linspace(0, 1, 1001)
    sig_a, sig_b = signal(x, MAST_A), signal(x, MAST_B)

    # Reactive: the handover fires once B has been better than A by HYSTERESIS for TRIGGER_DELAY.
    x_condition = x[np.argmax(sig_b - sig_a >= HYSTERESIS)]
    x_reactive = x_condition + TRIGGER_DELAY
    # Predictive: the next cell is known in advance, so the switch is made where the curves cross.
    x_predictive = x[np.argmax(sig_b >= sig_a)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    draw_panel(axes[0], x, sig_a, sig_b, x_reactive,
               "(a) Reactive handover (today)",
               "switch decided after\nmeasuring a better cell", show_degraded=True)
    draw_panel(axes[1], x, sig_a, sig_b, x_predictive,
               "(b) Predictive handover (aim for 6G)",
               "switch prepared in advance:\nthe next cell was predicted", show_degraded=False)
    axes[0].set_ylabel("received signal strength\n(illustrative, no unit)", fontsize=9, color=COLOR_MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    out = OUTPUT_DIR / "handover_reactive_vs_predictive.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"Saved {out}")
