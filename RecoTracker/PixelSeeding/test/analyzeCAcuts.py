#!/usr/bin/env python3
"""
CA Connection Cut Tuning Analysis.

Reads per-TP ntuple from TrueStubNtupleAnalyzer, forms doublets from
consecutive-layer hits/stubs within each TP, computes CA cut variables,
and plots per-layer-pair distributions with current cut thresholds overlaid.

Usage:
    python3 analyzeCAcuts.py <rootfile> [--output-dir plots/] [--pairs ot-ot|pix-ot|all]
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

try:
    import uproot
except ImportError:
    print("ERROR: uproot is required. Install with: pip install uproot awkward", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Layer pair definitions from caHitNtupletAlpakaPhase2OTStubs_cfi.py
# Each entry: (inner_CA, outer_CA, label, cuts_dict)
# cuts_dict keys: phiCut, minDZ, maxDZ, ptCut, stubSigmaCut, cellZ0Cut
# ============================================================================

CELL_Z0_CUT = 12.5  # Global z0 cut

# OT barrel-barrel consecutive pairs
OT_BARREL_PAIRS = [
    (28, 29, "OT_B1-B2", dict(phiCut=1100, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=4.0)),
    (29, 30, "OT_B2-B3", dict(phiCut=1250, minDZ=-40, maxDZ=40, ptCut=0.85, stubSigmaCut=4.0)),
    (30, 31, "OT_B3-B4", dict(phiCut=1250, minDZ=-40, maxDZ=40, ptCut=0.85, stubSigmaCut=4.0)),
    (31, 32, "OT_B4-B5", dict(phiCut=2000, minDZ=-40, maxDZ=40, ptCut=0.85, stubSigmaCut=4.0)),
    (32, 33, "OT_B5-B6", dict(phiCut=2000, minDZ=-40, maxDZ=40, ptCut=0.85, stubSigmaCut=4.0)),
]

# OT barrel to backward disk
OT_BARREL_BWD_PAIRS = [
    (28, 34, "OT_B1-BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (29, 34, "OT_B2-BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (30, 34, "OT_B3-BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (31, 34, "OT_B4-BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (32, 34, "OT_B5-BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
]

# OT barrel to forward disk
OT_BARREL_FWD_PAIRS = [
    (28, 39, "OT_B1-FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (29, 39, "OT_B2-FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (30, 39, "OT_B3-FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (31, 39, "OT_B4-FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
    (32, 39, "OT_B5-FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=5.0)),
]

# OT backward disk consecutive
OT_BWD_DISK_PAIRS = [
    (34, 35, "OT_BD1-BD2", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (35, 36, "OT_BD2-BD3", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (36, 37, "OT_BD3-BD4", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (37, 38, "OT_BD4-BD5", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
]

# OT forward disk consecutive
OT_FWD_DISK_PAIRS = [
    (39, 40, "OT_FD1-FD2", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (40, 41, "OT_FD2-FD3", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (41, 42, "OT_FD3-FD4", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
    (42, 43, "OT_FD4-FD5", dict(phiCut=1500, minDZ=-50, maxDZ=50, ptCut=0.85, stubSigmaCut=5.0)),
]

# Pixel barrel to OT barrel L1
PIX_B_TO_OT_B1_PAIRS = [
    (2, 28, "Pix_B3-OT_B1", dict(phiCut=1200, minDZ=-15, maxDZ=15, ptCut=2.0, stubSigmaCut=0.0)),
    (3, 28, "Pix_B4-OT_B1", dict(phiCut=1000, minDZ=-22, maxDZ=22, ptCut=0.85, stubSigmaCut=0.0)),
]

# Forward pixel disks 1-5 (CA 4-8) to OT barrel L1 (CA 28)
PIX_FWD_TO_OT_B1_PAIRS = [
    (4, 28, "Pix_FD1-OT_B1", dict(phiCut=1000, minDZ=5.0, maxDZ=32.5, ptCut=0.85, stubSigmaCut=0.0)),
    (5, 28, "Pix_FD2-OT_B1", dict(phiCut=1000, minDZ=-10, maxDZ=50, ptCut=0.85, stubSigmaCut=0.0)),
    (6, 28, "Pix_FD3-OT_B1", dict(phiCut=1000, minDZ=5.0, maxDZ=50, ptCut=0.85, stubSigmaCut=0.0)),
    (7, 28, "Pix_FD4-OT_B1", dict(phiCut=1000, minDZ=15.0, maxDZ=70, ptCut=0.85, stubSigmaCut=0.0)),
    (8, 28, "Pix_FD5-OT_B1", dict(phiCut=850, minDZ=25.0, maxDZ=70, ptCut=0.85, stubSigmaCut=0.0)),
]

# Forward pixel disks 1-5 (CA 4-8) to OT forward disks 1-3 (CA 39-41)
PIX_FWD_TO_OT_FWD_PAIRS = [
    (4, 39, "Pix_FD1-OT_FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (5, 39, "Pix_FD2-OT_FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (6, 39, "Pix_FD3-OT_FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (7, 39, "Pix_FD4-OT_FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (8, 39, "Pix_FD5-OT_FD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (4, 40, "Pix_FD1-OT_FD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (5, 40, "Pix_FD2-OT_FD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (6, 40, "Pix_FD3-OT_FD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (7, 40, "Pix_FD4-OT_FD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (8, 40, "Pix_FD5-OT_FD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (4, 41, "Pix_FD1-OT_FD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (5, 41, "Pix_FD2-OT_FD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (6, 41, "Pix_FD3-OT_FD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (7, 41, "Pix_FD4-OT_FD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (8, 41, "Pix_FD5-OT_FD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (9, 39, "Pix_FD6-OT_FD1", dict(phiCut=2000, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (9, 40, "Pix_FD6-OT_FD2", dict(phiCut=2000, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
]

# Backward pixel disks 1-5 (CA 16-20) to OT barrel L1 (CA 28)
PIX_BWD_TO_OT_B1_PAIRS = [
    (16, 28, "Pix_BD1-OT_B1", dict(phiCut=1000, minDZ=-32.5, maxDZ=-5.0, ptCut=0.85, stubSigmaCut=0.0)),
    (17, 28, "Pix_BD2-OT_B1", dict(phiCut=1000, minDZ=-50, maxDZ=-10, ptCut=0.85, stubSigmaCut=0.0)),
    (18, 28, "Pix_BD3-OT_B1", dict(phiCut=1000, minDZ=-50, maxDZ=-5.0, ptCut=0.85, stubSigmaCut=0.0)),
    (19, 28, "Pix_BD4-OT_B1", dict(phiCut=1000, minDZ=-70, maxDZ=-15.0, ptCut=0.85, stubSigmaCut=0.0)),
    (20, 28, "Pix_BD5-OT_B1", dict(phiCut=850, minDZ=-70, maxDZ=-25.0, ptCut=0.85, stubSigmaCut=0.0)),
]

# Backward pixel disks 1-5 (CA 16-20) to OT backward disks 1-3 (CA 34-36)
PIX_BWD_TO_OT_BWD_PAIRS = [
    (16, 34, "Pix_BD1-OT_BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (17, 34, "Pix_BD2-OT_BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (18, 34, "Pix_BD3-OT_BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (19, 34, "Pix_BD4-OT_BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (20, 34, "Pix_BD5-OT_BD1", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (16, 35, "Pix_BD1-OT_BD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (17, 35, "Pix_BD2-OT_BD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (18, 35, "Pix_BD3-OT_BD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (19, 35, "Pix_BD4-OT_BD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (20, 35, "Pix_BD5-OT_BD2", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (16, 36, "Pix_BD1-OT_BD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (17, 36, "Pix_BD2-OT_BD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (18, 36, "Pix_BD3-OT_BD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (19, 36, "Pix_BD4-OT_BD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (20, 36, "Pix_BD5-OT_BD3", dict(phiCut=1500, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (21, 34, "Pix_BD6-OT_BD1", dict(phiCut=2000, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
    (21, 35, "Pix_BD6-OT_BD2", dict(phiCut=2000, minDZ=-1e4, maxDZ=1e4, ptCut=0.85, stubSigmaCut=0.0)),
]

PIX_TO_OT_PAIRS = (PIX_B_TO_OT_B1_PAIRS
                    + PIX_FWD_TO_OT_B1_PAIRS + PIX_FWD_TO_OT_FWD_PAIRS
                    + PIX_BWD_TO_OT_B1_PAIRS + PIX_BWD_TO_OT_BWD_PAIRS)

ALL_OT_PAIRS = OT_BARREL_PAIRS + OT_BARREL_BWD_PAIRS + OT_BARREL_FWD_PAIRS + OT_BWD_DISK_PAIRS + OT_FWD_DISK_PAIRS
ALL_PAIRS = ALL_OT_PAIRS + PIX_TO_OT_PAIRS

# Build lookup: (inner_CA, outer_CA) -> list of (label, cuts)
PAIR_LOOKUP = defaultdict(list)
for inner_ca, outer_ca, label, cuts in ALL_PAIRS:
    PAIR_LOOKUP[(inner_ca, outer_ca)].append((label, cuts))


# ============================================================================
# CA cut variable computations
# ============================================================================

def dphi_wrap(phi1, phi2):
    """Compute phi2 - phi1 with wraparound to [-pi, pi]."""
    d = phi2 - phi1
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def compute_doublet_variables(ri, zi, phi_i, ro, zo, phi_o,
                              dPhiDr_i=None, dPhiDrErr_i=None,
                              dPhiDr_o=None, dPhiDrErr_o=None,
                              stubType_i=None, stubType_o=None):
    """Compute all CA doublet cut variables.

    Returns dict of arrays (one value per doublet).
    """
    dphi = dphi_wrap(phi_i, phi_o)
    dr = ro - ri
    dz = zo - zi

    # D6: signed dz
    # D7: z0 = |zi*ro - ri*zo| / dr  (skip for SS stubs)
    z0_cross = np.abs(zi * ro - ri * zo)
    z0_dr = np.abs(dr)
    z0_val = np.where(z0_dr > 1e-6, z0_cross / z0_dr, 0.0)

    # D8: dphi in int16 units (short2phi: x * pi / 32768)
    idphi = np.abs(dphi) * 32768.0 / np.pi

    # D10: pt cut test
    # Fails if dphi^2 * (ptCut - ri*ro) > (ro - ri)^2
    # For signal tracks this ratio should be < 1
    pt_lhs = dphi ** 2 * np.abs(ri * ro)  # approximate; exact formula uses (ptCut - ri*ro)
    pt_rhs = dr ** 2

    result = {
        "dphi": dphi,
        "abs_dphi": np.abs(dphi),
        "idphi": idphi,
        "dr": dr,
        "dz": dz,
        "z0": z0_val,
    }

    # D11: stub-stub kappa significance (when both are stubs with dPhiDr)
    if dPhiDr_i is not None and dPhiDr_o is not None:
        # kappa = dPhiDr / sqrt(1 + r^2 * dPhiDr^2)
        den_i = 1.0 + ri ** 2 * dPhiDr_i ** 2
        k_i = dPhiDr_i / np.sqrt(den_i)
        sk_i = np.where(dPhiDrErr_i is not None,
                        dPhiDrErr_i / (den_i * np.sqrt(den_i)),
                        np.full_like(k_i, 1e-6))

        den_o = 1.0 + ro ** 2 * dPhiDr_o ** 2
        k_o = dPhiDr_o / np.sqrt(den_o)
        sk_o = np.where(dPhiDrErr_o is not None,
                        dPhiDrErr_o / (den_o * np.sqrt(den_o)),
                        np.full_like(k_o, 1e-6))

        denom = np.sqrt(sk_i ** 2 + sk_o ** 2)
        kappa_signif = np.where(denom > 1e-12,
                                np.abs(k_i - k_o) / denom,
                                0.0)
        result["kappa_signif"] = kappa_signif

    return result


# ============================================================================
# Plotting
# ============================================================================

CUT_VARIABLES = [
    ("abs_dphi", r"$|\Delta\phi|$ [rad]", None),
    ("idphi", r"$|\Delta\phi|$ [int16 units]", "phiCut"),
    ("dz", r"$\Delta z$ [cm]", None),
    ("z0", r"$z_0$ [cm]", "cellZ0Cut"),
    ("kappa_signif", r"$\kappa$ significance", "stubSigmaCut"),
]


def plot_distributions(pair_label, cuts, var_arrays, output_dir, containment=95.0):
    """Plot distributions of cut variables for one layer pair."""
    n_vars = len([v for v in CUT_VARIABLES if v[0] in var_arrays])
    if n_vars == 0:
        return

    fig, axes = plt.subplots(1, n_vars, figsize=(4.5 * n_vars, 4), squeeze=False)
    fig.suptitle(f"CA cut distributions: {pair_label} (N={len(next(iter(var_arrays.values())))})",
                 fontsize=13)

    col = 0
    for var_name, var_label, cut_key in CUT_VARIABLES:
        if var_name not in var_arrays:
            continue
        vals = np.array(var_arrays[var_name])
        ax = axes[0, col]

        # Remove NaN/inf
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            col += 1
            continue

        # Auto-range: use 1st and 99th percentiles
        lo, hi = np.percentile(vals, [1, 99])
        margin = 0.1 * (hi - lo) if hi > lo else 1.0
        # Cap idphi (uint16 units) plots at 10000
        if var_name == "idphi":
            hi = min(hi + margin, 10000)
            lo = max(lo - margin, 0)
            bins = np.linspace(lo, hi, 100)
        # Cap kappa significance at 25; overflow goes into last bin
        elif var_name == "kappa_signif":
            vals = np.clip(vals, 0, 25)
            bins = np.linspace(0, 25, 100)
        else:
            bins = np.linspace(lo - margin, hi + margin, 100)

        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.7, color="C0", label="Signal")
        ax.set_xlabel(var_label, fontsize=10)
        ax.set_ylabel("Entries", fontsize=10)

        # Overlay cut threshold
        if cut_key and cut_key in cuts and cuts[cut_key] > 0:
            threshold = cuts[cut_key]
            ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Cut = {threshold}")
            ax.axvline(-threshold, color="red", linestyle="--", linewidth=2) if var_name == "dz" else None
            # Compute efficiency
            if var_name in ("idphi", "abs_dphi", "kappa_signif", "z0"):
                eff = np.sum(vals <= threshold) / len(vals) * 100 if len(vals) > 0 else 0
                ax.text(0.95, 0.95, f"Eff: {eff:.1f}%", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.legend(fontsize=8)

        # Containment percentile line
        if containment > 0 and len(vals) > 0:
            if var_name in ("idphi", "abs_dphi", "kappa_signif", "z0"):
                # One-sided: containment% of signal below this value
                pct_val = np.percentile(vals, containment)
                ax.axvline(pct_val, color="green", linestyle="-.", linewidth=1.5,
                           label=f"{containment:.0f}% = {pct_val:.1f}")
            elif var_name == "dz":
                # Two-sided: symmetric containment
                lo_pct = (100 - containment) / 2
                hi_pct = 100 - lo_pct
                lo_val, hi_val = np.percentile(vals, [lo_pct, hi_pct])
                ax.axvline(lo_val, color="green", linestyle="-.", linewidth=1.5,
                           label=f"{containment:.0f}%: [{lo_val:.2f}, {hi_val:.2f}]")
                ax.axvline(hi_val, color="green", linestyle="-.", linewidth=1.5)
            ax.legend(fontsize=8)

        # Add statistics
        ax.text(0.95, 0.85, f"Mean: {np.mean(vals):.4f}\nRMS: {np.std(vals):.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

        ax.tick_params(labelsize=8)
        col += 1

    plt.tight_layout()
    safe_label = pair_label.replace("/", "_").replace(" ", "_")
    outpath = os.path.join(output_dir, f"ca_cuts_{safe_label}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_dz_2d(pair_label, dz_vals, tp_eta_vals, output_dir):
    """Plot dz vs TP eta for a layer pair."""
    if len(dz_vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(tp_eta_vals, dz_vals, s=1, alpha=0.3, c="C0")
    ax.set_xlabel(r"TP $\eta$", fontsize=10)
    ax.set_ylabel(r"$\Delta z$ [cm]", fontsize=10)
    ax.set_title(f"dz vs eta: {pair_label}", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_label = pair_label.replace("/", "_").replace(" ", "_")
    outpath = os.path.join(output_dir, f"dz_vs_eta_{safe_label}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CA connection cut tuning from true stub ntuple")
    parser.add_argument("rootfile", help="ROOT file from TrueStubNtupleAnalyzer")
    parser.add_argument("--output-dir", default="ca_cut_plots",
                        help="Directory for output plots (default: ca_cut_plots/)")
    parser.add_argument("--pairs", default="all",
                        choices=["ot-ot", "pix-ot", "pix-ot-b1", "pix-fwd-ot",
                                 "pix-bwd-ot", "all"],
                        help="Which layer pairs to analyze: ot-ot, pix-ot (all pixel-to-OT), "
                             "pix-ot-b1 (pixel to OT barrel L1 only), "
                             "pix-fwd-ot (fwd pixel to OT barrel+fwd disks), "
                             "pix-bwd-ot (bwd pixel to OT barrel+bwd disks), all")
    parser.add_argument("--max-events", type=int, default=-1,
                        help="Maximum number of events to process (-1 = all)")
    parser.add_argument("--tp-min-pt", type=float, default=0.0,
                        help="Minimum TP pT [GeV] (default: 0)")
    parser.add_argument("--tp-max-pt", type=float, default=1e9,
                        help="Maximum TP pT [GeV] (default: no cut)")
    parser.add_argument("--tp-min-eta", type=float, default=-1e9,
                        help="Minimum TP eta (default: no cut)")
    parser.add_argument("--tp-max-eta", type=float, default=1e9,
                        help="Maximum TP |eta| or eta (default: no cut)")
    parser.add_argument("--tp-max-vz", type=float, default=1e9,
                        help="Maximum TP |vz| [cm] (default: no cut)")
    parser.add_argument("--tp-max-d0", type=float, default=1e9,
                        help="Maximum TP |d0| [cm] (default: no cut)")
    parser.add_argument("--tp-min-lxy", type=float, default=0.0,
                        help="Minimum TP Lxy [cm] (default: 0)")
    parser.add_argument("--tp-max-lxy", type=float, default=1e9,
                        help="Maximum TP Lxy [cm] (default: no cut)")
    parser.add_argument("--containment", type=float, default=95.0,
                        help="Signal containment percentile to show on plots (default: 95)")
    args = parser.parse_args()

    if not os.path.exists(args.rootfile):
        print(f"ERROR: File not found: {args.rootfile}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Select which pairs to analyze
    if args.pairs == "ot-ot":
        active_pairs = ALL_OT_PAIRS
    elif args.pairs == "pix-ot":
        active_pairs = PIX_TO_OT_PAIRS
    elif args.pairs == "pix-ot-b1":
        active_pairs = PIX_B_TO_OT_B1_PAIRS + PIX_FWD_TO_OT_B1_PAIRS + PIX_BWD_TO_OT_B1_PAIRS
    elif args.pairs == "pix-fwd-ot":
        active_pairs = PIX_FWD_TO_OT_B1_PAIRS + PIX_FWD_TO_OT_FWD_PAIRS
    elif args.pairs == "pix-bwd-ot":
        active_pairs = PIX_BWD_TO_OT_B1_PAIRS + PIX_BWD_TO_OT_BWD_PAIRS
    else:
        active_pairs = ALL_PAIRS

    active_pair_keys = set()
    for inner_ca, outer_ca, _, _ in active_pairs:
        active_pair_keys.add((inner_ca, outer_ca))

    # Accumulate per-layer-pair distributions
    pair_vars = defaultdict(lambda: defaultdict(list))
    pair_dz_eta = defaultdict(lambda: ([], []))  # (dz_list, eta_list)

    print(f"Reading {args.rootfile}...")

    # Print applied TP cuts
    tp_cuts_active = []
    if args.tp_min_pt > 0:
        tp_cuts_active.append(f"pT > {args.tp_min_pt} GeV")
    if args.tp_max_pt < 1e8:
        tp_cuts_active.append(f"pT < {args.tp_max_pt} GeV")
    if args.tp_min_eta > -1e8:
        tp_cuts_active.append(f"eta > {args.tp_min_eta}")
    if args.tp_max_eta < 1e8:
        tp_cuts_active.append(f"|eta| < {args.tp_max_eta}")
    if args.tp_max_vz < 1e8:
        tp_cuts_active.append(f"|vz| < {args.tp_max_vz} cm")
    if args.tp_max_d0 < 1e8:
        tp_cuts_active.append(f"|d0| < {args.tp_max_d0} cm")
    if args.tp_min_lxy > 0:
        tp_cuts_active.append(f"Lxy > {args.tp_min_lxy} cm")
    if args.tp_max_lxy < 1e8:
        tp_cuts_active.append(f"Lxy < {args.tp_max_lxy} cm")
    if tp_cuts_active:
        print(f"  TP cuts: {', '.join(tp_cuts_active)}")
    else:
        print("  TP cuts: none (all TPs used)")

    # Determine tree path
    with uproot.open(args.rootfile) as f:
        tree_keys = [k for k in f.keys() if "stubs" in k.lower()]
        if not tree_keys:
            print("ERROR: No 'stubs' tree found. Available keys:", list(f.keys()), file=sys.stderr)
            sys.exit(1)
        tree_key = tree_keys[0]
        print(f"  Using tree: {tree_key}")

    # Branches to read
    branches = [
        "tp_pt", "tp_eta", "tp_phi", "tp_vz", "tp_d0", "tp_lxy",
        "pix_x", "pix_y", "pix_z", "pix_r", "pix_phi", "pix_caLayerId",
        "stub_x", "stub_y", "stub_z", "stub_r", "stub_phi",
        "stub_caLayerId", "stub_stubType",
        "stub_dPhiDr", "stub_dPhiDrError",
    ]

    n_tps_processed = 0
    n_doublets_formed = 0

    for batch in uproot.iterate(f"{args.rootfile}:{tree_key}", branches,
                                step_size=1000, library="np"):
        n_batch = len(batch["tp_pt"])

        for i in range(n_batch):
            if args.max_events > 0 and n_tps_processed >= args.max_events:
                break

            # Apply TP kinematic cuts
            tp_pt_val = float(batch["tp_pt"][i])
            tp_eta_val = float(batch["tp_eta"][i])
            tp_vz_val = float(batch["tp_vz"][i])
            tp_d0_val = float(batch["tp_d0"][i])
            tp_lxy_val = float(batch["tp_lxy"][i])

            if tp_pt_val < args.tp_min_pt or tp_pt_val > args.tp_max_pt:
                continue
            if tp_eta_val < args.tp_min_eta or abs(tp_eta_val) > args.tp_max_eta:
                continue
            if abs(tp_vz_val) > args.tp_max_vz:
                continue
            if abs(tp_d0_val) > args.tp_max_d0:
                continue
            if tp_lxy_val < args.tp_min_lxy or tp_lxy_val > args.tp_max_lxy:
                continue

            n_tps_processed += 1
            tp_eta = tp_eta_val

            # Collect all "CA hits" for this TP: pixel hits + OT stubs
            # Each hit: (caLayerId, r, z, phi, is_stub, dPhiDr, dPhiDrErr, stubType)
            hits = []

            # Pixel hits
            pix_x = batch["pix_x"][i]
            pix_y = batch["pix_y"][i]
            pix_z = batch["pix_z"][i]
            pix_r = batch["pix_r"][i]
            pix_phi = batch["pix_phi"][i]
            pix_layer = batch["pix_caLayerId"][i]
            for j in range(len(pix_x)):
                hits.append((int(pix_layer[j]), float(pix_r[j]), float(pix_z[j]),
                              float(pix_phi[j]), False, 0.0, 0.0, -1))

            # OT stubs
            s_x = batch["stub_x"][i]
            s_y = batch["stub_y"][i]
            s_z = batch["stub_z"][i]
            s_r = batch["stub_r"][i]
            s_phi = batch["stub_phi"][i]
            s_layer = batch["stub_caLayerId"][i]
            s_type = batch["stub_stubType"][i]
            s_dPhiDr = batch["stub_dPhiDr"][i]
            s_dPhiDrErr = batch["stub_dPhiDrError"][i]
            for j in range(len(s_x)):
                hits.append((int(s_layer[j]), float(s_r[j]), float(s_z[j]),
                              float(s_phi[j]), True, float(s_dPhiDr[j]),
                              float(s_dPhiDrErr[j]), int(s_type[j])))

            # Sort by CA layer ID
            hits.sort(key=lambda h: h[0])

            # Form doublets between valid layer pairs
            for a_idx in range(len(hits)):
                for b_idx in range(a_idx + 1, len(hits)):
                    a = hits[a_idx]
                    b = hits[b_idx]
                    pair_key = (a[0], b[0])

                    if pair_key not in active_pair_keys:
                        continue

                    ri, zi, phi_i = a[1], a[2], a[3]
                    ro, zo, phi_o = b[1], b[2], b[3]

                    dphi = phi_o - phi_i
                    if dphi > np.pi:
                        dphi -= 2 * np.pi
                    if dphi < -np.pi:
                        dphi += 2 * np.pi

                    dr = ro - ri
                    dz = zo - zi
                    z0_cross = abs(zi * ro - ri * zo)
                    z0_val = z0_cross / abs(dr) if abs(dr) > 1e-6 else 0.0
                    idphi_val = abs(dphi) * 32768.0 / np.pi

                    pair_vars[pair_key]["abs_dphi"].append(abs(dphi))
                    pair_vars[pair_key]["idphi"].append(idphi_val)
                    pair_vars[pair_key]["dz"].append(dz)
                    pair_vars[pair_key]["z0"].append(z0_val)
                    pair_vars[pair_key]["dr"].append(dr)

                    # Kappa significance (only when both are stubs with dPhiDr)
                    if a[4] and b[4] and a[7] != 2 and b[7] != 2:
                        dPhiDr_a, dPhiDrErr_a = a[5], a[6]
                        dPhiDr_b, dPhiDrErr_b = b[5], b[6]

                        den_a = 1.0 + ri ** 2 * dPhiDr_a ** 2
                        k_a = dPhiDr_a / np.sqrt(den_a)
                        sk_a = dPhiDrErr_a / (den_a * np.sqrt(den_a)) if dPhiDrErr_a > 0 else 1e-6

                        den_b = 1.0 + ro ** 2 * dPhiDr_b ** 2
                        k_b = dPhiDr_b / np.sqrt(den_b)
                        sk_b = dPhiDrErr_b / (den_b * np.sqrt(den_b)) if dPhiDrErr_b > 0 else 1e-6

                        denom = np.sqrt(sk_a ** 2 + sk_b ** 2)
                        kappa_sig = abs(k_a - k_b) / denom if denom > 1e-12 else 0.0
                        pair_vars[pair_key]["kappa_signif"].append(kappa_sig)

                    # Store dz vs eta for 2D plots
                    pair_dz_eta[pair_key][0].append(dz)
                    pair_dz_eta[pair_key][1].append(tp_eta)

                    n_doublets_formed += 1

        if args.max_events > 0 and n_tps_processed >= args.max_events:
            break

    print(f"\n  TPs processed: {n_tps_processed}")
    print(f"  Doublets formed: {n_doublets_formed}")
    print(f"  Active layer pairs with data: {len(pair_vars)}")

    # Print summary table
    print(f"\n{'Pair':<20} {'N doublets':>10} {'<|dphi|>':>10} {'<|dz|>':>10} {'<z0>':>10} {'<kappa>':>10}")
    print("-" * 72)
    for inner_ca, outer_ca, label, cuts in active_pairs:
        key = (inner_ca, outer_ca)
        if key not in pair_vars:
            continue
        n = len(pair_vars[key]["abs_dphi"])
        mean_dphi = np.mean(pair_vars[key]["abs_dphi"]) if n > 0 else 0
        mean_dz = np.mean(np.abs(pair_vars[key]["dz"])) if n > 0 else 0
        mean_z0 = np.mean(pair_vars[key]["z0"]) if n > 0 else 0
        mean_kappa = np.mean(pair_vars[key].get("kappa_signif", [0])) if n > 0 else 0
        print(f"{label:<20} {n:>10} {mean_dphi:>10.5f} {mean_dz:>10.3f} {mean_z0:>10.3f} {mean_kappa:>10.3f}")

    # Generate plots
    print(f"\nGenerating plots in {args.output_dir}/...")
    for inner_ca, outer_ca, label, cuts in active_pairs:
        key = (inner_ca, outer_ca)
        if key not in pair_vars or len(pair_vars[key]["abs_dphi"]) == 0:
            continue

        cuts_with_z0 = dict(cuts, cellZ0Cut=CELL_Z0_CUT)
        plot_distributions(label, cuts_with_z0, pair_vars[key], args.output_dir, args.containment)

        dz_list, eta_list = pair_dz_eta[key]
        if len(dz_list) > 100:
            plot_dz_2d(label, np.array(dz_list), np.array(eta_list), args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
