#!/usr/bin/env python3
"""
OT Stub Ntuple Analysis Script

This script reads the ROOT TTree produced by OTStubNtupleAnalyzer and creates
distributions useful for understanding dPhiDr, dPhiDz, and stub compatibility
in different detector regions.

Usage in Jupyter notebook:
    %run analyze_stub_ntuple.py

Or import specific functions:
    from analyze_stub_ntuple import load_stubs, plot_dPhiDr_comparison

Available plotting functions:
    - plot_dPhiDr_comparison: Compare dPhiDr distributions barrel vs endcap
    - plot_dPhiDz_comparison: Compare dPhiDz distributions barrel vs endcap
    - plot_eta_consistency: Plot eta consistency between inner/outer hits
    - plot_bend_and_pt: Plot bend and pT distributions
    - plot_geometry: Plot stub positions in various projections
    - plot_sensor_separation_details: Detailed sensor separation plots
    - plot_all_distance_definitions: All distance/separation definitions (dr, dz, d3, sensorSeparation, dr_effective)
    - plot_deltaPhi_distributions: DeltaPhi quantities (dphi_raw, dPhiDr, dPhiDrError, dPhiDz) by region
    - make_layer_plots: Per-layer distributions for barrel/forward+/forward-
    - make_width_plots: Per-layer width and parallax correction distributions
    - plot_compatibility_cuts: CA doublet/triplet compatibility cut variables
    - plot_layer_vs_width: 2D layer vs width (parallax-corrected) scatter plots
    - plot_layer_vs_width_raw: 2D layer vs width_raw (no parallax correction) scatter plots
    - plot_layer_vs_width_comparison: Combined side-by-side comparison of width and width_raw
    - plot_xy_angle: XY azimuthal angle distributions
    - plot_consecutive_layer_dotproduct: Scalar products between consecutive layers
    - plot_proposed_compatibility: Proposed quantities for the three geometry cases
    - plot_cuts_per_layer_pair: Cut variable distributions per layer pair for CA compatibility
    - plot_ca_tuning_variables: Per-layer distributions of CA cut tuning variables (dr_sensor, dz_sensor, dPhiDr, dPhiDz, eta_diff)
    - plot_ca_layer_pair_variables: Per-layer-PAIR distributions for CA doublet building cuts (stub_z, stub_r, dphi_raw, dz_sensor)
    - plot_kappa_comparison: Compare naive dPhiDr vs curvature-corrected kappa significance for barrel stub linking
    - plot_forward_dPhiDz_significance: Pairwise |dPhiDz| significance for forward endcap consecutive disk pairs
    - plot_pairwise_significance_by_transition: Pairwise dPhiDr and dPhiDz significance broken down by module transition type
    - plot_event_stubs: Visualize stub segments for a specific event (manual use only)

Utility functions:
    - compute_kappa(dPhiDr, r): Extract radius-independent half-curvature from dPhiDr
    - compute_kappa_error(dPhiDrError, dPhiDr, r): Propagate dPhiDr error to kappa error
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot

# Default file path - modify as needed
DEFAULT_FILE = "stubs.root"
DEFAULT_TREE = "otStubNtupleAnalyzer/stubs"


def load_stubs(filename=DEFAULT_FILE, treename=DEFAULT_TREE, exclude_phitonly=True):
    """Load stub data from ROOT file into a dictionary of numpy arrays.

    Args:
        filename: Path to the ROOT file.
        treename: Name of the TTree inside the ROOT file.
        exclude_phitonly: If True (default), exclude PHitOnly stubs (stubType==2)
            which have no valid outer hit. Their outer_x/y/z fields are zero and
            derived quantities (dr_sensor, dphi_raw, width, etc.) are meaningless.
            Set to False to include all stubs (use 'hasOuterHit' branch to filter).
    """
    with uproot.open(filename) as f:
        tree = f[treename]
        # Load all branches
        data = tree.arrays(library="np")
    if exclude_phitonly and "hasOuterHit" in data:
        mask = data["hasOuterHit"] == True
        data = {k: v[mask] for k, v in data.items()}
    return data


def select_barrel(data):
    """Select barrel stubs."""
    mask = data["isBarrel"] == True
    return {k: v[mask] for k, v in data.items()}


def select_endcap(data):
    """Select endcap stubs."""
    mask = data["isEndcap"] == True
    return {k: v[mask] for k, v in data.items()}


def select_forward(data):
    """Select forward endcap stubs (positive z)."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] > 0)
    return {k: v[mask] for k, v in data.items()}


def select_backward(data):
    """Select backward endcap stubs (negative z)."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] < 0)
    return {k: v[mask] for k, v in data.items()}


def select_tilted(data):
    """Select tilted barrel stubs."""
    mask = data["isTilted"] == True
    return {k: v[mask] for k, v in data.items()}


def select_flat(data):
    """Select flat barrel stubs."""
    mask = (data["isBarrel"] == True) & (data["isFlat"] == True)
    return {k: v[mask] for k, v in data.items()}


def select_layer(data, layer):
    """Select stubs from a specific layer."""
    mask = data["layer"] == layer
    return {k: v[mask] for k, v in data.items()}


def compute_kappa(dPhiDr, r):
    """Extract the radius-independent half-curvature kappa = 1/(2*rho) from dPhiDr.

    For a helix in a uniform B field, dPhiDr depends on the radius r where it is
    measured: |dPhiDr(r)| = kappa / sqrt(1 - (r*kappa)^2). The inverse gives:

        kappa = |dPhiDr| / sqrt(1 + r^2 * dPhiDr^2)

    This quantity equals 0.3*B/(2*pT) and is the same for all stubs on the same
    track, regardless of the layer radius. Comparing kappa values between stubs
    removes the systematic radial dependence that biases naive dPhiDr comparisons.

    Args:
        dPhiDr: array of dPhiDr values (rad/cm)
        r: array of stub radii (cm)

    Returns:
        kappa: array of half-curvature values (cm^-1), same for all layers on same track
    """
    dPhiDr2 = dPhiDr * dPhiDr
    return np.abs(dPhiDr) / np.sqrt(1.0 + r * r * dPhiDr2)


def compute_kappa_error(dPhiDrError, dPhiDr, r):
    """Propagate dPhiDrError to kappa error via the Jacobian.

    The Jacobian d(kappa)/d(dPhiDr) = 1 / (1 + r^2 * dPhiDr^2)^(3/2).

    Args:
        dPhiDrError: array of dPhiDr errors (rad/cm)
        dPhiDr: array of dPhiDr values (rad/cm)
        r: array of stub radii (cm)

    Returns:
        kappa_error: array of kappa errors (cm^-1)
    """
    dPhiDr2 = dPhiDr * dPhiDr
    factor = 1.0 + r * r * dPhiDr2
    return dPhiDrError / (factor * np.sqrt(factor))


def compute_pairwise_significance(data, mask1, mask2, val_key, err_key):
    """Compute pairwise significance for same-event stub pairs.

    For all stub pairs (i from mask1, j from mask2) in the same event, computes
    |val_i - val_j| / sqrt(err_i^2 + err_j^2).

    Args:
        data: dict of numpy arrays (from load_stubs)
        mask1: boolean mask selecting the first set of stubs
        mask2: boolean mask selecting the second set of stubs
        val_key: key in data for the value array (e.g. "dPhiDr", "dPhiDz")
        err_key: key in data for the error array (e.g. "dPhiDrError", "dPhiDzError")

    Returns:
        1-D numpy array of significance values
    """
    if err_key not in data or val_key not in data:
        return np.array([])

    events = data["event"]
    val_arr = data[val_key]
    err_arr = data[err_key]

    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    if len(idx1) == 0 or len(idx2) == 0:
        return np.array([])

    ev1, ev2 = events[idx1], events[idx2]
    v1, v2 = val_arr[idx1], val_arr[idx2]
    e1, e2 = err_arr[idx1], err_arr[idx2]

    unique_events = np.intersect1d(np.unique(ev1), np.unique(ev2))

    sig_list = []
    for evt in unique_events:
        m1 = ev1 == evt
        m2 = ev2 == evt
        d1, d2 = v1[m1], v2[m2]
        s1, s2 = e1[m1], e2[m2]
        for i in range(len(d1)):
            for j in range(len(d2)):
                combined_err = np.sqrt(s1[i]**2 + s2[j]**2)
                if combined_err > 0:
                    sig_list.append(np.abs(d1[i] - d2[j]) / combined_err)
    return np.array(sig_list)


def compute_pairwise_signed_pull(data, mask1, mask2, val_key, err_key):
    """Compute SIGNED pairwise pulls for same-event stub pairs.

    For all stub pairs (i from mask1, j from mask2) in the same event, computes
    (val_i - val_j) / sqrt(err_i^2 + err_j^2)  (signed, NOT absolute value).

    Also returns the signed difference (val_i - val_j) and combined error.

    Args:
        data: dict of numpy arrays (from load_stubs)
        mask1: boolean mask selecting the first set of stubs (inner disk)
        mask2: boolean mask selecting the second set of stubs (outer disk)
        val_key: key in data for the value array (e.g. "dPhiDr", "dPhiDz")
        err_key: key in data for the error array (e.g. "dPhiDrError", "dPhiDzError")

    Returns:
        dict with keys:
            "signed_diff": 1-D array of (val_i - val_j)
            "combined_err": 1-D array of sqrt(err_i^2 + err_j^2)
            "pull": 1-D array of signed pulls (val_i - val_j) / combined_err
            "val1": 1-D array of val_i values (inner disk)
            "val2": 1-D array of val_j values (outer disk)
    """
    empty = {"signed_diff": np.array([]), "combined_err": np.array([]),
             "pull": np.array([]), "val1": np.array([]), "val2": np.array([])}
    if err_key not in data or val_key not in data:
        return empty

    events = data["event"]
    val_arr = data[val_key]
    err_arr = data[err_key]

    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]
    if len(idx1) == 0 or len(idx2) == 0:
        return empty

    ev1, ev2 = events[idx1], events[idx2]
    v1, v2 = val_arr[idx1], val_arr[idx2]
    e1, e2 = err_arr[idx1], err_arr[idx2]

    unique_events = np.intersect1d(np.unique(ev1), np.unique(ev2))

    diff_list = []
    cerr_list = []
    pull_list = []
    val1_list = []
    val2_list = []
    for evt in unique_events:
        m1 = ev1 == evt
        m2 = ev2 == evt
        d1, d2 = v1[m1], v2[m2]
        s1, s2 = e1[m1], e2[m2]
        for i in range(len(d1)):
            for j in range(len(d2)):
                combined_err = np.sqrt(s1[i]**2 + s2[j]**2)
                if combined_err > 0:
                    diff = d1[i] - d2[j]
                    diff_list.append(diff)
                    cerr_list.append(combined_err)
                    pull_list.append(diff / combined_err)
                    val1_list.append(d1[i])
                    val2_list.append(d2[j])
    return {"signed_diff": np.array(diff_list),
            "combined_err": np.array(cerr_list),
            "pull": np.array(pull_list),
            "val1": np.array(val1_list),
            "val2": np.array(val2_list)}


def plot_dPhiDr_comparison(data, figsize=(16, 12)):
    """
    Plot dPhiDr distributions comparing barrel categories vs endcap.

    Barrel is split into 4 categories:
    - Flat non-flipped (blue)
    - Flat flipped (cornflowerblue)
    - Tilted non-flipped (orange)
    - Tilted flipped (goldenrod)

    Endcap is shown with the full distribution.
    Also shows dr_sensor and dz_sensor distributions.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # Create barrel category masks
    barrel_mask = data["isBarrel"] == True
    flat_mask = data["isFlat"] == True
    tilted_mask = data["isTilted"] == True
    flipped_mask = data["isFlipped"] == True

    # Barrel categories
    flat_nonflipped_mask = barrel_mask & flat_mask & (~flipped_mask)
    flat_flipped_mask = barrel_mask & flat_mask & flipped_mask
    tilted_nonflipped_mask = barrel_mask & tilted_mask & (~flipped_mask)
    tilted_flipped_mask = barrel_mask & tilted_mask & flipped_mask

    # Endcap mask
    endcap_mask = data["isEndcap"] == True

    # Define categories with their masks, colors, and labels
    barrel_categories = [
        (flat_nonflipped_mask, "blue", "Flat non-flipped"),
        (flat_flipped_mask, "cornflowerblue", "Flat flipped"),
        (tilted_nonflipped_mask, "orange", "Tilted non-flipped"),
        (tilted_flipped_mask, "goldenrod", "Tilted flipped"),
    ]

    # --- Row 0: dPhiDr histograms for barrel categories ---
    # Flat non-flipped
    ax = axes[0, 0]
    mask = flat_nonflipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDr"][mask]
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Flat Non-Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Flat flipped
    ax = axes[0, 1]
    mask = flat_flipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDr"][mask]
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="cornflowerblue",
            label=f"Flat flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Flat Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap (note different scale!)
    ax = axes[0, 2]
    n_stubs = endcap_mask.sum()
    values = data["dPhiDr"][endcap_mask]
    ax.hist(values, bins=100, range=(-1.0, 1.0),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Endcap (note 100x larger scale!)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: Tilted barrel and overlay comparison ---
    # Tilted non-flipped
    ax = axes[1, 0]
    mask = tilted_nonflipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDr"][mask]
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Tilted Non-Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Tilted flipped
    ax = axes[1, 1]
    mask = tilted_flipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDr"][mask]
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="goldenrod",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Tilted Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # All barrel categories overlayed
    ax = axes[1, 2]
    for mask, color, label in barrel_categories:
        n_stubs = mask.sum()
        ax.hist(data["dPhiDr"][mask], bins=100, range=(-0.01, 0.01),
                histtype="step", linewidth=1.5, color=color, alpha=0.8,
                label=f"{label} (N={n_stubs})")
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - All Barrel Categories")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Row 2: dr and dz sensor distributions ---
    # dr distribution - barrel categories
    ax = axes[2, 0]
    for mask, color, label in barrel_categories:
        ax.hist(data["dr_sensor"][mask], bins=50, range=(-5, 5),
                histtype="step", linewidth=1.5, color=color, alpha=0.8,
                label=label)
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor separation in r - Barrel")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # dr distribution - endcap
    ax = axes[2, 1]
    ax.hist(data["dr_sensor"][endcap_mask], bins=50, range=(-5, 5),
            histtype="step", linewidth=2, color="red", label="Endcap")
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor separation in r - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dz distribution - all regions
    ax = axes[2, 2]
    for mask, color, label in barrel_categories:
        ax.hist(data["dz_sensor"][mask], bins=50, range=(-5, 5),
                histtype="step", linewidth=1.5, color=color, alpha=0.8,
                label=label)
    ax.hist(data["dz_sensor"][endcap_mask], bins=50, range=(-5, 5),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label="Endcap")
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor separation in z")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_dPhiDz_comparison(data, figsize=(16, 12)):
    """
    Plot dPhiDz distributions - this is the proposed alternative for endcap.
    For endcap modules, dPhiDz should be well-behaved since dz is ~4mm.

    Barrel is split into 4 categories:
    - Flat non-flipped (blue)
    - Flat flipped (cornflowerblue)
    - Tilted non-flipped (orange)
    - Tilted flipped (goldenrod)

    Endcap is shown with the full distribution.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # Create barrel category masks
    barrel_mask = data["isBarrel"] == True
    flat_mask = data["isFlat"] == True
    tilted_mask = data["isTilted"] == True
    flipped_mask = data["isFlipped"] == True

    # Barrel categories
    flat_nonflipped_mask = barrel_mask & flat_mask & (~flipped_mask)
    flat_flipped_mask = barrel_mask & flat_mask & flipped_mask
    tilted_nonflipped_mask = barrel_mask & tilted_mask & (~flipped_mask)
    tilted_flipped_mask = barrel_mask & tilted_mask & flipped_mask

    # Endcap mask
    endcap_mask = data["isEndcap"] == True

    # Define categories with their masks, colors, and labels
    barrel_categories = [
        (flat_nonflipped_mask, "blue", "Flat non-flipped"),
        (flat_flipped_mask, "cornflowerblue", "Flat flipped"),
        (tilted_nonflipped_mask, "orange", "Tilted non-flipped"),
        (tilted_flipped_mask, "goldenrod", "Tilted flipped"),
    ]

    # --- Row 0: dPhiDz histograms for barrel categories ---
    # Flat non-flipped
    ax = axes[0, 0]
    mask = flat_nonflipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDz"][mask]
    ax.hist(values, bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Flat Non-Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Flat flipped
    ax = axes[0, 1]
    mask = flat_flipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDz"][mask]
    ax.hist(values, bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="cornflowerblue",
            label=f"Flat flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Flat Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap (note different scale!)
    ax = axes[0, 2]
    n_stubs = endcap_mask.sum()
    values = data["dPhiDz"][endcap_mask]
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Endcap (well-behaved!)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: Tilted barrel and overlay comparison ---
    # Tilted non-flipped
    ax = axes[1, 0]
    mask = tilted_nonflipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDz"][mask]
    ax.hist(values, bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Tilted Non-Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Tilted flipped
    ax = axes[1, 1]
    mask = tilted_flipped_mask
    n_stubs = mask.sum()
    values = data["dPhiDz"][mask]
    ax.hist(values, bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="goldenrod",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Tilted Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # All barrel categories overlayed
    ax = axes[1, 2]
    for mask, color, label in barrel_categories:
        n_stubs = mask.sum()
        ax.hist(data["dPhiDz"][mask], bins=100, range=(-0.1, 0.1),
                histtype="step", linewidth=1.5, color=color, alpha=0.8,
                label=f"{label} (N={n_stubs})")
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - All Barrel Categories")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Row 2: Comparison plots ---
    # Barrel vs Endcap comparison on same scale
    ax = axes[2, 0]
    for mask, color, label in barrel_categories:
        n_stubs = mask.sum()
        ax.hist(data["dPhiDz"][mask], bins=100, range=(-0.05, 0.05),
                histtype="step", linewidth=1.5, color=color, alpha=0.8,
                label=f"{label} (N={n_stubs})")
    ax.hist(data["dPhiDz"][endcap_mask], bins=100, range=(-0.05, 0.05),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label=f"Endcap (N={endcap_mask.sum()})")
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - All Regions Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Flat vs Tilted comparison (combined flipped states)
    ax = axes[2, 1]
    flat_mask_combined = barrel_mask & flat_mask
    tilted_mask_combined = barrel_mask & tilted_mask
    n_flat = flat_mask_combined.sum()
    n_tilted = tilted_mask_combined.sum()
    ax.hist(data["dPhiDz"][flat_mask_combined], bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={n_flat})")
    ax.hist(data["dPhiDz"][tilted_mask_combined], bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={n_tilted})")
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Flat vs Tilted Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Flipped vs Non-flipped comparison (combined flat/tilted)
    ax = axes[2, 2]
    flipped_combined = barrel_mask & flipped_mask
    nonflipped_combined = barrel_mask & (~flipped_mask)
    n_flipped = flipped_combined.sum()
    n_nonflipped = nonflipped_combined.sum()
    ax.hist(data["dPhiDz"][nonflipped_combined], bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="green", alpha=0.8,
            label=f"Non-flipped (N={n_nonflipped})")
    ax.hist(data["dPhiDz"][flipped_combined], bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, color="purple", alpha=0.8,
            label=f"Flipped (N={n_flipped})")
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Flipped vs Non-Flipped Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_eta_consistency(data, figsize=(14, 5)):
    """
    Plot eta (z/r) consistency between inner and outer hits.
    For valid stubs, the difference should be small (straight line in r-z).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    barrel = select_barrel(data)
    endcap = select_endcap(data)

    # Eta difference - barrel
    ax = axes[0]
    values = barrel["eta_diff"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.1, 0.1),
            histtype="step", linewidth=2, label=f"Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("eta_outer - eta_inner")
    ax.set_ylabel("Stubs")
    ax.set_title("Eta consistency - Barrel")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eta difference - endcap
    ax = axes[1]
    values = endcap["eta_diff"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.005, 0.005),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("eta_outer - eta_inner")
    ax.set_ylabel("Stubs")
    ax.set_title("Eta consistency - Endcap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eta vs r
    ax = axes[2]
    ax.scatter(data["stub_r"], data["eta_diff"], s=1, alpha=0.3)
    ax.set_xlabel("r [cm]")
    ax.set_ylabel("eta_outer - eta_inner")
    ax.set_title("Eta consistency vs radius")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bend_and_pt(data, figsize=(14, 10)):
    """
    Plot bend and pT distributions for barrel vs endcap.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    barrel = select_barrel(data)
    endcap = select_endcap(data)

    # Bend - barrel
    ax = axes[0, 0]
    values = barrel["bend"] * 1000
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, label=f"Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.3f}\nstd={np.std(values):.3f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("bend [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Bend - Barrel")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bend - endcap
    ax = axes[0, 1]
    values = endcap["bend"] * 1000
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.3f}\nstd={np.std(values):.3f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("bend [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Bend - Endcap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # pT estimate - barrel (log scale)
    ax = axes[1, 0]
    pt_barrel = barrel["ptEst"]
    pt_barrel = pt_barrel[(pt_barrel > 0) & (pt_barrel < 1e5)]
    n_stubs = len(pt_barrel)
    ax.hist(pt_barrel, bins=100, range=(0, 100),
            histtype="step", linewidth=2, label=f"Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(pt_barrel):.2f}\nstd={np.std(pt_barrel):.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("pT estimate [GeV]")
    ax.set_ylabel("Stubs")
    ax.set_title("pT estimate - Barrel")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # pT estimate - endcap
    ax = axes[1, 1]
    pt_endcap = endcap["ptEst"]
    pt_endcap = pt_endcap[(pt_endcap > 0) & (pt_endcap < 1e5)]
    n_stubs = len(pt_endcap)
    ax.hist(pt_endcap, bins=100, range=(0, 100),
            histtype="step", linewidth=2, color="red", label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(pt_endcap):.2f}\nstd={np.std(pt_endcap):.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("pT estimate [GeV]")
    ax.set_ylabel("Stubs")
    ax.set_title("pT estimate - Endcap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_geometry(data, figsize=(14, 10)):
    """
    Plot stub positions in various projections.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    barrel = select_barrel(data)
    endcap = select_endcap(data)

    # x-y view
    ax = axes[0, 0]
    ax.scatter(barrel["stub_x"], barrel["stub_y"], s=1, alpha=0.3, label="Barrel")
    ax.scatter(endcap["stub_x"], endcap["stub_y"], s=1, alpha=0.3, color="red", label="Endcap")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title("x-y projection")
    ax.set_aspect("equal")
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # r-z view
    ax = axes[0, 1]
    ax.scatter(barrel["stub_z"], barrel["stub_r"], s=1, alpha=0.3, label="Barrel")
    ax.scatter(endcap["stub_z"], endcap["stub_r"], s=1, alpha=0.3, color="red", label="Endcap")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("r [cm]")
    ax.set_title("r-z projection")
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)

    # Layer distribution
    ax = axes[1, 0]
    ax.hist(barrel["layer"], bins=np.arange(0.5, 7.5, 1),
            histtype="step", linewidth=2, label="Barrel")
    ax.hist(endcap["layer"], bins=np.arange(0.5, 6.5, 1),
            histtype="step", linewidth=2, color="red", label="Endcap")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Stubs")
    ax.set_title("Layer distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CA layer ID distribution
    ax = axes[1, 1]
    ax.hist(data["caLayerId"], bins=np.arange(27.5, 44.5, 1),
            histtype="step", linewidth=2)
    ax.set_xlabel("CA Layer ID")
    ax.set_ylabel("Stubs")
    ax.set_title("CA Layer ID distribution")
    ax.axvline(x=33.5, color="gray", linestyle="--", alpha=0.5, label="Barrel/Endcap boundary")
    ax.axvline(x=38.5, color="gray", linestyle=":", alpha=0.5, label="Backward/Forward boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sensor_separation_details(data, figsize=(14, 10)):
    """
    Detailed plots of sensor separation to understand geometry.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    barrel = select_barrel(data)
    endcap = select_endcap(data)
    tilted = select_tilted(data)
    flat = select_flat(data)

    # dr vs layer - barrel
    ax = axes[0, 0]
    for layer in range(1, 7):
        layer_data = select_layer(barrel, layer)
        if len(layer_data["dr_sensor"]) > 0:
            ax.hist(layer_data["dr_sensor"], bins=50, range=(-1, 5),
                    histtype="step", linewidth=1.5, label=f"L{layer}")
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Barrel: dr by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dz vs layer - barrel
    ax = axes[0, 1]
    for layer in range(1, 7):
        layer_data = select_layer(barrel, layer)
        if len(layer_data["dz_sensor"]) > 0:
            ax.hist(layer_data["dz_sensor"], bins=50, range=(-1, 5),
                    histtype="step", linewidth=1.5, label=f"L{layer}")
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Barrel: dz by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Flat vs tilted
    ax = axes[0, 2]
    if len(flat["dr_sensor"]) > 0:
        ax.hist(flat["dr_sensor"], bins=50, range=(-1, 5),
                histtype="step", linewidth=2, label="Flat")
    if len(tilted["dr_sensor"]) > 0:
        ax.hist(tilted["dr_sensor"], bins=50, range=(-1, 5),
                histtype="step", linewidth=2, color="orange", label="Tilted")
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Barrel: Flat vs Tilted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # dr vs layer - endcap
    ax = axes[1, 0]
    for disk in range(1, 6):
        disk_data = select_layer(endcap, disk)
        if len(disk_data["dr_sensor"]) > 0:
            ax.hist(disk_data["dr_sensor"], bins=50, range=(-0.5, 0.5),
                    histtype="step", linewidth=1.5, label=f"D{disk}")
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Endcap: dr by disk (note small range!)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dz vs layer - endcap
    ax = axes[1, 1]
    for disk in range(1, 6):
        disk_data = select_layer(endcap, disk)
        if len(disk_data["dr_sensor"]) > 0:
            ax.hist(disk_data["dz_sensor"], bins=50, range=(-1, 1),
                    histtype="step", linewidth=1.5, label=f"D{disk}")
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Endcap: dz by disk")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3D separation distribution
    ax = axes[1, 2]
    ax.hist(barrel["d3_sensor"], bins=50, range=(0, 5),
            histtype="step", linewidth=2, label="Barrel")
    ax.hist(endcap["d3_sensor"], bins=50, range=(0, 5),
            histtype="step", linewidth=2, color="red", label="Endcap")
    ax.set_xlabel("3D sensor separation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("3D sensor separation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all_distance_definitions(data, figsize=(26, 24)):
    """
    Comprehensive plot of all distance/separation definitions available in the ntuple.

    This function creates histograms for:
    - dr_sensor: radial separation (outer_r - inner_r)
    - dz_sensor: z separation (outer_z - inner_z)
    - d3_sensor: 3D separation (Euclidean distance)
    - sensorSeparation: nominal separation from geometry (in mm, converted to cm)
    - dr_effective: effective dr from L1 trigger formula

    For each variable, plots are created for:
    - Barrel: flat, tilted non-flipped, and tilted flipped separately
    - Endcap
    """
    fig, axes = plt.subplots(5, 4, figsize=figsize)

    # Create masks
    flat_barrel = data["isBarrel"] & data["isFlat"]
    tilted_barrel = data["isBarrel"] & ~data["isFlat"]
    tilted_nonflipped = tilted_barrel & ~data["isFlipped"]
    tilted_flipped = tilted_barrel & data["isFlipped"]
    endcap = ~data["isBarrel"]

    # --- Row 0: dr_sensor (radial separation) ---
    # Barrel flat
    ax = axes[0, 0]
    values = data["dr_sensor"][flat_barrel]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 5), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_sensor - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[0, 1]
    values = data["dr_sensor"][tilted_nonflipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 5), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_sensor - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[0, 2]
    values = data["dr_sensor"][tilted_flipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 5), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_sensor - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[0, 3]
    values = data["dr_sensor"][endcap]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.5, 0.5), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_sensor - Endcap (note small range)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: dz_sensor (z separation) ---
    # Barrel flat
    ax = axes[1, 0]
    values = data["dz_sensor"][flat_barrel]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 1), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dz_sensor - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[1, 1]
    values = data["dz_sensor"][tilted_nonflipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-5, 5), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dz_sensor - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[1, 2]
    values = data["dz_sensor"][tilted_flipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-5, 5), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dz_sensor - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[1, 3]
    values = data["dz_sensor"][endcap]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 1), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dz_sensor - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 2: d3_sensor (3D separation) ---
    # Barrel flat
    ax = axes[2, 0]
    values = data["d3_sensor"][flat_barrel]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 5), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("d3_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("d3_sensor (3D) - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[2, 1]
    values = data["d3_sensor"][tilted_nonflipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 5), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("d3_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("d3_sensor (3D) - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[2, 2]
    values = data["d3_sensor"][tilted_flipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 5), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("d3_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("d3_sensor (3D) - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[2, 3]
    values = data["d3_sensor"][endcap]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 1), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("d3_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("d3_sensor (3D) - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 3: sensorSeparation (geometry nominal, mm -> cm) ---
    # Barrel flat
    ax = axes[3, 0]
    # sensorSeparation is in mm, convert to cm for plotting
    values = data["sensorSeparation"][flat_barrel] * 0.1  # mm to cm
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.5), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("sensorSeparation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("sensorSeparation (geometry) - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[3, 1]
    values = data["sensorSeparation"][tilted_nonflipped] * 0.1  # mm to cm
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.5), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("sensorSeparation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("sensorSeparation (geometry) - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[3, 2]
    values = data["sensorSeparation"][tilted_flipped] * 0.1  # mm to cm
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.5), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("sensorSeparation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("sensorSeparation (geometry) - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[3, 3]
    values = data["sensorSeparation"][endcap] * 0.1  # mm to cm
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.5), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("sensorSeparation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("sensorSeparation (geometry) - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 4: dr_effective ---
    # Note: dr_effective = separation / (cosTilt + sinTilt * z/r)
    # tiltAngle = atan2(dz, dr) measured from radial axis, so sinTilt > 0 for +z tilt
    # For flat barrel: sinTilt ~ 0, denominator ~ 1, so dr_effective ~ separation_cm (0.1-0.4 cm)
    # For tilted barrel: denominator > 1, so dr_effective < separation_cm
    # For endcap: cosTilt ~ 0, sinTilt ~ ±1, denominator ~ ±z/r

    # Barrel flat
    ax = axes[4, 0]
    raw_values = data["dr_effective"][flat_barrel]
    # Filter out non-finite values (inf, nan) for robust plotting
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.5, 0.5), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_effective [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_effective (L1 formula) - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped - use wider range since denominator < 1 makes values larger
    # Allow negative values since formula can produce negative results for certain orientations
    ax = axes[4, 1]
    raw_values = data["dr_effective"][tilted_nonflipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 2.5), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_effective [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_effective (L1 formula) - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[4, 2]
    raw_values = data["dr_effective"][tilted_flipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 2.5), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_effective [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_effective (L1 formula) - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap - denominator = cosTilt + sinTilt*z/r ~ sinTilt*z/r (cosTilt ~ 0)
    ax = axes[4, 3]
    raw_values = data["dr_effective"][endcap]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-1, 2.5), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.4f} cm\nstd={np.std(values):.4f} cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_effective [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dr_effective (L1 formula) - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("All Distance/Separation Definitions", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_deltaPhi_distributions(data, figsize=(26, 30)):
    """
    Plot deltaPhi-related quantities across detector regions.

    This function creates histograms for:
    - dphi_raw: raw phi difference (outer_phi - inner_phi) with wraparound handling
    - dphi_parallax: parallax contribution to dphi (if available)
    - dphi_corrected: dphi_raw minus parallax correction (if available)
    - dPhiDr: deltaPhi/dr from stub (L1 trigger quantity)
    - dPhiDrError: error on dPhiDr
    - dPhiDz: deltaPhi/dz (for endcap analysis)

    For each variable, plots are created for:
    - Barrel: flat, tilted non-flipped, and tilted flipped separately
    - Endcap

    Note: dphi_parallax and dphi_corrected may not be present in older ntuples.
    If these columns are missing, the corresponding rows will show "Data not available".
    """
    fig, axes = plt.subplots(6, 4, figsize=figsize)

    # Check for optional columns (for backward compatibility with older ntuples)
    has_dphi_parallax = "dphi_parallax" in data.columns if hasattr(data, "columns") else "dphi_parallax" in data
    has_dphi_corrected = "dphi_corrected" in data.columns if hasattr(data, "columns") else "dphi_corrected" in data

    # Create masks
    flat_barrel = data["isBarrel"] & data["isFlat"]
    tilted_barrel = data["isBarrel"] & ~data["isFlat"]
    tilted_nonflipped = tilted_barrel & ~data["isFlipped"]
    tilted_flipped = tilted_barrel & data["isFlipped"]
    endcap = ~data["isBarrel"]

    # --- Row 0: dphi_raw (raw phi difference) ---
    # Barrel flat
    ax = axes[0, 0]
    values = data["dphi_raw"][flat_barrel]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dphi_raw [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("dphi_raw - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[0, 1]
    values = data["dphi_raw"][tilted_nonflipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dphi_raw [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("dphi_raw - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[0, 2]
    values = data["dphi_raw"][tilted_flipped]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dphi_raw [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("dphi_raw - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[0, 3]
    values = data["dphi_raw"][endcap]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dphi_raw [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("dphi_raw - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: dphi_parallax (parallax contribution to dphi) ---
    if has_dphi_parallax:
        # Barrel flat
        ax = axes[1, 0]
        values = data["dphi_parallax"][flat_barrel]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.005, 0.005), histtype="step", linewidth=2, color="blue",
                label=f"Flat barrel (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_parallax [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_parallax - Flat Barrel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Barrel tilted non-flipped
        ax = axes[1, 1]
        values = data["dphi_parallax"][tilted_nonflipped]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.005, 0.005), histtype="step", linewidth=2, color="orange",
                label=f"Tilted non-flipped (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_parallax [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_parallax - Tilted Non-Flipped")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Barrel tilted flipped
        ax = axes[1, 2]
        values = data["dphi_parallax"][tilted_flipped]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.005, 0.005), histtype="step", linewidth=2, color="green",
                label=f"Tilted flipped (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_parallax [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_parallax - Tilted Flipped")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Endcap
        ax = axes[1, 3]
        values = data["dphi_parallax"][endcap]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.005, 0.005), histtype="step", linewidth=2, color="red",
                label=f"Endcap (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_parallax [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_parallax - Endcap")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # dphi_parallax not available - show placeholder message
        for col in range(4):
            ax = axes[1, col]
            ax.text(0.5, 0.5, "Data not available\n(dphi_parallax not in ntuple)",
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment="center", horizontalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
            ax.set_xlabel("dphi_parallax [rad]")
            ax.set_ylabel("Stubs")
            titles = ["Flat Barrel", "Tilted Non-Flipped", "Tilted Flipped", "Endcap"]
            ax.set_title(f"dphi_parallax - {titles[col]}")
            ax.grid(True, alpha=0.3)

    # --- Row 2: dphi_corrected (dphi_raw minus parallax) ---
    if has_dphi_corrected:
        # Barrel flat
        ax = axes[2, 0]
        values = data["dphi_corrected"][flat_barrel]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="blue",
                label=f"Flat barrel (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_corrected [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_corrected - Flat Barrel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Barrel tilted non-flipped
        ax = axes[2, 1]
        values = data["dphi_corrected"][tilted_nonflipped]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="orange",
                label=f"Tilted non-flipped (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_corrected [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_corrected - Tilted Non-Flipped")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Barrel tilted flipped
        ax = axes[2, 2]
        values = data["dphi_corrected"][tilted_flipped]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="green",
                label=f"Tilted flipped (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_corrected [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_corrected - Tilted Flipped")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Endcap
        ax = axes[2, 3]
        values = data["dphi_corrected"][endcap]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(-0.01, 0.01), histtype="step", linewidth=2, color="red",
                label=f"Endcap (N={n_stubs})")
        if n_stubs > 0:
            stats_text = f"mean={np.mean(values):.6f} rad\nstd={np.std(values):.6f} rad"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dphi_corrected [rad]")
        ax.set_ylabel("Stubs")
        ax.set_title("dphi_corrected - Endcap")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # dphi_corrected not available - show placeholder message
        for col in range(4):
            ax = axes[2, col]
            ax.text(0.5, 0.5, "Data not available\n(dphi_corrected not in ntuple)",
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment="center", horizontalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
            ax.set_xlabel("dphi_corrected [rad]")
            ax.set_ylabel("Stubs")
            titles = ["Flat Barrel", "Tilted Non-Flipped", "Tilted Flipped", "Endcap"]
            ax.set_title(f"dphi_corrected - {titles[col]}")
            ax.grid(True, alpha=0.3)

    # --- Row 3: dPhiDr (L1 trigger quantity) ---
    # Barrel flat
    ax = axes[3, 0]
    raw_values = data["dPhiDr"][flat_barrel]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr (L1 trigger) - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[3, 1]
    raw_values = data["dPhiDr"][tilted_nonflipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr (L1 trigger) - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[3, 2]
    raw_values = data["dPhiDr"][tilted_flipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr (L1 trigger) - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[3, 3]
    raw_values = data["dPhiDr"][endcap]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr (L1 trigger) - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 4: dPhiDrError ---
    # Barrel flat
    ax = axes[4, 0]
    raw_values = data["dPhiDrError"][flat_barrel]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.01), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[4, 1]
    raw_values = data["dPhiDrError"][tilted_nonflipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.01), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[4, 2]
    raw_values = data["dPhiDrError"][tilted_flipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.01), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[4, 3]
    raw_values = data["dPhiDrError"][endcap]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(0, 0.01), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 5: dPhiDz (proposed for endcap analysis) ---
    # Barrel flat
    ax = axes[5, 0]
    raw_values = data["dPhiDz"][flat_barrel]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="blue",
            label=f"Flat barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted non-flipped
    ax = axes[5, 1]
    raw_values = data["dPhiDz"][tilted_nonflipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="orange",
            label=f"Tilted non-flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Tilted Non-Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Barrel tilted flipped
    ax = axes[5, 2]
    raw_values = data["dPhiDz"][tilted_flipped]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="green",
            label=f"Tilted flipped (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Tilted Flipped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap
    ax = axes[5, 3]
    raw_values = data["dPhiDz"][endcap]
    values = raw_values[np.isfinite(raw_values)]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.02, 0.02), histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"mean={np.mean(values):.6f} rad/cm\nstd={np.std(values):.6f} rad/cm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDz - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("DeltaPhi Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def select_barrel_layer(data, layer):
    """Select barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_forward_disk(data, disk):
    """Select forward+ (positive z) endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def select_backward_disk(data, disk):
    """Select forward- (negative z) endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def select_flat_barrel_layer(data, layer):
    """Select flat barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isFlat"] == True) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_tilted_barrel_layer(data, layer):
    """Select tilted barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isTilted"] == True) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_flat_nonflipped_barrel_layer(data, layer):
    """Select flat non-flipped barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isFlat"] == True) & (data["isFlipped"] == False) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_flat_flipped_barrel_layer(data, layer):
    """Select flat flipped barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isFlat"] == True) & (data["isFlipped"] == True) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_tilted_nonflipped_barrel_layer(data, layer):
    """Select tilted non-flipped barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isTilted"] == True) & (data["isFlipped"] == False) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_tilted_flipped_barrel_layer(data, layer):
    """Select tilted flipped barrel stubs from a specific layer."""
    mask = (data["isBarrel"] == True) & (data["isTilted"] == True) & (data["isFlipped"] == True) & (data["layer"] == layer)
    return {k: v[mask] for k, v in data.items()}


def select_forward_nonflipped_disk(data, disk):
    """Select forward+ (positive z) non-flipped endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["isFlipped"] == False) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def select_forward_flipped_disk(data, disk):
    """Select forward+ (positive z) flipped endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["isFlipped"] == True) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def select_backward_nonflipped_disk(data, disk):
    """Select forward- (negative z) non-flipped endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["isFlipped"] == False) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def select_backward_flipped_disk(data, disk):
    """Select forward- (negative z) flipped endcap stubs from a specific disk."""
    mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["isFlipped"] == True) & (data["layer"] == disk)
    return {k: v[mask] for k, v in data.items()}


def make_layer_plots(data, figsize=(16, 12)):
    """
    Create per-layer distribution plots for barrel and endcap regions.

    Generates separate plots for 8 categories:
    - Flat non-flipped barrel layers 1-6
    - Flat flipped barrel layers 1-6
    - Tilted non-flipped barrel layers 1-6
    - Tilted flipped barrel layers 1-6
    - Forward+ non-flipped disks 1-5
    - Forward+ flipped disks 1-5
    - Forward- non-flipped disks 1-5
    - Forward- flipped disks 1-5

    Each subplot shows dPhiDr, dPhiDz, bend, and pT distributions.

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        Dictionary of matplotlib figures keyed by region name
    """
    figs = {}

    # Define variables and their plot ranges
    variables = [
        ("dPhiDr", "dPhiDr [rad/cm]", (-0.01, 0.01), (-0.5, 0.5)),  # (barrel range, endcap range)
        ("dPhiDz", "dPhiDz [rad/cm]", (-0.1, 0.1), (-0.01, 0.01)),
        ("bend", "bend [mrad]", (-5, 5), (-5, 5)),  # will multiply by 1000
        ("ptEst", "pT estimate [GeV]", (0, 100), (0, 100)),
    ]

    # --- Flat Non-Flipped Barrel layers ---
    fig_flat_nonflipped_barrel, axes_flat_nonflipped_barrel = plt.subplots(4, 6, figsize=figsize)
    fig_flat_nonflipped_barrel.suptitle("Flat Non-Flipped Barrel Layer Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_flat_nonflipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _) in enumerate(variables):
            ax = axes_flat_nonflipped_barrel[var_idx, layer - 1]

            if n_stubs > 0:
                values = layer_data[var_name]
                if var_name == "bend":
                    values = values * 1000  # Convert to mrad
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="blue")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_flat_nonflipped_barrel.tight_layout()
    figs["flat_nonflipped_barrel_layers"] = fig_flat_nonflipped_barrel

    # --- Flat Flipped Barrel layers ---
    fig_flat_flipped_barrel, axes_flat_flipped_barrel = plt.subplots(4, 6, figsize=figsize)
    fig_flat_flipped_barrel.suptitle("Flat Flipped Barrel Layer Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_flat_flipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _) in enumerate(variables):
            ax = axes_flat_flipped_barrel[var_idx, layer - 1]

            if n_stubs > 0:
                values = layer_data[var_name]
                if var_name == "bend":
                    values = values * 1000  # Convert to mrad
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="cornflowerblue")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_flat_flipped_barrel.tight_layout()
    figs["flat_flipped_barrel_layers"] = fig_flat_flipped_barrel

    # --- Tilted Non-Flipped Barrel layers ---
    fig_tilted_nonflipped_barrel, axes_tilted_nonflipped_barrel = plt.subplots(4, 6, figsize=figsize)
    fig_tilted_nonflipped_barrel.suptitle("Tilted Non-Flipped Barrel Layer Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_tilted_nonflipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _) in enumerate(variables):
            ax = axes_tilted_nonflipped_barrel[var_idx, layer - 1]

            if n_stubs > 0:
                values = layer_data[var_name]
                if var_name == "bend":
                    values = values * 1000  # Convert to mrad
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="orange")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_tilted_nonflipped_barrel.tight_layout()
    figs["tilted_nonflipped_barrel_layers"] = fig_tilted_nonflipped_barrel

    # --- Tilted Flipped Barrel layers ---
    fig_tilted_flipped_barrel, axes_tilted_flipped_barrel = plt.subplots(4, 6, figsize=figsize)
    fig_tilted_flipped_barrel.suptitle("Tilted Flipped Barrel Layer Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_tilted_flipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _) in enumerate(variables):
            ax = axes_tilted_flipped_barrel[var_idx, layer - 1]

            if n_stubs > 0:
                values = layer_data[var_name]
                if var_name == "bend":
                    values = values * 1000  # Convert to mrad
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="goldenrod")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_tilted_flipped_barrel.tight_layout()
    figs["tilted_flipped_barrel_layers"] = fig_tilted_flipped_barrel

    # --- Forward+ Non-Flipped disks ---
    fig_fwd_pos_nonflipped, axes_fwd_pos_nonflipped = plt.subplots(4, 5, figsize=(14, 12))
    fig_fwd_pos_nonflipped.suptitle("Forward+ (z>0) Non-Flipped Disk Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_forward_nonflipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range) in enumerate(variables):
            ax = axes_fwd_pos_nonflipped[var_idx, disk - 1]

            if n_stubs > 0:
                values = disk_data[var_name]
                if var_name == "bend":
                    values = values * 1000
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="red")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_pos_nonflipped.tight_layout()
    figs["forward_plus_nonflipped_disks"] = fig_fwd_pos_nonflipped

    # --- Forward+ Flipped disks ---
    fig_fwd_pos_flipped, axes_fwd_pos_flipped = plt.subplots(4, 5, figsize=(14, 12))
    fig_fwd_pos_flipped.suptitle("Forward+ (z>0) Flipped Disk Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_forward_flipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range) in enumerate(variables):
            ax = axes_fwd_pos_flipped[var_idx, disk - 1]

            if n_stubs > 0:
                values = disk_data[var_name]
                if var_name == "bend":
                    values = values * 1000
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="salmon")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_pos_flipped.tight_layout()
    figs["forward_plus_flipped_disks"] = fig_fwd_pos_flipped

    # --- Forward- Non-Flipped disks ---
    fig_fwd_neg_nonflipped, axes_fwd_neg_nonflipped = plt.subplots(4, 5, figsize=(14, 12))
    fig_fwd_neg_nonflipped.suptitle("Forward- (z<0) Non-Flipped Disk Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_backward_nonflipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range) in enumerate(variables):
            ax = axes_fwd_neg_nonflipped[var_idx, disk - 1]

            if n_stubs > 0:
                values = disk_data[var_name]
                if var_name == "bend":
                    values = values * 1000
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="green")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_neg_nonflipped.tight_layout()
    figs["forward_minus_nonflipped_disks"] = fig_fwd_neg_nonflipped

    # --- Forward- Flipped disks ---
    fig_fwd_neg_flipped, axes_fwd_neg_flipped = plt.subplots(4, 5, figsize=(14, 12))
    fig_fwd_neg_flipped.suptitle("Forward- (z<0) Flipped Disk Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_backward_flipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range) in enumerate(variables):
            ax = axes_fwd_neg_flipped[var_idx, disk - 1]

            if n_stubs > 0:
                values = disk_data[var_name]
                if var_name == "bend":
                    values = values * 1000
                elif var_name == "ptEst":
                    values = values[(values > 0) & (values < 1e5)]

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="lightgreen")

                if len(values) > 0:
                    stats_text = f"N={len(values)}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_neg_flipped.tight_layout()
    figs["forward_minus_flipped_disks"] = fig_fwd_neg_flipped

    return figs


def make_width_plots(data, figsize=(16, 12)):
    """
    Create per-layer distribution plots for width-related quantities.

    Shows distributions of:
    - width_raw: Raw local-x difference (outer_xLocal - inner_xLocal) [microns]
    - width: Parallax-corrected width (computed from bend * r) [microns]
    - parallaxCorr: The parallax correction value applied [microns]

    Separated by 8 categories:
    - Flat non-flipped barrel layers 1-6
    - Flat flipped barrel layers 1-6
    - Tilted non-flipped barrel layers 1-6
    - Tilted flipped barrel layers 1-6
    - Forward+ non-flipped disks 1-5
    - Forward+ flipped disks 1-5
    - Forward- non-flipped disks 1-5
    - Forward- flipped disks 1-5

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        Dictionary of matplotlib figures keyed by region name
    """
    figs = {}

    # Define variables and their plot ranges
    # (variable_name, xlabel, barrel_range, endcap_range, scale_factor)
    # width_raw and width are converted from cm to microns (multiply by 10000)
    # parallaxCorr is converted from cm to microns (multiply by 10000)
    variables = [
        ("width_raw", "width_raw [um]", (-500.0, 500.0), (-500.0, 500.0), 10000.0),
        ("width", "width [um]", (-500.0, 500.0), (-500.0, 500.0), 10000.0),
        ("parallaxCorr", "parallax corr [um]", (-500, 500), (-500, 500), 10000.0),
    ]

    # --- Flat Non-Flipped Barrel layers ---
    fig_flat_nonflipped_barrel, axes_flat_nonflipped_barrel = plt.subplots(3, 6, figsize=figsize)
    fig_flat_nonflipped_barrel.suptitle("Flat Non-Flipped Barrel Layer Width Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_flat_nonflipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _, scale_factor) in enumerate(variables):
            ax = axes_flat_nonflipped_barrel[var_idx, layer - 1]

            if n_stubs > 0 and var_name in layer_data:
                values = layer_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="blue")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_flat_nonflipped_barrel.tight_layout()
    figs["flat_nonflipped_barrel_layers_width"] = fig_flat_nonflipped_barrel

    # --- Flat Flipped Barrel layers ---
    fig_flat_flipped_barrel, axes_flat_flipped_barrel = plt.subplots(3, 6, figsize=figsize)
    fig_flat_flipped_barrel.suptitle("Flat Flipped Barrel Layer Width Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_flat_flipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _, scale_factor) in enumerate(variables):
            ax = axes_flat_flipped_barrel[var_idx, layer - 1]

            if n_stubs > 0 and var_name in layer_data:
                values = layer_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="cornflowerblue")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_flat_flipped_barrel.tight_layout()
    figs["flat_flipped_barrel_layers_width"] = fig_flat_flipped_barrel

    # --- Tilted Non-Flipped Barrel layers ---
    fig_tilted_nonflipped_barrel, axes_tilted_nonflipped_barrel = plt.subplots(3, 6, figsize=figsize)
    fig_tilted_nonflipped_barrel.suptitle("Tilted Non-Flipped Barrel Layer Width Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_tilted_nonflipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _, scale_factor) in enumerate(variables):
            ax = axes_tilted_nonflipped_barrel[var_idx, layer - 1]

            if n_stubs > 0 and var_name in layer_data:
                values = layer_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="orange")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_tilted_nonflipped_barrel.tight_layout()
    figs["tilted_nonflipped_barrel_layers_width"] = fig_tilted_nonflipped_barrel

    # --- Tilted Flipped Barrel layers ---
    fig_tilted_flipped_barrel, axes_tilted_flipped_barrel = plt.subplots(3, 6, figsize=figsize)
    fig_tilted_flipped_barrel.suptitle("Tilted Flipped Barrel Layer Width Distributions (L1-L6)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        layer_data = select_tilted_flipped_barrel_layer(data, layer)
        n_stubs = len(layer_data["stubIndex"]) if "stubIndex" in layer_data else 0

        for var_idx, (var_name, xlabel, barrel_range, _, scale_factor) in enumerate(variables):
            ax = axes_tilted_flipped_barrel[var_idx, layer - 1]

            if n_stubs > 0 and var_name in layer_data:
                values = layer_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="goldenrod")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_tilted_flipped_barrel.tight_layout()
    figs["tilted_flipped_barrel_layers_width"] = fig_tilted_flipped_barrel

    # --- Forward+ Non-Flipped disks ---
    fig_fwd_pos_nonflipped, axes_fwd_pos_nonflipped = plt.subplots(3, 5, figsize=(14, 10))
    fig_fwd_pos_nonflipped.suptitle("Forward+ (z>0) Non-Flipped Disk Width Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_forward_nonflipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range, scale_factor) in enumerate(variables):
            ax = axes_fwd_pos_nonflipped[var_idx, disk - 1]

            if n_stubs > 0 and var_name in disk_data:
                values = disk_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="red")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_pos_nonflipped.tight_layout()
    figs["forward_plus_nonflipped_disks_width"] = fig_fwd_pos_nonflipped

    # --- Forward+ Flipped disks ---
    fig_fwd_pos_flipped, axes_fwd_pos_flipped = plt.subplots(3, 5, figsize=(14, 10))
    fig_fwd_pos_flipped.suptitle("Forward+ (z>0) Flipped Disk Width Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_forward_flipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range, scale_factor) in enumerate(variables):
            ax = axes_fwd_pos_flipped[var_idx, disk - 1]

            if n_stubs > 0 and var_name in disk_data:
                values = disk_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="salmon")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_pos_flipped.tight_layout()
    figs["forward_plus_flipped_disks_width"] = fig_fwd_pos_flipped

    # --- Forward- Non-Flipped disks ---
    fig_fwd_neg_nonflipped, axes_fwd_neg_nonflipped = plt.subplots(3, 5, figsize=(14, 10))
    fig_fwd_neg_nonflipped.suptitle("Forward- (z<0) Non-Flipped Disk Width Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_backward_nonflipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range, scale_factor) in enumerate(variables):
            ax = axes_fwd_neg_nonflipped[var_idx, disk - 1]

            if n_stubs > 0 and var_name in disk_data:
                values = disk_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="green")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_neg_nonflipped.tight_layout()
    figs["forward_minus_nonflipped_disks_width"] = fig_fwd_neg_nonflipped

    # --- Forward- Flipped disks ---
    fig_fwd_neg_flipped, axes_fwd_neg_flipped = plt.subplots(3, 5, figsize=(14, 10))
    fig_fwd_neg_flipped.suptitle("Forward- (z<0) Flipped Disk Width Distributions (D1-D5)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        disk_data = select_backward_flipped_disk(data, disk)
        n_stubs = len(disk_data["stubIndex"]) if "stubIndex" in disk_data else 0

        for var_idx, (var_name, xlabel, _, endcap_range, scale_factor) in enumerate(variables):
            ax = axes_fwd_neg_flipped[var_idx, disk - 1]

            if n_stubs > 0 and var_name in disk_data:
                values = disk_data[var_name] * scale_factor
                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="lightgreen")

                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                if scale_factor > 1:
                    stats_text = f"mean={mean_val:.1f}\nstd={std_val:.1f}"
                else:
                    stats_text = f"mean={mean_val:.4f}\nstd={std_val:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=7, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk} (N={n_stubs})", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig_fwd_neg_flipped.tight_layout()
    figs["forward_minus_flipped_disks_width"] = fig_fwd_neg_flipped

    return figs


def plot_compatibility_cuts(data, figsize=(20, 16)):
    """
    Plot distributions of all the variables used in CA doublet/triplet compatibility checks.

    The CA algorithm uses these key variables for stub compatibility:
    - dPhiDr: stub direction measurement (rad/cm), compared with doublet direction
    - dPhiDrError: uncertainty on dPhiDr measurement (rad/cm)
    - dr_sensor: radial separation between sensors (cm)
    - dz_sensor: z separation between sensors (cm)
    - bend: angular deviation from straight track (rad)
    - ptEst: pT estimate from stub (GeV)
    - eta_diff: eta consistency between inner/outer hits

    These are the main cut quantities used in:
    - CAPixelDoubletsAlgos.h: dPhiDr consistency check (5-sigma)
    - CAHitNtupletGeneratorKernelsImpl.h: theta alignment, DCA cuts in triplets

    Barrel stubs are split into flat barrel and tilted barrel to investigate
    potential differences in behavior.

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.suptitle("CA Doublet/Triplet Compatibility Cut Variables\n(Barrel split into Flat vs Tilted)",
                 fontsize=14, y=1.02)

    flat_barrel = select_flat(data)
    tilted_barrel = select_tilted(data)
    endcap = select_endcap(data)

    # --- Row 0: dPhiDr and dPhiDrError ---
    # dPhiDr - flat barrel
    ax = axes[0, 0]
    values = flat_barrel["dPhiDr"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr - tilted barrel
    ax = axes[0, 1]
    values = tilted_barrel["dPhiDr"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Tilted Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr - endcap
    ax = axes[0, 2]
    values = endcap["dPhiDr"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.5, 0.5),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Endcap (note 50x larger scale!)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr - overlay all three
    ax = axes[0, 3]
    ax.hist(flat_barrel["dPhiDr"], bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={len(flat_barrel['dPhiDr'])})")
    ax.hist(tilted_barrel["dPhiDr"], bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={len(tilted_barrel['dPhiDr'])})")
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr - Flat vs Tilted Overlay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: dPhiDrError ---
    # dPhiDrError - flat barrel
    ax = axes[1, 0]
    if "dPhiDrError" in flat_barrel:
        values = flat_barrel["dPhiDrError"]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(0, 0.01),
                histtype="step", linewidth=2, color="blue",
                label=f"Flat Barrel (N={n_stubs})")
        if n_stubs > 0:
            mean_err = np.mean(values)
            std_err = np.std(values)
            ax.axvline(mean_err, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={n_stubs}\nmean={mean_err:.5f}\nstd={std_err:.5f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDrError - tilted barrel
    ax = axes[1, 1]
    if "dPhiDrError" in tilted_barrel:
        values = tilted_barrel["dPhiDrError"]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(0, 0.01),
                histtype="step", linewidth=2, color="orange",
                label=f"Tilted Barrel (N={n_stubs})")
        if n_stubs > 0:
            mean_err = np.mean(values)
            std_err = np.std(values)
            ax.axvline(mean_err, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={n_stubs}\nmean={mean_err:.5f}\nstd={std_err:.5f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Tilted Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDrError - endcap
    ax = axes[1, 2]
    if "dPhiDrError" in endcap:
        values = endcap["dPhiDrError"]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(0, 0.1),
                histtype="step", linewidth=2, color="red",
                label=f"Endcap (N={n_stubs})")
        if n_stubs > 0:
            mean_err = np.mean(values)
            std_err = np.std(values)
            ax.axvline(mean_err, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={n_stubs}\nmean={mean_err:.4f}\nstd={std_err:.4f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDrError - Endcap (note 10x larger scale!)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr significance - flat vs tilted overlay
    ax = axes[1, 3]
    if "dPhiDrError" in flat_barrel and "dPhiDrError" in tilted_barrel:
        # Flat barrel
        valid_mask_flat = flat_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask_flat) > 0:
            nsigma_flat = np.abs(flat_barrel["dPhiDr"][valid_mask_flat]) / flat_barrel["dPhiDrError"][valid_mask_flat]
            ax.hist(nsigma_flat, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="blue", alpha=0.8,
                    label=f"Flat (N={len(nsigma_flat)})")
        # Tilted barrel
        valid_mask_tilted = tilted_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask_tilted) > 0:
            nsigma_tilted = np.abs(tilted_barrel["dPhiDr"][valid_mask_tilted]) / tilted_barrel["dPhiDrError"][valid_mask_tilted]
            ax.hist(nsigma_tilted, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="orange", alpha=0.8,
                    label=f"Tilted (N={len(nsigma_tilted)})")
        ax.axvline(5.0, color="black", linestyle="--", alpha=0.7, label="5-sigma cut")
    ax.set_xlabel("|dPhiDr| / dPhiDrError")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr Significance - Flat vs Tilted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 2: Sensor separations ---
    # dr_sensor - flat barrel
    ax = axes[2, 0]
    values = flat_barrel["dr_sensor"]
    n_stubs = len(values)
    ax.hist(values, bins=50, range=(-0.5, 4.5),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor dr - Flat Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dr_sensor - tilted barrel
    ax = axes[2, 1]
    values = tilted_barrel["dr_sensor"]
    n_stubs = len(values)
    ax.hist(values, bins=50, range=(-0.5, 4.5),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor dr - Tilted Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dz_sensor - flat vs tilted overlay
    ax = axes[2, 2]
    ax.hist(flat_barrel["dz_sensor"], bins=50, range=(-0.5, 4.5),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={len(flat_barrel['dz_sensor'])})")
    ax.hist(tilted_barrel["dz_sensor"], bins=50, range=(-0.5, 4.5),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={len(tilted_barrel['dz_sensor'])})")
    ax.set_xlabel("dz_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor dz - Flat vs Tilted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dr_sensor and dz_sensor - endcap
    ax = axes[2, 3]
    ax.hist(endcap["dr_sensor"], bins=50, range=(-0.5, 0.5),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label=f"dr (N={len(endcap['dr_sensor'])})")
    ax.hist(endcap["dz_sensor"], bins=50, range=(-0.5, 0.5),
            histtype="step", linewidth=2, color="darkred", alpha=0.8,
            label=f"dz (N={len(endcap['dz_sensor'])})")
    ax.set_xlabel("Sensor separation [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor dr/dz - Endcap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 3: bend, ptEst, eta_diff ---
    # bend - flat, tilted, endcap overlay
    ax = axes[3, 0]
    values_flat = flat_barrel["bend"] * 1000
    values_tilted = tilted_barrel["bend"] * 1000
    values_endcap = endcap["bend"] * 1000
    ax.hist(values_flat, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={len(values_flat)})")
    ax.hist(values_tilted, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={len(values_tilted)})")
    ax.hist(values_endcap, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label=f"Endcap (N={len(values_endcap)})")
    stats_text = f"Flat: std={np.std(values_flat):.3f}\nTilted: std={np.std(values_tilted):.3f}\nEndcap: std={np.std(values_endcap):.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("bend [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Bend (angular deviation)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ptEst - flat, tilted, endcap overlay
    ax = axes[3, 1]
    pt_flat = flat_barrel["ptEst"]
    pt_flat = pt_flat[(pt_flat > 0) & (pt_flat < 1e5)]
    pt_tilted = tilted_barrel["ptEst"]
    pt_tilted = pt_tilted[(pt_tilted > 0) & (pt_tilted < 1e5)]
    pt_endcap = endcap["ptEst"]
    pt_endcap = pt_endcap[(pt_endcap > 0) & (pt_endcap < 1e5)]
    ax.hist(pt_flat, bins=100, range=(0, 50),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={len(pt_flat)})")
    ax.hist(pt_tilted, bins=100, range=(0, 50),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={len(pt_tilted)})")
    ax.hist(pt_endcap, bins=100, range=(0, 50),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label=f"Endcap (N={len(pt_endcap)})")
    ax.set_xlabel("pT estimate [GeV]")
    ax.set_ylabel("Stubs")
    ax.set_title("pT Estimate")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # eta_diff - flat, tilted, endcap overlay
    ax = axes[3, 2]
    values_flat = flat_barrel["eta_diff"]
    values_tilted = tilted_barrel["eta_diff"]
    values_endcap = endcap["eta_diff"]
    ax.hist(values_flat, bins=100, range=(-0.05, 0.05),
            histtype="step", linewidth=2, color="blue", alpha=0.8,
            label=f"Flat (N={len(values_flat)})")
    ax.hist(values_tilted, bins=100, range=(-0.05, 0.05),
            histtype="step", linewidth=2, color="orange", alpha=0.8,
            label=f"Tilted (N={len(values_tilted)})")
    ax.hist(values_endcap, bins=100, range=(-0.05, 0.05),
            histtype="step", linewidth=2, color="red", alpha=0.8,
            label=f"Endcap (N={len(values_endcap)})")
    stats_text = f"Flat: std={np.std(values_flat):.6f}\nTilted: std={np.std(values_tilted):.6f}\nEndcap: std={np.std(values_endcap):.6f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("eta_outer - eta_inner")
    ax.set_ylabel("Stubs")
    ax.set_title("Eta consistency (z/r alignment)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # dPhiDr/dPhiDrError ratio (nsigma-like) - all three
    ax = axes[3, 3]
    if "dPhiDrError" in flat_barrel:
        valid_mask = flat_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask) > 0:
            nsigma = np.abs(flat_barrel["dPhiDr"][valid_mask]) / flat_barrel["dPhiDrError"][valid_mask]
            ax.hist(nsigma, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="blue", alpha=0.8,
                    label=f"Flat (N={len(nsigma)})")
    if "dPhiDrError" in tilted_barrel:
        valid_mask = tilted_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask) > 0:
            nsigma = np.abs(tilted_barrel["dPhiDr"][valid_mask]) / tilted_barrel["dPhiDrError"][valid_mask]
            ax.hist(nsigma, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="orange", alpha=0.8,
                    label=f"Tilted (N={len(nsigma)})")
    ax.axvline(5.0, color="black", linestyle="--", alpha=0.7, label="5-sigma cut")
    ax.set_xlabel("|dPhiDr| / dPhiDrError")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr significance (nsigma)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_layer_vs_width(data, figsize=(16, 6)):
    """
    Plot 2D scatter plots of layer vs width for barrel and endcap regions.

    Width is the parallax-corrected local-x difference between inner and outer sensors,
    multiplied by 10000 to convert from cm to microns.

    Separate plots for:
    - Barrel: layers 1-6
    - Endcap+ (forward, z > 0): disks 1-5
    - Endcap- (backward, z < 0): disks 1-5

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Layer vs Width (parallax-corrected) [um]", fontsize=14, y=1.02)

    barrel = select_barrel(data)
    forward = select_forward(data)
    backward = select_backward(data)

    # Barrel: layer vs width
    ax = axes[0]
    if "width" in barrel and len(barrel["width"]) > 0:
        width_um = barrel["width"] * 10000  # Convert cm to microns
        h = ax.hist2d(barrel["layer"], width_um, bins=[6, 100],
                      range=[[0.5, 6.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Barrel (N={len(barrel.get('width', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap+ (forward): disk vs width
    ax = axes[1]
    if "width" in forward and len(forward["width"]) > 0:
        width_um = forward["width"] * 10000  # Convert cm to microns
        h = ax.hist2d(forward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Endcap+ z>0 (N={len(forward.get('width', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap- (backward): disk vs width
    ax = axes[2]
    if "width" in backward and len(backward["width"]) > 0:
        width_um = backward["width"] * 10000  # Convert cm to microns
        h = ax.hist2d(backward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Endcap- z<0 (N={len(backward.get('width', []))})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_layer_vs_width_raw(data, figsize=(16, 6)):
    """
    Plot 2D scatter plots of layer vs width_raw for barrel and endcap regions.

    Width_raw is the raw local-x difference (outer_xLocal - inner_xLocal) without
    parallax correction, multiplied by 10000 to convert from cm to microns.

    Separate plots for:
    - Barrel: layers 1-6
    - Endcap+ (forward, z > 0): disks 1-5
    - Endcap- (backward, z < 0): disks 1-5

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Layer vs Width Raw (no parallax correction) [um]", fontsize=14, y=1.02)

    barrel = select_barrel(data)
    forward = select_forward(data)
    backward = select_backward(data)

    # Barrel: layer vs width_raw
    ax = axes[0]
    if "width_raw" in barrel and len(barrel["width_raw"]) > 0:
        width_um = barrel["width_raw"] * 10000  # Convert cm to microns
        h = ax.hist2d(barrel["layer"], width_um, bins=[6, 100],
                      range=[[0.5, 6.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Barrel (N={len(barrel.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap+ (forward): disk vs width_raw
    ax = axes[1]
    if "width_raw" in forward and len(forward["width_raw"]) > 0:
        width_um = forward["width_raw"] * 10000  # Convert cm to microns
        h = ax.hist2d(forward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Endcap+ z>0 (N={len(forward.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap- (backward): disk vs width_raw
    ax = axes[2]
    if "width_raw" in backward and len(backward["width_raw"]) > 0:
        width_um = backward["width_raw"] * 10000  # Convert cm to microns
        h = ax.hist2d(backward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Endcap- z<0 (N={len(backward.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_layer_vs_width_comparison(data, figsize=(16, 10)):
    """
    Combined plot showing layer vs width and layer vs width_raw side by side.

    Shows both parallax-corrected width and raw width for comparison,
    separated by barrel and endcap regions.

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Layer vs Width Comparison: Raw (top) vs Parallax-Corrected (bottom) [um]",
                 fontsize=14, y=1.02)

    barrel = select_barrel(data)
    forward = select_forward(data)
    backward = select_backward(data)

    # Row 0: width_raw (no parallax correction)
    # Barrel
    ax = axes[0, 0]
    if "width_raw" in barrel and len(barrel["width_raw"]) > 0:
        width_um = barrel["width_raw"] * 10000
        h = ax.hist2d(barrel["layer"], width_um, bins=[6, 100],
                      range=[[0.5, 6.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Barrel - Raw (N={len(barrel.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap+
    ax = axes[0, 1]
    if "width_raw" in forward and len(forward["width_raw"]) > 0:
        width_um = forward["width_raw"] * 10000
        h = ax.hist2d(forward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Endcap+ - Raw (N={len(forward.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap-
    ax = axes[0, 2]
    if "width_raw" in backward and len(backward["width_raw"]) > 0:
        width_um = backward["width_raw"] * 10000
        h = ax.hist2d(backward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width_raw [um]")
    ax.set_title(f"Endcap- - Raw (N={len(backward.get('width_raw', []))})")
    ax.grid(True, alpha=0.3)

    # Row 1: width (parallax-corrected)
    # Barrel
    ax = axes[1, 0]
    if "width" in barrel and len(barrel["width"]) > 0:
        width_um = barrel["width"] * 10000
        h = ax.hist2d(barrel["layer"], width_um, bins=[6, 100],
                      range=[[0.5, 6.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Barrel - Corrected (N={len(barrel.get('width', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap+
    ax = axes[1, 1]
    if "width" in forward and len(forward["width"]) > 0:
        width_um = forward["width"] * 10000
        h = ax.hist2d(forward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Endcap+ - Corrected (N={len(forward.get('width', []))})")
    ax.grid(True, alpha=0.3)

    # Endcap-
    ax = axes[1, 2]
    if "width" in backward and len(backward["width"]) > 0:
        width_um = backward["width"] * 10000
        h = ax.hist2d(backward["layer"], width_um, bins=[5, 100],
                      range=[[0.5, 5.5], [-500, 500]], cmin=1)
        plt.colorbar(h[3], ax=ax, label="Stubs")
    ax.set_xlabel("Disk")
    ax.set_ylabel("width [um]")
    ax.set_title(f"Endcap- - Corrected (N={len(backward.get('width', []))})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_xy_angle(data, figsize=(14, 10)):
    """
    Plot the azimuthal angle of the stub direction vector (inner->outer) distribution.

    The stub direction vector is computed from the inner hit to the outer hit:
        direction = (outer_x - inner_x, outer_y - inner_y)
        direction_phi = atan2(outer_y - inner_y, outer_x - inner_x)

    This shows the direction each stub is pointing in the XY plane, comparing
    barrel vs endcap regions. Also shows 2D distributions of direction phi vs layer/disk.

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    barrel = select_barrel(data)
    forward = select_forward(data)
    backward = select_backward(data)

    # Compute direction phi from inner->outer hit displacement
    # direction_phi = atan2(outer_y - inner_y, outer_x - inner_x)
    dir_phi_all = np.arctan2(data["outer_y"] - data["inner_y"],
                             data["outer_x"] - data["inner_x"])
    dir_phi_barrel = np.arctan2(barrel["outer_y"] - barrel["inner_y"],
                                barrel["outer_x"] - barrel["inner_x"])
    dir_phi_forward = np.arctan2(forward["outer_y"] - forward["inner_y"],
                                 forward["outer_x"] - forward["inner_x"])
    dir_phi_backward = np.arctan2(backward["outer_y"] - backward["inner_y"],
                                  backward["outer_x"] - backward["inner_x"])

    # 1D histograms of direction phi
    ax = axes[0, 0]
    ax.hist(dir_phi_barrel, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, label=f"Barrel (N={len(dir_phi_barrel)})")
    ax.set_xlabel(r"direction $\phi$ = atan2(dy, dx) [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Stub Direction Angle - Barrel")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(dir_phi_forward, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, color="red",
            label=f"Forward+ (N={len(dir_phi_forward)})")
    ax.set_xlabel(r"direction $\phi$ = atan2(dy, dx) [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Stub Direction Angle - Forward+ (z>0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.hist(dir_phi_backward, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, color="green",
            label=f"Forward- (N={len(dir_phi_backward)})")
    ax.set_xlabel(r"direction $\phi$ = atan2(dy, dx) [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Stub Direction Angle - Forward- (z<0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison plot
    ax = axes[1, 0]
    ax.hist(dir_phi_barrel, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, label="Barrel", alpha=0.8)
    ax.hist(dir_phi_forward, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, color="red", label="Forward+", alpha=0.8)
    ax.hist(dir_phi_backward, bins=100, range=(-np.pi, np.pi),
            histtype="step", linewidth=2, color="green", label="Forward-", alpha=0.8)
    ax.set_xlabel(r"direction $\phi$ = atan2(dy, dx) [rad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Stub Direction Angle - All Regions Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D: direction phi vs layer for barrel
    ax = axes[1, 1]
    h = ax.hist2d(dir_phi_barrel, barrel["layer"], bins=[50, 6],
                  range=[[-np.pi, np.pi], [0.5, 6.5]], cmin=1)
    ax.set_xlabel(r"direction $\phi$ [rad]")
    ax.set_ylabel("Layer")
    ax.set_title("Barrel: direction phi vs Layer")
    plt.colorbar(h[3], ax=ax, label="Stubs")

    # 2D: direction phi vs disk for endcap (combined forward+ and forward-)
    endcap = select_endcap(data)
    dir_phi_endcap = np.arctan2(endcap["outer_y"] - endcap["inner_y"],
                                endcap["outer_x"] - endcap["inner_x"])
    ax = axes[1, 2]
    h = ax.hist2d(dir_phi_endcap, endcap["layer"], bins=[50, 5],
                  range=[[-np.pi, np.pi], [0.5, 5.5]], cmin=1)
    ax.set_xlabel(r"direction $\phi$ [rad]")
    ax.set_ylabel("Disk")
    ax.set_title("Endcap: direction phi vs Disk")
    plt.colorbar(h[3], ax=ax, label="Stubs")

    plt.tight_layout()
    return fig


def plot_consecutive_layer_dotproduct(data, figsize=(16, 14)):
    """
    Compute and plot the absolute dot product of stub direction vectors from consecutive layers,
    along with dZ (for barrel-to-barrel) and dR (for connections ending in forward disks).

    For stubs in the SAME EVENT, computes:
    1. Absolute value of the dot product between NORMALIZED direction vectors in XY plane
    2. dZ = stub2_z - stub1_z for barrel-to-barrel connections
    3. dR = stub2_r - stub1_r for connections ending in a forward disk

    Layer pairs analyzed:
    - Barrel consecutive: L1->L2, L2->L3, L3->L4, L4->L5, L5->L6
    - Barrel to Endcap+: L1->D1, L2->D1, L3->D1, L4->D1, L5->D1, L6->D1
    - Barrel to Endcap-: L1->D1, L2->D1, L3->D1, L4->D1, L5->D1, L6->D1
    - Endcap+ consecutive: D1->D2, D2->D3, D3->D4, D4->D5
    - Endcap- consecutive: D1->D2, D2->D3, D3->D4, D4->D5

    The direction vector for each stub is computed from the inner hit to the outer hit:
        direction = (outer_x - inner_x, outer_y - inner_y)
        normalized_direction = direction / |direction|

    The absolute dot product is computed as:
        abs_dot = |dir_x1 * dir_x2 + dir_y1 * dir_y2|

    This gives values in range [0, 1]:
    - Values near 1 indicate aligned stubs (parallel directions)
    - Values near 0 indicate perpendicular stubs

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        Dictionary containing figures and computed values for each layer pair
    """
    # Compute direction vectors (inner->outer) for all stubs
    dx = data["outer_x"] - data["inner_x"]
    dy = data["outer_y"] - data["inner_y"]
    # Compute magnitude for normalization
    mag = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero (should not happen for valid stubs)
    mag = np.where(mag > 0, mag, 1.0)
    # Normalized direction components
    dir_x = dx / mag
    dir_y = dy / mag

    # Get event numbers and create masks for each region/layer combination
    events = data["event"]
    is_barrel = data["isBarrel"]
    is_endcap = data["isEndcap"]
    stub_z = data["stub_z"]
    stub_r = data["stub_r"]
    layer = data["layer"]

    # Helper function to compute absolute dot products and dZ/dR between stubs in consecutive layers
    def compute_layer_pair_quantities(mask1, mask2, compute_dz=False, compute_dr=False):
        """
        Compute absolute dot products, dZ, and dR for all stub pairs from two layer
        selections within the same event.

        Args:
            mask1: Boolean mask for first layer selection
            mask2: Boolean mask for second layer selection
            compute_dz: Whether to compute dZ (for barrel-to-barrel)
            compute_dr: Whether to compute dR (for connections ending in endcap)

        Returns:
            Dictionary with 'abs_dots', 'dz', and 'dr' arrays
        """
        # Get indices for each layer
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return {'abs_dots': np.array([]), 'dz': np.array([]), 'dr': np.array([])}

        # Get event numbers and normalized direction components for each layer
        events1 = events[idx1]
        events2 = events[idx2]
        dir_x1 = dir_x[idx1]
        dir_y1 = dir_y[idx1]
        dir_x2 = dir_x[idx2]
        dir_y2 = dir_y[idx2]
        z1 = stub_z[idx1]
        z2 = stub_z[idx2]
        r1 = stub_r[idx1]
        r2 = stub_r[idx2]

        # Find unique events that have stubs in both layers
        unique_events = np.intersect1d(np.unique(events1), np.unique(events2))

        abs_dots = []
        dz_vals = []
        dr_vals = []

        for evt in unique_events:
            # Get values for this event in each layer
            evt_mask1 = events1 == evt
            evt_mask2 = events2 == evt
            dx1_evt = dir_x1[evt_mask1]
            dy1_evt = dir_y1[evt_mask1]
            dx2_evt = dir_x2[evt_mask2]
            dy2_evt = dir_y2[evt_mask2]
            z1_evt = z1[evt_mask1]
            z2_evt = z2[evt_mask2]
            r1_evt = r1[evt_mask1]
            r2_evt = r2[evt_mask2]

            # Compute quantities for all pairs
            for i in range(len(dx1_evt)):
                for j in range(len(dx2_evt)):
                    dot = dx1_evt[i] * dx2_evt[j] + dy1_evt[i] * dy2_evt[j]
                    abs_dots.append(np.abs(dot))
                    if compute_dz:
                        dz_vals.append(z2_evt[j] - z1_evt[i])
                    if compute_dr:
                        dr_vals.append(r2_evt[j] - r1_evt[i])

        return {
            'abs_dots': np.array(abs_dots),
            'dz': np.array(dz_vals) if compute_dz else np.array([]),
            'dr': np.array(dr_vals) if compute_dr else np.array([])
        }

    # Define layer pairs to analyze with their type (for dZ/dR computation)
    # Type: 'barrel' = barrel-to-barrel (compute dZ), 'to_endcap' = ending in endcap (compute dR)
    layer_pairs = {}

    # Barrel consecutive layers: L1->L2, L2->L3, ..., L5->L6
    for l in range(1, 6):
        pair_name = f"Barrel L{l}->L{l+1}"
        mask1 = is_barrel & (layer == l)
        mask2 = is_barrel & (layer == l + 1)
        layer_pairs[pair_name] = (mask1, mask2, 'barrel')

    # Barrel to Forward+ transition: All barrel layers to D1 (for z > 0)
    mask_fwd_pos_d1 = is_endcap & (stub_z > 0) & (layer == 1)
    for l in range(1, 7):
        pair_name = f"Barrel L{l}->Fwd+ D1"
        mask_barrel = is_barrel & (layer == l)
        layer_pairs[pair_name] = (mask_barrel, mask_fwd_pos_d1, 'to_endcap')

    # Barrel to Forward- transition: All barrel layers to D1 (for z < 0)
    mask_fwd_neg_d1 = is_endcap & (stub_z < 0) & (layer == 1)
    for l in range(1, 7):
        pair_name = f"Barrel L{l}->Fwd- D1"
        mask_barrel = is_barrel & (layer == l)
        layer_pairs[pair_name] = (mask_barrel, mask_fwd_neg_d1, 'to_endcap')

    # Forward+ consecutive disks: D1->D2, D2->D3, D3->D4, D4->D5
    for d in range(1, 5):
        pair_name = f"Fwd+ D{d}->D{d+1}"
        mask1 = is_endcap & (stub_z > 0) & (layer == d)
        mask2 = is_endcap & (stub_z > 0) & (layer == d + 1)
        layer_pairs[pair_name] = (mask1, mask2, 'to_endcap')

    # Forward- consecutive disks: D1->D2, D2->D3, D3->D4, D4->D5
    for d in range(1, 5):
        pair_name = f"Fwd- D{d}->D{d+1}"
        mask1 = is_endcap & (stub_z < 0) & (layer == d)
        mask2 = is_endcap & (stub_z < 0) & (layer == d + 1)
        layer_pairs[pair_name] = (mask1, mask2, 'to_endcap')

    # Compute quantities for all layer pairs
    print("Computing |dot product|, dZ, and dR between stubs from consecutive layers...")
    results = {}
    for pair_name, (mask1, mask2, pair_type) in layer_pairs.items():
        print(f"  Processing {pair_name}...")
        compute_dz = (pair_type == 'barrel')
        compute_dr = (pair_type == 'to_endcap')
        quantities = compute_layer_pair_quantities(mask1, mask2, compute_dz=compute_dz, compute_dr=compute_dr)
        results[pair_name] = quantities
        print(f"    Found {len(quantities['abs_dots'])} pairs")

    # Order for plotting
    plot_order = [
        # Row 0: Barrel consecutive L1-L2 through L5-L6
        "Barrel L1->L2", "Barrel L2->L3", "Barrel L3->L4", "Barrel L4->L5", "Barrel L5->L6",
        # Row 1: Barrel to Forward+ transitions (L1-L6 -> D1)
        "Barrel L1->Fwd+ D1", "Barrel L2->Fwd+ D1", "Barrel L3->Fwd+ D1", "Barrel L4->Fwd+ D1", "Barrel L5->Fwd+ D1",
        # Row 2: Barrel L6 to Fwd+ D1, then Forward+ consecutive D1-D5
        "Barrel L6->Fwd+ D1", "Fwd+ D1->D2", "Fwd+ D2->D3", "Fwd+ D3->D4", "Fwd+ D4->D5",
        # Row 3: Barrel to Forward- transitions (L1-L6 -> D1)
        "Barrel L1->Fwd- D1", "Barrel L2->Fwd- D1", "Barrel L3->Fwd- D1", "Barrel L4->Fwd- D1", "Barrel L5->Fwd- D1",
        # Row 4: Barrel L6 to Fwd- D1, then Forward- consecutive D1-D5
        "Barrel L6->Fwd- D1", "Fwd- D1->D2", "Fwd- D2->D3", "Fwd- D3->D4", "Fwd- D4->D5",
    ]

    colors = {
        "Barrel": "blue",
        "Fwd+": "red",
        "Fwd-": "green",
    }

    # Helper function to get color based on pair name
    def get_color(pair_name):
        if pair_name.startswith("Barrel") and "Fwd" not in pair_name:
            return colors["Barrel"]
        elif "Fwd+" in pair_name:
            return colors["Fwd+"]
        else:
            return colors["Fwd-"]

    # --- Figure 1: |Dot Product| plots ---
    fig_dotprod, axes_dotprod = plt.subplots(5, 5, figsize=figsize)
    fig_dotprod.suptitle("|Dot Product| of Stub Direction Vectors from Consecutive Layers\n"
                         "(Values near 1 = aligned/parallel stubs, values near 0 = perpendicular stubs)",
                         fontsize=12, y=1.02)

    for idx, pair_name in enumerate(plot_order):
        row = idx // 5
        col = idx % 5
        ax = axes_dotprod[row, col]

        if pair_name is None:
            ax.axis("off")
            continue

        quantities = results.get(pair_name, {'abs_dots': np.array([])})
        abs_dots = quantities['abs_dots']
        color = get_color(pair_name)

        if len(abs_dots) > 0:
            ax.hist(abs_dots, bins=100, range=(0, 1),
                    histtype="step", linewidth=1.5, color=color)

            # Add statistics
            mean_val = np.mean(abs_dots)
            std_val = np.std(abs_dots)
            ax.axvline(mean_val, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={len(abs_dots)}\nmean={mean_val:.3f}\nstd={std_val:.3f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("|dot product|", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.set_title(pair_name, fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    fig_dotprod.tight_layout()

    # --- Figure 2: dZ plots for barrel-to-barrel connections ---
    # Only first row has dZ data (barrel consecutive layers)
    barrel_pairs = ["Barrel L1->L2", "Barrel L2->L3", "Barrel L3->L4", "Barrel L4->L5", "Barrel L5->L6"]
    fig_dz, axes_dz = plt.subplots(1, 5, figsize=(16, 4))
    fig_dz.suptitle("dZ (z2 - z1) for Barrel-to-Barrel Consecutive Layer Connections [mm]",
                    fontsize=12, y=1.02)

    for idx, pair_name in enumerate(barrel_pairs):
        ax = axes_dz[idx]
        quantities = results.get(pair_name, {'dz': np.array([])})
        dz_vals = quantities['dz'] * 10  # Convert from cm to mm

        if len(dz_vals) > 0:
            ax.hist(dz_vals, bins=100, range=(-50, 50),
                    histtype="step", linewidth=1.5, color=colors["Barrel"])

            # Add statistics
            mean_val = np.mean(dz_vals)
            std_val = np.std(dz_vals)
            ax.axvline(mean_val, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={len(dz_vals)}\nmean={mean_val:.1f} mm\nstd={std_val:.1f} mm"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("dZ [mm]", fontsize=9)
        ax.set_ylabel("Pairs", fontsize=9)
        ax.set_title(pair_name, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, alpha=0.3)

    fig_dz.tight_layout()

    # --- Figure 3: dR plots for connections ending in forward disks ---
    # This includes barrel->endcap and endcap->endcap transitions
    fig_dr, axes_dr = plt.subplots(4, 5, figsize=(16, 12))
    fig_dr.suptitle("dR (r2 - r1) for Connections Ending in Forward Disks [mm]",
                    fontsize=12, y=1.02)

    # Order for dR plots (exclude pure barrel-to-barrel)
    dr_plot_order = [
        # Row 0: Barrel to Forward+ transitions (L1-L5 -> D1)
        "Barrel L1->Fwd+ D1", "Barrel L2->Fwd+ D1", "Barrel L3->Fwd+ D1", "Barrel L4->Fwd+ D1", "Barrel L5->Fwd+ D1",
        # Row 1: Barrel L6 to Fwd+ D1, then Forward+ consecutive D1-D5
        "Barrel L6->Fwd+ D1", "Fwd+ D1->D2", "Fwd+ D2->D3", "Fwd+ D3->D4", "Fwd+ D4->D5",
        # Row 2: Barrel to Forward- transitions (L1-L5 -> D1)
        "Barrel L1->Fwd- D1", "Barrel L2->Fwd- D1", "Barrel L3->Fwd- D1", "Barrel L4->Fwd- D1", "Barrel L5->Fwd- D1",
        # Row 3: Barrel L6 to Fwd- D1, then Forward- consecutive D1-D5
        "Barrel L6->Fwd- D1", "Fwd- D1->D2", "Fwd- D2->D3", "Fwd- D3->D4", "Fwd- D4->D5",
    ]

    for idx, pair_name in enumerate(dr_plot_order):
        row = idx // 5
        col = idx % 5
        ax = axes_dr[row, col]

        quantities = results.get(pair_name, {'dr': np.array([])})
        dr_vals = quantities['dr'] * 10  # Convert from cm to mm
        color = get_color(pair_name)

        if len(dr_vals) > 0:
            ax.hist(dr_vals, bins=100, range=(-50, 50),
                    histtype="step", linewidth=1.5, color=color)

            # Add statistics
            mean_val = np.mean(dr_vals)
            std_val = np.std(dr_vals)
            ax.axvline(mean_val, color="black", linestyle="--", alpha=0.5)
            stats_text = f"N={len(dr_vals)}\nmean={mean_val:.1f} mm\nstd={std_val:.1f} mm"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("dR [mm]", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.set_title(pair_name, fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.3)

    fig_dr.tight_layout()

    return {
        "figure_dotproduct": fig_dotprod,
        "figure_dz": fig_dz,
        "figure_dr": fig_dr,
        "results": results
    }


def plot_event_stubs(data, event_number, figsize=(16, 8)):
    """
    Visualize stub segments for a specific event in XY and r-Z projections.

    For the specified event, draws all stub segments from inner hit to outer hit
    in two side-by-side plots:
    - Left plot: XY plane projection
    - Right plot: r-Z plane projection

    Visual distinctions:
    - Color by module type:
      - Flat barrel stubs: blue
      - Tilted barrel stubs: green
      - Endcap stubs: red
    - Line width by sensor type:
      - PS modules: thick line (linewidth=4)
      - SS modules: normal line (linewidth=3)
    - Extended direction lines: thin lines (linewidth=0.5, alpha=0.3) showing
      the stub direction extended to the plot edges

    Note: This function is intended for manual use and is NOT called by make_all_plots().

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        event_number: The specific event number to visualize
        figsize: Optional figure size tuple (default: (16, 8))

    Returns:
        matplotlib figure
    """

    def compute_line_to_boundary(px, py, dx, dy, xlim, ylim):
        """
        Compute line endpoints that extend from point (px, py) in direction (dx, dy)
        to the plot boundaries defined by xlim and ylim.

        Returns the two intersection points with the plot boundary.
        """
        if dx == 0 and dy == 0:
            # No direction, just return the point
            return [px, px], [py, py]

        t_values = []

        # Find t values for intersection with each boundary
        # x = px + t * dx, y = py + t * dy

        if dx != 0:
            # Left boundary: x = xlim[0]
            t_left = (xlim[0] - px) / dx
            t_values.append(t_left)
            # Right boundary: x = xlim[1]
            t_right = (xlim[1] - px) / dx
            t_values.append(t_right)

        if dy != 0:
            # Bottom boundary: y = ylim[0]
            t_bottom = (ylim[0] - py) / dy
            t_values.append(t_bottom)
            # Top boundary: y = ylim[1]
            t_top = (ylim[1] - py) / dy
            t_values.append(t_top)

        # Find valid intersection points (within the plot boundaries)
        valid_points = []
        for t in t_values:
            x = px + t * dx
            y = py + t * dy
            # Check if point is within boundaries (with small tolerance)
            eps = 1e-9
            if (xlim[0] - eps <= x <= xlim[1] + eps and
                ylim[0] - eps <= y <= ylim[1] + eps):
                valid_points.append((x, y, t))

        if len(valid_points) < 2:
            # Fallback: return the original point
            return [px, px], [py, py]

        # Sort by t to get the two extreme points
        valid_points.sort(key=lambda p: p[2])
        p1 = valid_points[0]
        p2 = valid_points[-1]

        return [p1[0], p2[0]], [p1[1], p2[1]]

    # Select stubs for the specified event
    event_mask = data["event"] == event_number
    n_stubs = np.sum(event_mask)

    if n_stubs == 0:
        print(f"No stubs found for event {event_number}")
        return None

    # Extract data for this event
    inner_x = data["inner_x"][event_mask]
    inner_y = data["inner_y"][event_mask]
    inner_z = data["inner_z"][event_mask]
    outer_x = data["outer_x"][event_mask]
    outer_y = data["outer_y"][event_mask]
    outer_z = data["outer_z"][event_mask]
    is_barrel = data["isBarrel"][event_mask]
    is_tilted = data["isTilted"][event_mask]
    is_ps = data["isPS"][event_mask]

    # Compute r coordinates for r-Z projection
    r_inner = np.sqrt(inner_x**2 + inner_y**2)
    r_outer = np.sqrt(outer_x**2 + outer_y**2)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Stub Segments - Event {event_number} (N={n_stubs})", fontsize=14)

    ax_xy = axes[0]
    ax_rz = axes[1]

    # Track whether we've added legend entries for each category
    legend_added = {
        "flat_barrel_ps": False,
        "flat_barrel_ss": False,
        "tilted_barrel_ps": False,
        "tilted_barrel_ss": False,
        "endcap_ps": False,
        "endcap_ss": False,
    }

    # Define styling for each category
    # Color: flat barrel=blue, tilted barrel=green, endcap=red
    # Line width: PS=4 (thick), SS=3 (normal) for bold stub segments
    category_styles = {
        "flat_barrel_ps": {"color": "blue", "linewidth": 4, "label": "Flat Barrel (PS)"},
        "flat_barrel_ss": {"color": "blue", "linewidth": 3, "label": "Flat Barrel (SS)"},
        "tilted_barrel_ps": {"color": "green", "linewidth": 4, "label": "Tilted Barrel (PS)"},
        "tilted_barrel_ss": {"color": "green", "linewidth": 3, "label": "Tilted Barrel (SS)"},
        "endcap_ps": {"color": "red", "linewidth": 4, "label": "Endcap (PS)"},
        "endcap_ss": {"color": "red", "linewidth": 3, "label": "Endcap (SS)"},
    }

    # Store stub data for drawing extended lines later
    stub_data_for_extension = []

    # Step 1: Draw all bold stub segments first to determine axis limits
    for i in range(n_stubs):
        # XY coordinates
        x_coords = [inner_x[i], outer_x[i]]
        y_coords = [inner_y[i], outer_y[i]]

        # r-Z coordinates
        z_coords = [inner_z[i], outer_z[i]]
        r_coords = [r_inner[i], r_outer[i]]

        # Determine category
        if is_barrel[i]:
            if is_tilted[i]:
                category = "tilted_barrel_ps" if is_ps[i] else "tilted_barrel_ss"
            else:
                category = "flat_barrel_ps" if is_ps[i] else "flat_barrel_ss"
        else:
            category = "endcap_ps" if is_ps[i] else "endcap_ss"

        style = category_styles[category]
        color = style["color"]
        linewidth = style["linewidth"]
        label = style["label"]

        # Draw bold stub segments
        if not legend_added[category]:
            ax_xy.plot(x_coords, y_coords, color=color, linewidth=linewidth, alpha=0.7, label=label)
            ax_rz.plot(z_coords, r_coords, color=color, linewidth=linewidth, alpha=0.7, label=label)
            legend_added[category] = True
        else:
            ax_xy.plot(x_coords, y_coords, color=color, linewidth=linewidth, alpha=0.7)
            ax_rz.plot(z_coords, r_coords, color=color, linewidth=linewidth, alpha=0.7)

        # Store data for extended lines
        stub_data_for_extension.append({
            "center_x": (inner_x[i] + outer_x[i]) / 2,
            "center_y": (inner_y[i] + outer_y[i]) / 2,
            "dx_xy": outer_x[i] - inner_x[i],
            "dy_xy": outer_y[i] - inner_y[i],
            "center_z": (inner_z[i] + outer_z[i]) / 2,
            "center_r": (r_inner[i] + r_outer[i]) / 2,
            "dz": outer_z[i] - inner_z[i],
            "dr": r_outer[i] - r_inner[i],
            "color": color,
        })

    # Step 2: Get axis limits determined by stub data
    ax_xy.set_aspect("equal")  # Set aspect before getting limits
    ax_xy.autoscale_view()
    ax_rz.autoscale_view()

    xlim_xy = ax_xy.get_xlim()
    ylim_xy = ax_xy.get_ylim()
    xlim_rz = ax_rz.get_xlim()
    ylim_rz = ax_rz.get_ylim()

    # Step 3: Draw thin extended direction lines clipped to plot boundaries
    extend_linewidth = 0.5
    extend_alpha = 0.3

    for stub in stub_data_for_extension:
        # XY extended line
        ext_x, ext_y = compute_line_to_boundary(
            stub["center_x"], stub["center_y"],
            stub["dx_xy"], stub["dy_xy"],
            xlim_xy, ylim_xy
        )
        ax_xy.plot(ext_x, ext_y, color=stub["color"],
                   linewidth=extend_linewidth, alpha=extend_alpha)

        # r-Z extended line
        ext_z, ext_r = compute_line_to_boundary(
            stub["center_z"], stub["center_r"],
            stub["dz"], stub["dr"],
            xlim_rz, ylim_rz
        )
        ax_rz.plot(ext_z, ext_r, color=stub["color"],
                   linewidth=extend_linewidth, alpha=extend_alpha)

    # Step 4: Restore axis limits (in case extended lines affected them)
    ax_xy.set_xlim(xlim_xy)
    ax_xy.set_ylim(ylim_xy)
    ax_rz.set_xlim(xlim_rz)
    ax_rz.set_ylim(ylim_rz)

    # Configure XY plot
    ax_xy.set_xlabel("x [cm]")
    ax_xy.set_ylabel("y [cm]")
    ax_xy.set_title("XY Projection")
    ax_xy.legend(fontsize=9)
    ax_xy.grid(True, alpha=0.3)

    # Configure r-Z plot
    ax_rz.set_xlabel("z [cm]")
    ax_rz.set_ylabel("r [cm]")
    ax_rz.set_title("r-Z Projection")
    ax_rz.legend(fontsize=9)
    ax_rz.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_proposed_compatibility(data, figsize=(20, 20)):
    """
    Plot distributions of the proposed quantities for stub compatibility checks
    in the three geometry cases identified in the documentation.

    The CA algorithm must handle three distinct geometric cases for doublet
    direction checks, each requiring a different approach:

    Case 1: Barrel-Barrel Doublets
        - Use dPhiDr (dphi/dr) - well-defined since dr_sensor is large (~2-4mm)
        - Comparison: dPhiDr_doublet vs dPhiDr_stub within nSigma * error
        - Split into flat barrel vs tilted barrel for investigation

    Case 2: Endcap-Endcap Doublets
        - dPhiDr is poorly defined (dr_sensor ~ 0)
        - Proposed alternatives:
          a) dPhiDz (dphi/dz) - well-behaved since dz_sensor is large (~4mm)
          b) Eta consistency (z/r) - should be constant for straight line in r-z
          c) Bend angle only (not divided by dr)

    Case 3: Barrel-Endcap Transition
        - Mixed geometry, most complex case
        - Proposed: Use bend sign check only (sign consistency)
        - L1T-style geometric correction: rho = (pitch/separation) * (z/r)

    This function visualizes these quantities separated by the three geometry cases.

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(6, 4, figsize=(figsize[0], int(figsize[1] * 6 / 5)))
    fig.suptitle("Proposed Compatibility Quantities for Three Geometry Cases\n"
                 "(Barrel split into Flat vs Tilted)", fontsize=14, y=1.02)

    # Create geometry masks
    barrel_mask = data["isBarrel"] == True
    endcap_mask = data["isEndcap"] == True
    forward_mask = endcap_mask & (data["stub_z"] > 0)
    backward_mask = endcap_mask & (data["stub_z"] < 0)

    barrel = select_barrel(data)
    flat_barrel = select_flat(data)
    tilted_barrel = select_tilted(data)
    endcap = select_endcap(data)
    forward = select_forward(data)
    backward = select_backward(data)

    # ========================================================================
    # Row 0: Case 1 - Flat Barrel quantities (dPhiDr is appropriate)
    # ========================================================================

    # dPhiDr - Flat Barrel
    ax = axes[0, 0]
    values = flat_barrel["dPhiDr"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDr - Flat Barrel\n(well-defined, use for barrel-barrel)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDrError - Flat Barrel (for nSigma cuts)
    ax = axes[0, 1]
    if "dPhiDrError" in flat_barrel:
        values = flat_barrel["dPhiDrError"]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(0, 0.005),
                histtype="step", linewidth=2, color="blue",
                label=f"Flat Barrel (N={n_stubs})")
        if n_stubs > 0:
            mean_err = np.mean(values)
            std_err = np.std(values)
            ax.axvline(mean_err, color="black", linestyle="--", alpha=0.7)
            stats_text = f"N={n_stubs}\nmean={mean_err:.5f}\nstd={std_err:.5f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDrError - Flat Barrel\n(for nSigma compatibility cut)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Flat Barrel dr_sensor - shows why dPhiDr works
    ax = axes[0, 2]
    values = flat_barrel["dr_sensor"]
    n_stubs = len(values)
    ax.hist(values, bins=50, range=(0, 5),
            histtype="step", linewidth=2, color="blue",
            label=f"Flat dr (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: Flat Barrel dr_sensor\n(large dr makes dPhiDr well-defined)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr significance (|dPhiDr|/error) for flat barrel
    ax = axes[0, 3]
    if "dPhiDrError" in flat_barrel:
        valid_mask = flat_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask) > 0:
            nsigma = np.abs(flat_barrel["dPhiDr"][valid_mask]) / flat_barrel["dPhiDrError"][valid_mask]
            n_stubs = len(nsigma)
            ax.hist(nsigma, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="blue",
                    label=f"Flat (N={n_stubs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
            # Count fraction within 5 sigma
            within_5sigma = np.sum(nsigma < 5.0) / len(nsigma) * 100
            stats_text = f"N={n_stubs}\nmean={np.mean(nsigma):.3f}\nstd={np.std(nsigma):.3f}\n{within_5sigma:.1f}% < 5 sigma"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("|dPhiDr| / dPhiDrError")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDr Significance - Flat\n(use 5-sigma cut for compatibility)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 1: Case 1 - Tilted Barrel quantities (dPhiDr behavior may differ)
    # ========================================================================

    # dPhiDr - Tilted Barrel
    ax = axes[1, 0]
    values = tilted_barrel["dPhiDr"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted Barrel (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDr - Tilted Barrel\n(may have different behavior)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDrError - Tilted Barrel (for nSigma cuts)
    ax = axes[1, 1]
    if "dPhiDrError" in tilted_barrel:
        values = tilted_barrel["dPhiDrError"]
        n_stubs = len(values)
        ax.hist(values, bins=100, range=(0, 0.005),
                histtype="step", linewidth=2, color="orange",
                label=f"Tilted Barrel (N={n_stubs})")
        if n_stubs > 0:
            mean_err = np.mean(values)
            std_err = np.std(values)
            ax.axvline(mean_err, color="black", linestyle="--", alpha=0.7)
            stats_text = f"N={n_stubs}\nmean={mean_err:.5f}\nstd={std_err:.5f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDrError [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDrError - Tilted Barrel")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Tilted Barrel dr_sensor - may have different distribution
    ax = axes[1, 2]
    values = tilted_barrel["dr_sensor"]
    n_stubs = len(values)
    ax.hist(values, bins=50, range=(0, 5),
            histtype="step", linewidth=2, color="orange",
            label=f"Tilted dr (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: Tilted Barrel dr_sensor\n(tilted geometry affects dr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dPhiDr significance (|dPhiDr|/error) for tilted barrel
    ax = axes[1, 3]
    if "dPhiDrError" in tilted_barrel:
        valid_mask = tilted_barrel["dPhiDrError"] > 0
        if np.sum(valid_mask) > 0:
            nsigma = np.abs(tilted_barrel["dPhiDr"][valid_mask]) / tilted_barrel["dPhiDrError"][valid_mask]
            n_stubs = len(nsigma)
            ax.hist(nsigma, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="orange",
                    label=f"Tilted (N={n_stubs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
            # Count fraction within 5 sigma
            within_5sigma = np.sum(nsigma < 5.0) / len(nsigma) * 100
            stats_text = f"N={n_stubs}\nmean={np.mean(nsigma):.3f}\nstd={np.std(nsigma):.3f}\n{within_5sigma:.1f}% < 5 sigma"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("|dPhiDr| / dPhiDrError")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 1: dPhiDr Significance - Tilted\n(compare with flat)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 2: Case 2 - Endcap quantities (dPhiDz and eta consistency)
    # ========================================================================

    # dPhiDz - Endcap (proposed alternative since dz is large)
    ax = axes[2, 0]
    values = endcap["dPhiDz"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.01, 0.01),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.6f}\nstd={np.std(values):.6f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("dPhiDz [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 2: dPhiDz - Endcap\n(well-behaved alternative to dPhiDr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endcap dz_sensor - shows why dPhiDz works
    ax = axes[2, 1]
    values = np.abs(endcap["dz_sensor"])
    n_stubs = len(values)
    ax.hist(values, bins=50, range=(0, 1),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap |dz| (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.4f}\nstd={np.std(values):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("|dz_sensor| [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 2: Endcap |dz_sensor|\n(large dz makes dPhiDz well-defined)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Eta consistency (z/r) - proposed for endcap-endcap
    ax = axes[2, 2]
    values = endcap["eta_diff"]
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-0.005, 0.005),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    mean_eta = np.mean(values)
    std_eta = np.std(values)
    ax.axvline(mean_eta, color="black", linestyle="--", alpha=0.7)
    stats_text = f"N={n_stubs}\nmean={mean_eta:.6f}\nstd={std_eta:.6f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("eta_outer - eta_inner (where eta = z/r)")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 2: Eta Consistency - Endcap\n(tracks from origin have constant z/r)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Raw bend angle for endcap (not divided by dr)
    ax = axes[2, 3]
    values = endcap["bend"] * 1000
    n_stubs = len(values)
    ax.hist(values, bins=100, range=(-3, 3),
            histtype="step", linewidth=2, color="red",
            label=f"Endcap (N={n_stubs})")
    if n_stubs > 0:
        stats_text = f"N={n_stubs}\nmean={np.mean(values):.3f}\nstd={np.std(values):.3f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("bend [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 2: Raw Bend Angle - Endcap\n(use dphi directly, not dphi/dr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 3: Case 3 - Transition region quantities (bend sign check)
    # ========================================================================

    # Bend sign - for transition region (sign consistency check)
    ax = axes[3, 0]
    bend_sign_barrel = np.sign(barrel["bend"])
    bend_sign_endcap = np.sign(endcap["bend"])
    bins = [-1.5, -0.5, 0.5, 1.5]
    ax.hist(bend_sign_barrel, bins=bins, histtype="step", linewidth=2, color="blue",
            alpha=0.8, label=f"Barrel (N={len(bend_sign_barrel)})")
    ax.hist(bend_sign_endcap, bins=bins, histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Endcap (N={len(bend_sign_endcap)})")
    ax.set_xlabel("sign(bend)")
    ax.set_ylabel("Stubs")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1 (negative)", "0", "+1 (positive)"])
    ax.set_title("Case 3: Bend Sign\n(for barrel-endcap transition checks)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # L1T-style geometric correction factor z/r
    ax = axes[3, 1]
    z_over_r_barrel = np.abs(barrel["stub_z"]) / barrel["stub_r"]
    z_over_r_endcap = np.abs(endcap["stub_z"]) / endcap["stub_r"]
    ax.hist(z_over_r_barrel, bins=100, range=(0, 5),
            histtype="step", linewidth=2, color="blue",
            alpha=0.8, label=f"Barrel (N={len(z_over_r_barrel)})")
    ax.hist(z_over_r_endcap, bins=100, range=(0, 5),
            histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Endcap (N={len(z_over_r_endcap)})")
    stats_text = f"Barrel: mean={np.mean(z_over_r_barrel):.3f}, std={np.std(z_over_r_barrel):.3f}\nEndcap: mean={np.mean(z_over_r_endcap):.3f}, std={np.std(z_over_r_endcap):.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("|z|/r")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 3: |z|/r Ratio\n(L1T geometric correction factor)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Corrected bend using L1T formula: bend_corrected = bend * (z/r) for endcap
    ax = axes[3, 2]
    # For endcap, apply the z/r correction from Aggleton paper
    bend_corrected_endcap = endcap["bend"] * np.abs(endcap["stub_z"]) / endcap["stub_r"]
    bend_barrel = barrel["bend"] * 1000
    bend_corrected_endcap_mrad = bend_corrected_endcap * 1000
    ax.hist(bend_barrel, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="blue",
            alpha=0.8, label=f"Barrel raw (N={len(bend_barrel)})")
    ax.hist(bend_corrected_endcap_mrad, bins=100, range=(-5, 5),
            histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Endcap corrected (N={len(bend_corrected_endcap_mrad)})")
    stats_text = f"Barrel: mean={np.mean(bend_barrel):.3f}, std={np.std(bend_barrel):.3f}\nEndcap: mean={np.mean(bend_corrected_endcap_mrad):.3f}, std={np.std(bend_corrected_endcap_mrad):.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("bend (or bend*z/r for endcap) [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 3: L1T-Style Corrected Bend\n(endcap: bend * z/r correction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Forward+ vs Forward- bend comparison
    ax = axes[3, 3]
    ax.hist(forward["bend"] * 1000, bins=100, range=(-3, 3),
            histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Forward+ z>0 (N={len(forward['bend'])})")
    ax.hist(backward["bend"] * 1000, bins=100, range=(-3, 3),
            histtype="step", linewidth=2, color="green",
            alpha=0.8, label=f"Forward- z<0 (N={len(backward['bend'])})")
    ax.set_xlabel("bend [mrad]")
    ax.set_ylabel("Stubs")
    ax.set_title("Case 3: Forward+/- Bend Comparison\n(for asymmetric z handling)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 4: Comparison plots showing the problem and solutions
    # ========================================================================

    # The problem: dPhiDr flat vs tilted vs endcap
    ax = axes[4, 0]
    ax.hist(flat_barrel["dPhiDr"], bins=100, range=(-0.02, 0.02),
            histtype="step", linewidth=2, color="blue",
            alpha=0.8, label=f"Flat (N={len(flat_barrel['dPhiDr'])})")
    ax.hist(tilted_barrel["dPhiDr"], bins=100, range=(-0.02, 0.02),
            histtype="step", linewidth=2, color="orange",
            alpha=0.8, label=f"Tilted (N={len(tilted_barrel['dPhiDr'])})")
    ax.hist(endcap["dPhiDr"], bins=100, range=(-0.02, 0.02),
            histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Endcap (N={len(endcap['dPhiDr'])})")
    ax.set_xlabel("dPhiDr [rad/cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr Comparison\n(Flat vs Tilted vs Endcap)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # dr_sensor comparison - flat vs tilted vs endcap
    ax = axes[4, 1]
    ax.hist(flat_barrel["dr_sensor"], bins=50, range=(-0.5, 5),
            histtype="step", linewidth=2, color="blue",
            alpha=0.8, label=f"Flat (N={len(flat_barrel['dr_sensor'])})")
    ax.hist(tilted_barrel["dr_sensor"], bins=50, range=(-0.5, 5),
            histtype="step", linewidth=2, color="orange",
            alpha=0.8, label=f"Tilted (N={len(tilted_barrel['dr_sensor'])})")
    ax.hist(endcap["dr_sensor"], bins=50, range=(-0.5, 5),
            histtype="step", linewidth=2, color="red",
            alpha=0.8, label=f"Endcap (N={len(endcap['dr_sensor'])})")
    ax.axvline(0.1, color="black", linestyle="--", linewidth=2, alpha=0.7, label="~0.1cm threshold")
    ax.set_xlabel("dr_sensor [cm]")
    ax.set_ylabel("Stubs")
    ax.set_title("Sensor dr Comparison\n(Flat vs Tilted geometry)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # dPhiDr significance comparison - flat vs tilted
    ax = axes[4, 2]
    if "dPhiDrError" in flat_barrel and "dPhiDrError" in tilted_barrel:
        valid_flat = flat_barrel["dPhiDrError"] > 0
        valid_tilted = tilted_barrel["dPhiDrError"] > 0
        if np.sum(valid_flat) > 0:
            nsigma_flat = np.abs(flat_barrel["dPhiDr"][valid_flat]) / flat_barrel["dPhiDrError"][valid_flat]
            ax.hist(nsigma_flat, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="blue", alpha=0.8,
                    label=f"Flat (N={len(nsigma_flat)})")
        if np.sum(valid_tilted) > 0:
            nsigma_tilted = np.abs(tilted_barrel["dPhiDr"][valid_tilted]) / tilted_barrel["dPhiDrError"][valid_tilted]
            ax.hist(nsigma_tilted, bins=100, range=(0, 10),
                    histtype="step", linewidth=2, color="orange", alpha=0.8,
                    label=f"Tilted (N={len(nsigma_tilted)})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
    ax.set_xlabel("|dPhiDr| / dPhiDrError")
    ax.set_ylabel("Stubs")
    ax.set_title("dPhiDr Significance\n(Flat vs Tilted comparison)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Summary: which quantity to use where
    ax = axes[4, 3]
    ax.axis("off")
    summary_text = """
    RECOMMENDED COMPATIBILITY CHECKS:

    Case 1: Barrel-Barrel (dr > 0)
      - Use dPhiDr with nSigma cut
      - |dPhiDr_doublet - dPhiDr_stub| < 5*error

    Case 2: Endcap-Endcap (dr ~ 0)
      - Option A: Skip dPhiDr check entirely
      - Option B: Use eta (z/r) consistency
      - Option C: Use dPhiDz instead

    Case 3: Barrel-Endcap Transition
      - Use bend sign check only
      - sign(dphi_doublet) == sign(bend_stub)
      - Optional: L1T-style z/r correction

    Key Insight: dPhiDr = dphi/dr fails when dr ~ 0
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    # ========================================================================
    # Row 5: Pairwise dPhiDr significance with correctly combined errors
    #
    # The per-stub significance |dPhiDr_i|/dPhiDrError_i (rows 0-1, 4) uses
    # only one stub's error. For doublet compatibility the relevant quantity is
    # the significance of the DIFFERENCE between two stubs:
    #   |dPhiDr_i - dPhiDr_j| / sqrt(dPhiDrError_i^2 + dPhiDrError_j^2)
    # This row computes that pairwise significance for same-event stub pairs
    # from consecutive barrel layers.
    # ========================================================================

    # Helper: compute pairwise dPhiDr significance with combined errors
    def compute_pairwise_dPhiDr_significance(data, mask1, mask2):
        """
        For all same-event stub pairs (i from mask1, j from mask2), compute
        |dPhiDr_i - dPhiDr_j| / sqrt(dPhiDrError_i^2 + dPhiDrError_j^2).

        Returns a 1-D numpy array of significance values.
        """
        if "dPhiDrError" not in data:
            return np.array([])

        events = data["event"]
        dPhiDr_arr = data["dPhiDr"]
        dPhiDrErr_arr = data["dPhiDrError"]

        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            return np.array([])

        events1, events2 = events[idx1], events[idx2]
        dPhiDr1, dPhiDr2 = dPhiDr_arr[idx1], dPhiDr_arr[idx2]
        err1, err2 = dPhiDrErr_arr[idx1], dPhiDrErr_arr[idx2]

        unique_events = np.intersect1d(np.unique(events1), np.unique(events2))

        sig_list = []
        for evt in unique_events:
            m1 = events1 == evt
            m2 = events2 == evt
            d1, d2 = dPhiDr1[m1], dPhiDr2[m2]
            e1, e2 = err1[m1], err2[m2]
            for i in range(len(d1)):
                for j in range(len(d2)):
                    combined_err = np.sqrt(e1[i]**2 + e2[j]**2)
                    if combined_err > 0:
                        sig_list.append(np.abs(d1[i] - d2[j]) / combined_err)
        return np.array(sig_list)

    # Build masks for barrel layers (flat / tilted)
    is_barrel_mask = data["isBarrel"] == True
    is_flat_mask = data["isFlat"] == True
    is_tilted_mask = data["isTilted"] == True
    layer_arr = data["layer"]

    # Collect pairwise significance for consecutive flat barrel layer pairs
    flat_pair_sig = []
    for l_inner in range(1, 6):
        m1 = is_barrel_mask & is_flat_mask & (layer_arr == l_inner)
        m2 = is_barrel_mask & is_flat_mask & (layer_arr == l_inner + 1)
        sig = compute_pairwise_dPhiDr_significance(data, m1, m2)
        if len(sig) > 0:
            flat_pair_sig.append(sig)
    flat_pair_sig = np.concatenate(flat_pair_sig) if flat_pair_sig else np.array([])

    # Collect pairwise significance for consecutive tilted barrel layer pairs
    tilted_pair_sig = []
    for l_inner in range(1, 4):
        m1 = is_barrel_mask & is_tilted_mask & (layer_arr == l_inner)
        m2 = is_barrel_mask & is_tilted_mask & (layer_arr == l_inner + 1)
        sig = compute_pairwise_dPhiDr_significance(data, m1, m2)
        if len(sig) > 0:
            tilted_pair_sig.append(sig)
    tilted_pair_sig = np.concatenate(tilted_pair_sig) if tilted_pair_sig else np.array([])

    # Pairwise significance - Flat Barrel
    ax = axes[5, 0]
    if len(flat_pair_sig) > 0:
        n_pairs = len(flat_pair_sig)
        ax.hist(flat_pair_sig, bins=100, range=(0, 10),
                histtype="step", linewidth=2, color="blue",
                label=f"Flat pairs (N={n_pairs})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
        within_5sigma = np.sum(flat_pair_sig < 5.0) / n_pairs * 100
        stats_text = (f"N={n_pairs}\nmean={np.mean(flat_pair_sig):.3f}\n"
                      f"std={np.std(flat_pair_sig):.3f}\n{within_5sigma:.1f}% < 5 sigma")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel(r"|$\Delta$dPhiDr| / $\sqrt{\sigma_i^2 + \sigma_j^2}$")
    ax.set_ylabel("Stub pairs")
    ax.set_title("Pairwise Significance - Flat Barrel\n(combined error from both stubs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pairwise significance - Tilted Barrel
    ax = axes[5, 1]
    if len(tilted_pair_sig) > 0:
        n_pairs = len(tilted_pair_sig)
        ax.hist(tilted_pair_sig, bins=100, range=(0, 10),
                histtype="step", linewidth=2, color="orange",
                label=f"Tilted pairs (N={n_pairs})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
        within_5sigma = np.sum(tilted_pair_sig < 5.0) / n_pairs * 100
        stats_text = (f"N={n_pairs}\nmean={np.mean(tilted_pair_sig):.3f}\n"
                      f"std={np.std(tilted_pair_sig):.3f}\n{within_5sigma:.1f}% < 5 sigma")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel(r"|$\Delta$dPhiDr| / $\sqrt{\sigma_i^2 + \sigma_j^2}$")
    ax.set_ylabel("Stub pairs")
    ax.set_title("Pairwise Significance - Tilted Barrel\n(combined error from both stubs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pairwise significance comparison - Flat vs Tilted overlay
    ax = axes[5, 2]
    if len(flat_pair_sig) > 0:
        ax.hist(flat_pair_sig, bins=100, range=(0, 10),
                histtype="step", linewidth=2, color="blue", alpha=0.8,
                label=f"Flat (N={len(flat_pair_sig)})")
    if len(tilted_pair_sig) > 0:
        ax.hist(tilted_pair_sig, bins=100, range=(0, 10),
                histtype="step", linewidth=2, color="orange", alpha=0.8,
                label=f"Tilted (N={len(tilted_pair_sig)})")
    ax.axvline(5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="5-sigma cut")
    ax.set_xlabel(r"|$\Delta$dPhiDr| / $\sqrt{\sigma_i^2 + \sigma_j^2}$")
    ax.set_ylabel("Stub pairs")
    ax.set_title("Pairwise Significance Comparison\n(Flat vs Tilted with combined errors)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Explanatory panel: single-stub vs pairwise significance
    ax = axes[5, 3]
    ax.axis("off")
    explanation_text = """
    SINGLE-STUB vs PAIRWISE SIGNIFICANCE:

    Rows 0-1 (per-stub):
      nsigma = |dPhiDr_i| / dPhiDrError_i
      Uses only one stub's error.

    Row 5 (pairwise, this row):
      nsigma = |dPhiDr_i - dPhiDr_j|
             / sqrt(err_i^2 + err_j^2)
      Uses BOTH stubs' errors combined
      in quadrature (correct for the
      error on a difference).

    The pairwise version is the proper
    significance for doublet compatibility
    cuts in the CA algorithm.
    """
    ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    plt.tight_layout()
    return fig


def plot_cuts_per_layer_pair(data, figsize_per_group=(18, 20)):
    """
    Plot cut variable distributions per layer pair for both current CA implementation
    and proposed alternatives.

    This function creates separate figures for each layer pair group, showing the
    distributions of cut variables for stubs in each layer of each pair.

    Layer pair groups:
    - Group A1: Flat Barrel -> Flat Barrel (L1f->L2f, L2f->L3f, L3f->L4f, L4f->L5f, L5f->L6f)
    - Group A2: Tilted Barrel -> Tilted Barrel (L1t->L2t, L2t->L3t, L3t->L4t)
    - Group A3: Mixed Flat <-> Tilted Barrel transitions
    - Group B1: Flat Barrel -> Forward+ D1 (L1f->D1+, L2f->D1+, L3f->D1+, L4f->D1+, L5f->D1+, L6f->D1+)
    - Group B2: Tilted Barrel -> Forward+ D1 (L1t->D1+, L2t->D1+, L3t->D1+, L4t->D1+)
    - Group C1: Flat Barrel -> Forward- D1 (L1f->D1-, L2f->D1-, L3f->D1-, L4f->D1-, L5f->D1-, L6f->D1-)
    - Group C2: Tilted Barrel -> Forward- D1 (L1t->D1-, L2t->D1-, L3t->D1-, L4t->D1-)
    - Group D: Forward+ consecutive disks (D1+->D2+, D2+->D3+, D3+->D4+, D4+->D5+)
    - Group E: Forward- consecutive disks (D1-->D2-, D2-->D3-, D3-->D4-, D4-->D5-)

    Variables plotted:
    1. Current CA cuts: dPhiDr, dPhiDr significance (|dPhiDr|/dPhiDrError)
    2. Proposed alternatives: dPhiDz, eta_diff, bend

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize_per_group: Figure size tuple for each group figure

    Returns:
        Dictionary of matplotlib figures keyed by group name
    """
    figs = {}

    # Define layer pair groups
    # Each entry: (group_name, group_title, list of (layer1_name, layer1_selector, layer2_name, layer2_selector))
    # Group A is split into three subgroups for flat/tilted barrel analysis:
    # - A1: Flat barrel -> Flat barrel
    # - A2: Tilted barrel -> Tilted barrel
    # - A3: Mixed (Flat <-> Tilted) transitions
    layer_pair_groups = {
        "A1_flat_flat": {
            "title": "Group A1: Flat Barrel -> Flat Barrel Consecutive Layers",
            "pairs": [
                ("L1f", lambda d: select_flat_barrel_layer(d, 1), "L2f", lambda d: select_flat_barrel_layer(d, 2)),
                ("L2f", lambda d: select_flat_barrel_layer(d, 2), "L3f", lambda d: select_flat_barrel_layer(d, 3)),
                ("L3f", lambda d: select_flat_barrel_layer(d, 3), "L4f", lambda d: select_flat_barrel_layer(d, 4)),
                ("L4f", lambda d: select_flat_barrel_layer(d, 4), "L5f", lambda d: select_flat_barrel_layer(d, 5)),
                ("L5f", lambda d: select_flat_barrel_layer(d, 5), "L6f", lambda d: select_flat_barrel_layer(d, 6)),
            ]
        },
        "A2_tilted_tilted": {
            "title": "Group A2: Tilted Barrel -> Tilted Barrel Consecutive Layers",
            "pairs": [
                ("L1t", lambda d: select_tilted_barrel_layer(d, 1), "L2t", lambda d: select_tilted_barrel_layer(d, 2)),
                ("L2t", lambda d: select_tilted_barrel_layer(d, 2), "L3t", lambda d: select_tilted_barrel_layer(d, 3)),
                ("L3t", lambda d: select_tilted_barrel_layer(d, 3), "L4t", lambda d: select_tilted_barrel_layer(d, 4)),
            ]
        },
        "A3_mixed_flat_tilted": {
            "title": "Group A3: Mixed Flat <-> Tilted Barrel Transitions",
            "pairs": [
                ("L1f", lambda d: select_flat_barrel_layer(d, 1), "L2t", lambda d: select_tilted_barrel_layer(d, 2)),
                ("L1t", lambda d: select_tilted_barrel_layer(d, 1), "L2f", lambda d: select_flat_barrel_layer(d, 2)),
                ("L2f", lambda d: select_flat_barrel_layer(d, 2), "L3t", lambda d: select_tilted_barrel_layer(d, 3)),
                ("L2t", lambda d: select_tilted_barrel_layer(d, 2), "L3f", lambda d: select_flat_barrel_layer(d, 3)),
                ("L3f", lambda d: select_flat_barrel_layer(d, 3), "L4t", lambda d: select_tilted_barrel_layer(d, 4)),
                ("L3t", lambda d: select_tilted_barrel_layer(d, 3), "L4f", lambda d: select_flat_barrel_layer(d, 4)),
            ]
        },
        "B1_flat_barrel_forward_plus": {
            "title": "Group B1: Flat Barrel to Forward+ (D1+)",
            "pairs": [
                ("L1f", lambda d: select_flat_barrel_layer(d, 1), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L2f", lambda d: select_flat_barrel_layer(d, 2), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L3f", lambda d: select_flat_barrel_layer(d, 3), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L4f", lambda d: select_flat_barrel_layer(d, 4), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L5f", lambda d: select_flat_barrel_layer(d, 5), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L6f", lambda d: select_flat_barrel_layer(d, 6), "D1+", lambda d: select_forward_disk(d, 1)),
            ]
        },
        "B2_tilted_barrel_forward_plus": {
            "title": "Group B2: Tilted Barrel to Forward+ (D1+)",
            "pairs": [
                ("L1t", lambda d: select_tilted_barrel_layer(d, 1), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L2t", lambda d: select_tilted_barrel_layer(d, 2), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L3t", lambda d: select_tilted_barrel_layer(d, 3), "D1+", lambda d: select_forward_disk(d, 1)),
                ("L4t", lambda d: select_tilted_barrel_layer(d, 4), "D1+", lambda d: select_forward_disk(d, 1)),
            ]
        },
        "C1_flat_barrel_forward_minus": {
            "title": "Group C1: Flat Barrel to Forward- (D1-)",
            "pairs": [
                ("L1f", lambda d: select_flat_barrel_layer(d, 1), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L2f", lambda d: select_flat_barrel_layer(d, 2), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L3f", lambda d: select_flat_barrel_layer(d, 3), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L4f", lambda d: select_flat_barrel_layer(d, 4), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L5f", lambda d: select_flat_barrel_layer(d, 5), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L6f", lambda d: select_flat_barrel_layer(d, 6), "D1-", lambda d: select_backward_disk(d, 1)),
            ]
        },
        "C2_tilted_barrel_forward_minus": {
            "title": "Group C2: Tilted Barrel to Forward- (D1-)",
            "pairs": [
                ("L1t", lambda d: select_tilted_barrel_layer(d, 1), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L2t", lambda d: select_tilted_barrel_layer(d, 2), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L3t", lambda d: select_tilted_barrel_layer(d, 3), "D1-", lambda d: select_backward_disk(d, 1)),
                ("L4t", lambda d: select_tilted_barrel_layer(d, 4), "D1-", lambda d: select_backward_disk(d, 1)),
            ]
        },
        "D_forward_plus_consecutive": {
            "title": "Group D: Forward+ Consecutive Disks",
            "pairs": [
                ("D1+", lambda d: select_forward_disk(d, 1), "D2+", lambda d: select_forward_disk(d, 2)),
                ("D2+", lambda d: select_forward_disk(d, 2), "D3+", lambda d: select_forward_disk(d, 3)),
                ("D3+", lambda d: select_forward_disk(d, 3), "D4+", lambda d: select_forward_disk(d, 4)),
                ("D4+", lambda d: select_forward_disk(d, 4), "D5+", lambda d: select_forward_disk(d, 5)),
            ]
        },
        "E_forward_minus_consecutive": {
            "title": "Group E: Forward- Consecutive Disks",
            "pairs": [
                ("D1-", lambda d: select_backward_disk(d, 1), "D2-", lambda d: select_backward_disk(d, 2)),
                ("D2-", lambda d: select_backward_disk(d, 2), "D3-", lambda d: select_backward_disk(d, 3)),
                ("D3-", lambda d: select_backward_disk(d, 3), "D4-", lambda d: select_backward_disk(d, 4)),
                ("D4-", lambda d: select_backward_disk(d, 4), "D5-", lambda d: select_backward_disk(d, 5)),
            ]
        },
    }

    # Define variables to plot with their ranges
    # Format: (var_name, xlabel, barrel_range, endcap_range, unit_multiplier, use_abs)
    current_cuts = [
        ("dPhiDr", "dPhiDr [rad/cm]", (-0.01, 0.01), (-0.5, 0.5), 1.0, False),
        ("dPhiDr_significance", "|dPhiDr|/Error", (0, 10), (0, 10), 1.0, False),
    ]

    proposed_cuts = [
        ("dPhiDz", "dPhiDz [rad/cm]", (-0.1, 0.1), (-0.01, 0.01), 1.0, False),
        ("eta_diff", "eta_diff (z/r outer - inner)", (-0.1, 0.1), (-0.01, 0.01), 1.0, False),
        ("bend", "bend [mrad]", (-5, 5), (-5, 5), 1000.0, False),
    ]

    def is_barrel_layer(layer_name):
        """Check if layer name corresponds to a barrel layer (flat or tilted)."""
        return layer_name.startswith("L")

    def is_flat_barrel_layer(layer_name):
        """Check if layer name corresponds to a flat barrel layer."""
        return layer_name.startswith("L") and layer_name.endswith("f")

    def is_tilted_barrel_layer(layer_name):
        """Check if layer name corresponds to a tilted barrel layer."""
        return layer_name.startswith("L") and layer_name.endswith("t")

    def is_endcap_layer(layer_name):
        """Check if layer name corresponds to an endcap disk."""
        return layer_name.startswith("D")

    def get_range(var_name, layer_name, barrel_range, endcap_range):
        """Get appropriate range based on layer type."""
        if is_barrel_layer(layer_name):
            return barrel_range
        else:
            return endcap_range

    def compute_dPhiDr_significance(layer_data):
        """Compute |dPhiDr|/dPhiDrError for a layer."""
        if "dPhiDrError" not in layer_data or len(layer_data["dPhiDr"]) == 0:
            return np.array([])
        valid_mask = layer_data["dPhiDrError"] > 0
        if np.sum(valid_mask) == 0:
            return np.array([])
        return np.abs(layer_data["dPhiDr"][valid_mask]) / layer_data["dPhiDrError"][valid_mask]

    # Process each group
    for group_name, group_info in layer_pair_groups.items():
        pairs = group_info["pairs"]
        n_pairs = len(pairs)

        # Determine total number of variable columns (current + proposed)
        n_current = len(current_cuts)
        n_proposed = len(proposed_cuts)
        n_cols = n_current + n_proposed

        # Create figure with 2 rows per pair (layer1 and layer2) and n_cols columns
        n_rows = n_pairs * 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_group[0], figsize_per_group[1]))
        fig.suptitle(f"{group_info['title']}\n"
                     f"Rows: Layer pair stubs (inner/outer). "
                     f"Cols: Current cuts (dPhiDr, significance) | Proposed (dPhiDz, eta_diff, bend)",
                     fontsize=12, y=1.02)

        for pair_idx, (layer1_name, layer1_selector, layer2_name, layer2_selector) in enumerate(pairs):
            # Get data for each layer
            layer1_data = layer1_selector(data)
            layer2_data = layer2_selector(data)

            # Row indices for this pair
            row1 = pair_idx * 2
            row2 = pair_idx * 2 + 1

            # Colors for the two layers
            # - Flat barrel: blue
            # - Tilted barrel: orange
            # - Flat barrel in B1/C1 groups (barrel->endcap): cyan
            # - Tilted barrel in B2/C2 groups (barrel->endcap): gold
            # - Forward+: red
            # - Forward-: green
            def get_layer_color(layer_name):
                # Check if we're in a barrel->endcap group
                is_barrel_to_endcap_group = (
                    "flat_barrel_forward" in group_name or
                    "tilted_barrel_forward" in group_name
                )
                if is_flat_barrel_layer(layer_name):
                    if is_barrel_to_endcap_group:
                        return "cyan"
                    return "blue"
                elif is_tilted_barrel_layer(layer_name):
                    if is_barrel_to_endcap_group:
                        return "gold"
                    return "orange"
                elif is_barrel_layer(layer_name):
                    return "blue"  # Default barrel color for layers without suffix
                elif "+" in layer_name:
                    return "red"
                else:
                    return "green"
            color1 = get_layer_color(layer1_name)
            color2 = get_layer_color(layer2_name)

            # Plot current cuts
            for col_idx, (var_name, xlabel, barrel_range, endcap_range, multiplier, use_abs) in enumerate(current_cuts):
                # Layer 1
                ax1 = axes[row1, col_idx]
                range1 = get_range(var_name, layer1_name, barrel_range, endcap_range)

                if var_name == "dPhiDr_significance":
                    values1 = compute_dPhiDr_significance(layer1_data)
                    n1 = len(values1)
                else:
                    values1 = layer1_data.get(var_name, np.array([]))
                    if multiplier != 1.0 and len(values1) > 0:
                        values1 = values1 * multiplier
                    if use_abs and len(values1) > 0:
                        values1 = np.abs(values1)
                    n1 = len(values1)

                if n1 > 0:
                    ax1.hist(values1, bins=50, range=range1, histtype="step",
                             linewidth=2, color=color1,
                             label=f"{layer1_name} (N={n1})")
                    if var_name == "dPhiDr_significance":
                        ax1.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
                    stats_text = f"N={n1}\nmean={np.mean(values1):.4f}\nstd={np.std(values1):.4f}"
                    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                             fontsize=6, verticalalignment="top", horizontalalignment="right",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                             ha="center", va="center", fontsize=10)

                ax1.set_xlabel(xlabel, fontsize=8)
                ax1.set_ylabel("Stubs", fontsize=8)
                ax1.set_title(f"{layer1_name}->{layer2_name}: {layer1_name}", fontsize=9)
                ax1.legend(fontsize=7)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis="both", labelsize=7)

                # Layer 2
                ax2 = axes[row2, col_idx]
                range2 = get_range(var_name, layer2_name, barrel_range, endcap_range)

                if var_name == "dPhiDr_significance":
                    values2 = compute_dPhiDr_significance(layer2_data)
                    n2 = len(values2)
                else:
                    values2 = layer2_data.get(var_name, np.array([]))
                    if multiplier != 1.0 and len(values2) > 0:
                        values2 = values2 * multiplier
                    if use_abs and len(values2) > 0:
                        values2 = np.abs(values2)
                    n2 = len(values2)

                if n2 > 0:
                    ax2.hist(values2, bins=50, range=range2, histtype="step",
                             linewidth=2, color=color2,
                             label=f"{layer2_name} (N={n2})")
                    if var_name == "dPhiDr_significance":
                        ax2.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
                    stats_text = f"N={n2}\nmean={np.mean(values2):.4f}\nstd={np.std(values2):.4f}"
                    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                             fontsize=6, verticalalignment="top", horizontalalignment="right",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes,
                             ha="center", va="center", fontsize=10)

                ax2.set_xlabel(xlabel, fontsize=8)
                ax2.set_ylabel("Stubs", fontsize=8)
                ax2.set_title(f"{layer1_name}->{layer2_name}: {layer2_name}", fontsize=9)
                ax2.legend(fontsize=7)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis="both", labelsize=7)

            # Plot proposed cuts
            for col_offset, (var_name, xlabel, barrel_range, endcap_range, multiplier, use_abs) in enumerate(proposed_cuts):
                col_idx = n_current + col_offset

                # Layer 1
                ax1 = axes[row1, col_idx]
                range1 = get_range(var_name, layer1_name, barrel_range, endcap_range)
                values1 = layer1_data.get(var_name, np.array([]))
                if multiplier != 1.0 and len(values1) > 0:
                    values1 = values1 * multiplier
                if use_abs and len(values1) > 0:
                    values1 = np.abs(values1)
                n1 = len(values1)

                if n1 > 0:
                    ax1.hist(values1, bins=50, range=range1, histtype="step",
                             linewidth=2, color=color1,
                             label=f"{layer1_name} (N={n1})")
                    stats_text = f"N={n1}\nmean={np.mean(values1):.4f}\nstd={np.std(values1):.4f}"
                    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                             fontsize=6, verticalalignment="top", horizontalalignment="right",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                             ha="center", va="center", fontsize=10)

                ax1.set_xlabel(xlabel, fontsize=8)
                ax1.set_ylabel("Stubs", fontsize=8)
                ax1.set_title(f"{layer1_name}->{layer2_name}: {layer1_name}", fontsize=9)
                ax1.legend(fontsize=7)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis="both", labelsize=7)

                # Layer 2
                ax2 = axes[row2, col_idx]
                range2 = get_range(var_name, layer2_name, barrel_range, endcap_range)
                values2 = layer2_data.get(var_name, np.array([]))
                if multiplier != 1.0 and len(values2) > 0:
                    values2 = values2 * multiplier
                if use_abs and len(values2) > 0:
                    values2 = np.abs(values2)
                n2 = len(values2)

                if n2 > 0:
                    ax2.hist(values2, bins=50, range=range2, histtype="step",
                             linewidth=2, color=color2,
                             label=f"{layer2_name} (N={n2})")
                    stats_text = f"N={n2}\nmean={np.mean(values2):.4f}\nstd={np.std(values2):.4f}"
                    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                             fontsize=6, verticalalignment="top", horizontalalignment="right",
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes,
                             ha="center", va="center", fontsize=10)

                ax2.set_xlabel(xlabel, fontsize=8)
                ax2.set_ylabel("Stubs", fontsize=8)
                ax2.set_title(f"{layer1_name}->{layer2_name}: {layer2_name}", fontsize=9)
                ax2.legend(fontsize=7)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis="both", labelsize=7)

        plt.tight_layout()
        figs[group_name] = fig

    return figs


def plot_cut_differences_per_layer_pair(data, figsize_per_group=(18, 14)):
    """
    Plot DIFFERENCES of cut variables between stubs from consecutive layer pairs.

    For each layer pair, finds all events that have stubs in both layers, then
    computes pairwise differences of cut variables between all stub pairs from
    those consecutive layers within the same event.

    This shows the actual cut distribution - what you would compare when forming
    doublets in the Cellular Automaton.

    Layer pair groups:
    - Group A1: Flat Barrel -> Flat Barrel (L1f->L2f, L2f->L3f, L3f->L4f, L4f->L5f, L5f->L6f)
    - Group A2: Tilted Barrel -> Tilted Barrel (L1t->L2t, L2t->L3t, L3t->L4t)
    - Group A3: Mixed Flat <-> Tilted Barrel transitions
    - Group B1: Flat Barrel -> Forward+ D1 (L1f->D1+, L2f->D1+, L3f->D1+, L4f->D1+, L5f->D1+, L6f->D1+)
    - Group B2: Tilted Barrel -> Forward+ D1 (L1t->D1+, L2t->D1+, L3t->D1+, L4t->D1+)
    - Group C1: Flat Barrel -> Forward- D1 (L1f->D1-, L2f->D1-, L3f->D1-, L4f->D1-, L5f->D1-, L6f->D1-)
    - Group C2: Tilted Barrel -> Forward- D1 (L1t->D1-, L2t->D1-, L3t->D1-, L4t->D1-)
    - Group D: Forward+ consecutive disks (D1+->D2+, D2+->D3+, D3+->D4+, D4+->D5+)
    - Group E: Forward- consecutive disks (D1-->D2-, D2-->D3-, D3-->D4-, D4-->D5-)

    Variables computed as differences (outer - inner):
    - Delta(dPhiDr): Current CA cut variable difference
    - Delta(dPhiDz): Proposed alternative difference
    - Delta(eta): eta_outer - eta_inner where eta = z/r (stub position based)
    - Delta(bend): bend_outer - bend_inner

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize_per_group: Figure size tuple for each group figure

    Returns:
        Dictionary of matplotlib figures keyed by group name
    """
    figs = {}

    # Get basic arrays from data
    events = data["event"]
    is_barrel = data["isBarrel"]
    is_endcap = data["isEndcap"]
    stub_z = data["stub_z"]
    stub_r = data["stub_r"]
    layer = data["layer"]
    dPhiDr = data["dPhiDr"]
    dPhiDz = data["dPhiDz"]
    bend = data["bend"]

    # Compute eta = z/r for each stub (this is the stub position-based eta)
    eta = stub_z / stub_r

    def compute_layer_pair_differences(mask1, mask2):
        """
        Compute differences of cut variables for all stub pairs from two layer
        selections within the same event.

        Args:
            mask1: Boolean mask for inner layer selection
            mask2: Boolean mask for outer layer selection

        Returns:
            Dictionary with difference arrays for each variable
        """
        # Get indices for each layer
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return {
                'delta_dPhiDr': np.array([]),
                'delta_dPhiDz': np.array([]),
                'delta_eta': np.array([]),
                'delta_bend': np.array([])
            }

        # Get values for each layer
        events1 = events[idx1]
        events2 = events[idx2]
        dPhiDr1 = dPhiDr[idx1]
        dPhiDr2 = dPhiDr[idx2]
        dPhiDz1 = dPhiDz[idx1]
        dPhiDz2 = dPhiDz[idx2]
        eta1 = eta[idx1]
        eta2 = eta[idx2]
        bend1 = bend[idx1]
        bend2 = bend[idx2]

        # Find unique events that have stubs in both layers
        unique_events = np.intersect1d(np.unique(events1), np.unique(events2))

        delta_dPhiDr_list = []
        delta_dPhiDz_list = []
        delta_eta_list = []
        delta_bend_list = []

        for evt in unique_events:
            # Get values for this event in each layer
            evt_mask1 = events1 == evt
            evt_mask2 = events2 == evt

            dPhiDr1_evt = dPhiDr1[evt_mask1]
            dPhiDr2_evt = dPhiDr2[evt_mask2]
            dPhiDz1_evt = dPhiDz1[evt_mask1]
            dPhiDz2_evt = dPhiDz2[evt_mask2]
            eta1_evt = eta1[evt_mask1]
            eta2_evt = eta2[evt_mask2]
            bend1_evt = bend1[evt_mask1]
            bend2_evt = bend2[evt_mask2]

            # Compute differences for all pairs (outer - inner)
            for i in range(len(dPhiDr1_evt)):
                for j in range(len(dPhiDr2_evt)):
                    delta_dPhiDr_list.append(dPhiDr2_evt[j] - dPhiDr1_evt[i])
                    delta_dPhiDz_list.append(dPhiDz2_evt[j] - dPhiDz1_evt[i])
                    delta_eta_list.append(eta2_evt[j] - eta1_evt[i])
                    delta_bend_list.append(bend2_evt[j] - bend1_evt[i])

        return {
            'delta_dPhiDr': np.array(delta_dPhiDr_list),
            'delta_dPhiDz': np.array(delta_dPhiDz_list),
            'delta_eta': np.array(delta_eta_list),
            'delta_bend': np.array(delta_bend_list)
        }

    # Get flat/tilted masks for barrel splitting
    is_flat = data["isFlat"]
    is_tilted = data["isTilted"]

    # Define layer pair groups with masks
    # Group A is split into three subgroups for flat/tilted barrel analysis:
    # - A1: Flat barrel -> Flat barrel
    # - A2: Tilted barrel -> Tilted barrel
    # - A3: Mixed (Flat <-> Tilted) transitions
    layer_pair_groups = {
        "A1_flat_flat": {
            "title": "Group A1: Flat Barrel -> Flat Barrel - Cut Variable Differences",
            "pairs": [
                ("L1f->L2f", is_barrel & is_flat & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3f", is_barrel & is_flat & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4f", is_barrel & is_flat & (layer == 3), is_barrel & is_flat & (layer == 4)),
                ("L4f->L5f", is_barrel & is_flat & (layer == 4), is_barrel & is_flat & (layer == 5)),
                ("L5f->L6f", is_barrel & is_flat & (layer == 5), is_barrel & is_flat & (layer == 6)),
            ]
        },
        "A2_tilted_tilted": {
            "title": "Group A2: Tilted Barrel -> Tilted Barrel - Cut Variable Differences",
            "pairs": [
                ("L1t->L2t", is_barrel & is_tilted & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L2t->L3t", is_barrel & is_tilted & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L3t->L4t", is_barrel & is_tilted & (layer == 3), is_barrel & is_tilted & (layer == 4)),
            ]
        },
        "A3_mixed_flat_tilted": {
            "title": "Group A3: Mixed Flat <-> Tilted Barrel - Cut Variable Differences",
            "pairs": [
                ("L1f->L2t", is_barrel & is_flat & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L1t->L2f", is_barrel & is_tilted & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3t", is_barrel & is_flat & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L2t->L3f", is_barrel & is_tilted & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4t", is_barrel & is_flat & (layer == 3), is_barrel & is_tilted & (layer == 4)),
                ("L3t->L4f", is_barrel & is_tilted & (layer == 3), is_barrel & is_flat & (layer == 4)),
            ]
        },
        "B1_flat_barrel_forward_plus": {
            "title": "Group B1: Flat Barrel to Forward+ (D1+) - Cut Variable Differences",
            "pairs": [
                ("L1f->D1+", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2f->D1+", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3f->D1+", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4f->D1+", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L5f->D1+", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L6f->D1+", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "B2_tilted_barrel_forward_plus": {
            "title": "Group B2: Tilted Barrel to Forward+ (D1+) - Cut Variable Differences",
            "pairs": [
                ("L1t->D1+", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2t->D1+", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3t->D1+", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4t->D1+", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "C1_flat_barrel_forward_minus": {
            "title": "Group C1: Flat Barrel to Forward- (D1-) - Cut Variable Differences",
            "pairs": [
                ("L1f->D1-", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2f->D1-", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3f->D1-", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4f->D1-", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L5f->D1-", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L6f->D1-", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "C2_tilted_barrel_forward_minus": {
            "title": "Group C2: Tilted Barrel to Forward- (D1-) - Cut Variable Differences",
            "pairs": [
                ("L1t->D1-", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2t->D1-", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3t->D1-", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4t->D1-", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "D_forward_plus_consecutive": {
            "title": "Group D: Forward+ Consecutive Disks - Cut Variable Differences",
            "pairs": [
                ("D1+->D2+", is_endcap & (stub_z > 0) & (layer == 1), is_endcap & (stub_z > 0) & (layer == 2)),
                ("D2+->D3+", is_endcap & (stub_z > 0) & (layer == 2), is_endcap & (stub_z > 0) & (layer == 3)),
                ("D3+->D4+", is_endcap & (stub_z > 0) & (layer == 3), is_endcap & (stub_z > 0) & (layer == 4)),
                ("D4+->D5+", is_endcap & (stub_z > 0) & (layer == 4), is_endcap & (stub_z > 0) & (layer == 5)),
            ]
        },
        "E_forward_minus_consecutive": {
            "title": "Group E: Forward- Consecutive Disks - Cut Variable Differences",
            "pairs": [
                ("D1-->D2-", is_endcap & (stub_z < 0) & (layer == 1), is_endcap & (stub_z < 0) & (layer == 2)),
                ("D2-->D3-", is_endcap & (stub_z < 0) & (layer == 2), is_endcap & (stub_z < 0) & (layer == 3)),
                ("D3-->D4-", is_endcap & (stub_z < 0) & (layer == 3), is_endcap & (stub_z < 0) & (layer == 4)),
                ("D4-->D5-", is_endcap & (stub_z < 0) & (layer == 4), is_endcap & (stub_z < 0) & (layer == 5)),
            ]
        },
    }

    # Define variables to plot with their properties
    # Format: (key, label, range, unit_multiplier)
    variables = [
        ('delta_dPhiDr', r'$\Delta$(dPhiDr) [rad/cm]', (-0.02, 0.02), 1.0),
        ('delta_dPhiDz', r'$\Delta$(dPhiDz) [rad/cm]', (-0.1, 0.1), 1.0),
        ('delta_eta', r'$\Delta$(eta) = $\Delta$(z/r)', (-0.5, 0.5), 1.0),
        ('delta_bend', r'$\Delta$(bend) [mrad]', (-5, 5), 1000.0),
    ]

    # Colors for different layer pair types
    colors = {
        "barrel_barrel": "blue",
        "barrel_fwd_plus": "red",
        "barrel_fwd_minus": "green",
        "flat_barrel_fwd_plus": "cyan",
        "flat_barrel_fwd_minus": "lightgreen",
        "tilted_barrel_fwd_plus": "gold",
        "tilted_barrel_fwd_minus": "goldenrod",
        "fwd_plus": "darkorange",
        "fwd_minus": "purple",
    }

    def get_pair_color(group_name):
        if "flat_flat" in group_name or "tilted_tilted" in group_name or "mixed_flat_tilted" in group_name:
            return colors["barrel_barrel"]
        elif "flat_barrel_forward_plus" in group_name:
            return colors["flat_barrel_fwd_plus"]
        elif "flat_barrel_forward_minus" in group_name:
            return colors["flat_barrel_fwd_minus"]
        elif "tilted_barrel_forward_plus" in group_name:
            return colors["tilted_barrel_fwd_plus"]
        elif "tilted_barrel_forward_minus" in group_name:
            return colors["tilted_barrel_fwd_minus"]
        elif "forward_plus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_plus"]
        elif "forward_minus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_minus"]
        elif "forward_plus" in group_name:
            return colors["fwd_plus"]
        else:
            return colors["fwd_minus"]

    # Process each group
    print("Computing cut variable differences between stubs from consecutive layers...")
    for group_name, group_info in layer_pair_groups.items():
        pairs = group_info["pairs"]
        n_pairs = len(pairs)
        n_vars = len(variables)

        # Create figure: one row per layer pair, one column per variable
        fig, axes = plt.subplots(n_pairs, n_vars, figsize=figsize_per_group)
        fig.suptitle(f"{group_info['title']}\n"
                     f"(Differences computed for all stub pairs from consecutive layers in same event)",
                     fontsize=12, y=1.02)

        color = get_pair_color(group_name)

        for pair_idx, (pair_name, mask1, mask2) in enumerate(pairs):
            print(f"  Processing {pair_name}...")
            differences = compute_layer_pair_differences(mask1, mask2)
            n_pairs_found = len(differences['delta_dPhiDr'])
            print(f"    Found {n_pairs_found} stub pairs")

            for var_idx, (var_key, var_label, var_range, multiplier) in enumerate(variables):
                # Handle single row case (axes is 1D)
                if n_pairs == 1:
                    ax = axes[var_idx]
                else:
                    ax = axes[pair_idx, var_idx]

                values = differences[var_key]
                if multiplier != 1.0 and len(values) > 0:
                    values = values * multiplier

                if len(values) > 0:
                    ax.hist(values, bins=100, range=var_range,
                            histtype="step", linewidth=1.5, color=color)

                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ax.axvline(mean_val, color="black", linestyle="--", alpha=0.5)
                    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
                    stats_text = f"N={len(values)}\nmean={mean_val:.4f}\nstd={std_val:.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=7, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10)

                ax.set_xlabel(var_label, fontsize=8)
                ax.set_ylabel("Stub pairs", fontsize=8)
                ax.set_title(f"{pair_name}", fontsize=9)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figs[f"cut_differences_{group_name}"] = fig

    return figs


def plot_relative_differences_per_layer_pair(data, figsize_per_group=(18, 14)):
    """
    Plot RELATIVE DIFFERENCES of cut variables between stubs from consecutive layer pairs.

    For each layer pair, finds all events that have stubs in both layers, then
    computes pairwise relative differences of cut variables between all stub pairs from
    those consecutive layers within the same event.

    Relative difference formula:
        rel_diff = |value_1 - value_2| / (|value_1| + |value_2|)

    This gives:
        - 0 when values are identical
        - 0.5 when one value is 0 (one is zero, other is nonzero)
        - 1 when values have opposite signs and equal magnitude (maximally different)

    Layer pair groups:
    - Group A1: Flat Barrel -> Flat Barrel (L1f->L2f, L2f->L3f, L3f->L4f, L4f->L5f, L5f->L6f)
    - Group A2: Tilted Barrel -> Tilted Barrel (L1t->L2t, L2t->L3t, L3t->L4t)
    - Group A3: Mixed Flat <-> Tilted Barrel transitions
    - Group B1: Flat Barrel -> Forward+ D1 (L1f->D1+, L2f->D1+, L3f->D1+, L4f->D1+, L5f->D1+, L6f->D1+)
    - Group B2: Tilted Barrel -> Forward+ D1 (L1t->D1+, L2t->D1+, L3t->D1+, L4t->D1+)
    - Group C1: Flat Barrel -> Forward- D1 (L1f->D1-, L2f->D1-, L3f->D1-, L4f->D1-, L5f->D1-, L6f->D1-)
    - Group C2: Tilted Barrel -> Forward- D1 (L1t->D1-, L2t->D1-, L3t->D1-, L4t->D1-)
    - Group D: Forward+ consecutive disks (D1+->D2+, D2+->D3+, D3+->D4+, D4+->D5+)
    - Group E: Forward- consecutive disks (D1-->D2-, D2-->D3-, D3-->D4-, D4-->D5-)

    Variables computed as relative differences:
    - rel_diff_dPhiDr: Relative difference in dPhiDr (for barrel-barrel)
    - rel_diff_dPhiDz: Relative difference in dPhiDz (for endcap-endcap)
    - rel_diff_bend: Relative difference in bend

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize_per_group: Figure size tuple for each group figure

    Returns:
        Dictionary of matplotlib figures keyed by group name
    """
    figs = {}

    # Get basic arrays from data
    events = data["event"]
    is_barrel = data["isBarrel"]
    is_endcap = data["isEndcap"]
    stub_z = data["stub_z"]
    layer = data["layer"]
    dPhiDr = data["dPhiDr"]
    dPhiDz = data["dPhiDz"]
    bend = data["bend"]

    def compute_relative_diff(val1, val2):
        """
        Compute relative difference: |v1 - v2| / (|v1| + |v2|)

        Returns NaN when both values are 0 (to be filtered out later).
        Range is [0, 1] where 0 = identical, 1 = maximally different.
        """
        abs_sum = np.abs(val1) + np.abs(val2)
        if abs_sum == 0:
            return np.nan
        return np.abs(val1 - val2) / abs_sum

    def compute_layer_pair_relative_differences(mask1, mask2):
        """
        Compute relative differences of cut variables for all stub pairs from two layer
        selections within the same event.

        Args:
            mask1: Boolean mask for inner layer selection
            mask2: Boolean mask for outer layer selection

        Returns:
            Dictionary with relative difference arrays for each variable
        """
        # Get indices for each layer
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return {
                'rel_diff_dPhiDr': np.array([]),
                'rel_diff_dPhiDz': np.array([]),
                'rel_diff_bend': np.array([])
            }

        # Get values for each layer
        events1 = events[idx1]
        events2 = events[idx2]
        dPhiDr1 = dPhiDr[idx1]
        dPhiDr2 = dPhiDr[idx2]
        dPhiDz1 = dPhiDz[idx1]
        dPhiDz2 = dPhiDz[idx2]
        bend1 = bend[idx1]
        bend2 = bend[idx2]

        # Find unique events that have stubs in both layers
        unique_events = np.intersect1d(np.unique(events1), np.unique(events2))

        rel_diff_dPhiDr_list = []
        rel_diff_dPhiDz_list = []
        rel_diff_bend_list = []

        for evt in unique_events:
            # Get values for this event in each layer
            evt_mask1 = events1 == evt
            evt_mask2 = events2 == evt

            dPhiDr1_evt = dPhiDr1[evt_mask1]
            dPhiDr2_evt = dPhiDr2[evt_mask2]
            dPhiDz1_evt = dPhiDz1[evt_mask1]
            dPhiDz2_evt = dPhiDz2[evt_mask2]
            bend1_evt = bend1[evt_mask1]
            bend2_evt = bend2[evt_mask2]

            # Compute relative differences for all pairs
            for i in range(len(dPhiDr1_evt)):
                for j in range(len(dPhiDr2_evt)):
                    rel_diff_dPhiDr_list.append(
                        compute_relative_diff(dPhiDr1_evt[i], dPhiDr2_evt[j]))
                    rel_diff_dPhiDz_list.append(
                        compute_relative_diff(dPhiDz1_evt[i], dPhiDz2_evt[j]))
                    rel_diff_bend_list.append(
                        compute_relative_diff(bend1_evt[i], bend2_evt[j]))

        return {
            'rel_diff_dPhiDr': np.array(rel_diff_dPhiDr_list),
            'rel_diff_dPhiDz': np.array(rel_diff_dPhiDz_list),
            'rel_diff_bend': np.array(rel_diff_bend_list)
        }

    # Get flat/tilted masks for barrel splitting
    is_flat = data["isFlat"]
    is_tilted = data["isTilted"]

    # Define layer pair groups with masks
    # Group A is split into three subgroups for flat/tilted barrel analysis:
    # - A1: Flat barrel -> Flat barrel
    # - A2: Tilted barrel -> Tilted barrel
    # - A3: Mixed (Flat <-> Tilted) transitions
    layer_pair_groups = {
        "A1_flat_flat": {
            "title": "Group A1: Flat Barrel -> Flat Barrel - Relative Differences",
            "pairs": [
                ("L1f->L2f", is_barrel & is_flat & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3f", is_barrel & is_flat & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4f", is_barrel & is_flat & (layer == 3), is_barrel & is_flat & (layer == 4)),
                ("L4f->L5f", is_barrel & is_flat & (layer == 4), is_barrel & is_flat & (layer == 5)),
                ("L5f->L6f", is_barrel & is_flat & (layer == 5), is_barrel & is_flat & (layer == 6)),
            ]
        },
        "A2_tilted_tilted": {
            "title": "Group A2: Tilted Barrel -> Tilted Barrel - Relative Differences",
            "pairs": [
                ("L1t->L2t", is_barrel & is_tilted & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L2t->L3t", is_barrel & is_tilted & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L3t->L4t", is_barrel & is_tilted & (layer == 3), is_barrel & is_tilted & (layer == 4)),
            ]
        },
        "A3_mixed_flat_tilted": {
            "title": "Group A3: Mixed Flat <-> Tilted Barrel - Relative Differences",
            "pairs": [
                ("L1f->L2t", is_barrel & is_flat & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L1t->L2f", is_barrel & is_tilted & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3t", is_barrel & is_flat & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L2t->L3f", is_barrel & is_tilted & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4t", is_barrel & is_flat & (layer == 3), is_barrel & is_tilted & (layer == 4)),
                ("L3t->L4f", is_barrel & is_tilted & (layer == 3), is_barrel & is_flat & (layer == 4)),
            ]
        },
        "B1_flat_barrel_forward_plus": {
            "title": "Group B1: Flat Barrel to Forward+ (D1+) - Relative Differences",
            "pairs": [
                ("L1f->D1+", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2f->D1+", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3f->D1+", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4f->D1+", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L5f->D1+", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L6f->D1+", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "B2_tilted_barrel_forward_plus": {
            "title": "Group B2: Tilted Barrel to Forward+ (D1+) - Relative Differences",
            "pairs": [
                ("L1t->D1+", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2t->D1+", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3t->D1+", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4t->D1+", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "C1_flat_barrel_forward_minus": {
            "title": "Group C1: Flat Barrel to Forward- (D1-) - Relative Differences",
            "pairs": [
                ("L1f->D1-", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2f->D1-", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3f->D1-", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4f->D1-", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L5f->D1-", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L6f->D1-", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "C2_tilted_barrel_forward_minus": {
            "title": "Group C2: Tilted Barrel to Forward- (D1-) - Relative Differences",
            "pairs": [
                ("L1t->D1-", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2t->D1-", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3t->D1-", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4t->D1-", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "D_forward_plus_consecutive": {
            "title": "Group D: Forward+ Consecutive Disks - Relative Differences",
            "pairs": [
                ("D1+->D2+", is_endcap & (stub_z > 0) & (layer == 1), is_endcap & (stub_z > 0) & (layer == 2)),
                ("D2+->D3+", is_endcap & (stub_z > 0) & (layer == 2), is_endcap & (stub_z > 0) & (layer == 3)),
                ("D3+->D4+", is_endcap & (stub_z > 0) & (layer == 3), is_endcap & (stub_z > 0) & (layer == 4)),
                ("D4+->D5+", is_endcap & (stub_z > 0) & (layer == 4), is_endcap & (stub_z > 0) & (layer == 5)),
            ]
        },
        "E_forward_minus_consecutive": {
            "title": "Group E: Forward- Consecutive Disks - Relative Differences",
            "pairs": [
                ("D1-->D2-", is_endcap & (stub_z < 0) & (layer == 1), is_endcap & (stub_z < 0) & (layer == 2)),
                ("D2-->D3-", is_endcap & (stub_z < 0) & (layer == 2), is_endcap & (stub_z < 0) & (layer == 3)),
                ("D3-->D4-", is_endcap & (stub_z < 0) & (layer == 3), is_endcap & (stub_z < 0) & (layer == 4)),
                ("D4-->D5-", is_endcap & (stub_z < 0) & (layer == 4), is_endcap & (stub_z < 0) & (layer == 5)),
            ]
        },
    }

    # Define variables to plot with their properties
    # Format: (key, label)
    variables = [
        ('rel_diff_dPhiDr', 'Relative diff(dPhiDr)'),
        ('rel_diff_dPhiDz', 'Relative diff(dPhiDz)'),
        ('rel_diff_bend', 'Relative diff(bend)'),
    ]

    # Colors for different layer pair types
    colors = {
        "barrel_barrel": "blue",
        "barrel_fwd_plus": "red",
        "barrel_fwd_minus": "green",
        "flat_barrel_fwd_plus": "cyan",
        "flat_barrel_fwd_minus": "lightgreen",
        "tilted_barrel_fwd_plus": "gold",
        "tilted_barrel_fwd_minus": "goldenrod",
        "fwd_plus": "darkorange",
        "fwd_minus": "purple",
    }

    def get_pair_color(group_name):
        if "flat_flat" in group_name or "tilted_tilted" in group_name or "mixed_flat_tilted" in group_name:
            return colors["barrel_barrel"]
        elif "flat_barrel_forward_plus" in group_name:
            return colors["flat_barrel_fwd_plus"]
        elif "flat_barrel_forward_minus" in group_name:
            return colors["flat_barrel_fwd_minus"]
        elif "tilted_barrel_forward_plus" in group_name:
            return colors["tilted_barrel_fwd_plus"]
        elif "tilted_barrel_forward_minus" in group_name:
            return colors["tilted_barrel_fwd_minus"]
        elif "forward_plus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_plus"]
        elif "forward_minus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_minus"]
        elif "forward_plus" in group_name:
            return colors["fwd_plus"]
        else:
            return colors["fwd_minus"]

    # Reference lines for cut values (now as fractions of [0,1] range)
    reference_lines = [0.1, 0.2, 0.3, 0.5]  # 10%, 20%, 30%, 50%

    # Process each group
    print("Computing relative differences between stubs from consecutive layers...")
    for group_name, group_info in layer_pair_groups.items():
        pairs = group_info["pairs"]
        n_pairs = len(pairs)
        n_vars = len(variables)

        # Create figure: one row per layer pair, one column per variable
        fig, axes = plt.subplots(n_pairs, n_vars, figsize=figsize_per_group)
        fig.suptitle(f"{group_info['title']}\n"
                     f"(Relative differences: |v1-v2|/(|v1|+|v2|), 0=identical, 1=maximally different)",
                     fontsize=12, y=1.02)

        color = get_pair_color(group_name)

        for pair_idx, (pair_name, mask1, mask2) in enumerate(pairs):
            print(f"  Processing {pair_name}...")
            rel_differences = compute_layer_pair_relative_differences(mask1, mask2)
            n_pairs_found = len(rel_differences['rel_diff_dPhiDr'])
            print(f"    Found {n_pairs_found} stub pairs")

            for var_idx, (var_key, var_label) in enumerate(variables):
                # Handle single row case (axes is 1D)
                if n_pairs == 1:
                    ax = axes[var_idx]
                else:
                    ax = axes[pair_idx, var_idx]

                values = rel_differences[var_key]

                # Filter out NaN values (from division by zero when both values are 0)
                if len(values) > 0:
                    values = values[~np.isnan(values)]

                if len(values) > 0:
                    ax.hist(values, bins=100, range=(0, 1),
                            histtype="step", linewidth=1.5, color=color)

                    # Add reference lines
                    for ref_val in reference_lines:
                        ax.axvline(ref_val, color="gray", linestyle="--", alpha=0.5)
                        ax.text(ref_val, ax.get_ylim()[1] * 0.95, f"{int(ref_val*100)}%",
                                fontsize=6, ha="center", va="top", color="gray")

                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    stats_text = f"N={len(values)}\nmean={mean_val:.4f}\nstd={std_val:.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=7, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10)

                ax.set_xlabel(var_label, fontsize=8)
                ax.set_ylabel("Stub pairs", fontsize=8)
                ax.set_title(f"{pair_name}", fontsize=9)
                ax.set_xlim(0, 1)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figs[f"rel_diff_{group_name}"] = fig

    return figs


def plot_geometric_compatibility(data, figsize_per_group=(20, 14)):
    """
    Plot geometric compatibility metrics between stubs from consecutive layer pairs.

    For each layer pair, finds all events that have stubs in both layers, then
    computes pairwise geometric compatibility metrics between all stub pairs from
    those consecutive layers within the same event.

    Metrics computed:
    a) Delta eta (z/r consistency):
       - Compute: delta_eta = (z_2 / r_2) - (z_1 / r_1)
       - For same-track stubs from origin, this should be ~0 (straight line in r-z)

    b) Direction vector dot product:
       - Each stub has a direction from inner to outer hit: (dx, dy) = (outer_x - inner_x, outer_y - inner_y)
       - Normalize and compute dot product: dir1 . dir2
       - Plot |dot product|, range [0, 1], where 1 = aligned

    c) Phi extrapolation consistency:
       - Extrapolate stub1's phi to stub2's radius: phi_pred = phi_1 + dPhiDr_1 * (r_2 - r_1)
       - Compute residual: delta_phi_extrap = phi_2 - phi_pred (with wraparound handling)
       - For same-track stubs, this should be ~0

    d) Bend sign consistency:
       - Compute product: bend_1 * bend_2
       - Positive = same sign, negative = opposite sign
       - Same-track stubs should have consistent bend signs

    Layer pair groups:
    - Group A1: Flat Barrel -> Flat Barrel (L1f->L2f, L2f->L3f, L3f->L4f, L4f->L5f, L5f->L6f)
    - Group A2: Tilted Barrel -> Tilted Barrel (L1t->L2t, L2t->L3t, L3t->L4t)
    - Group A3: Mixed Flat <-> Tilted Barrel transitions
    - Group B1: Flat Barrel -> Forward+ D1 (L1f->D1+, L2f->D1+, L3f->D1+, L4f->D1+, L5f->D1+, L6f->D1+)
    - Group B2: Tilted Barrel -> Forward+ D1 (L1t->D1+, L2t->D1+, L3t->D1+, L4t->D1+)
    - Group C1: Flat Barrel -> Forward- D1 (L1f->D1-, L2f->D1-, L3f->D1-, L4f->D1-, L5f->D1-, L6f->D1-)
    - Group C2: Tilted Barrel -> Forward- D1 (L1t->D1-, L2t->D1-, L3t->D1-, L4t->D1-)
    - Group D: Forward+ consecutive disks (D1+->D2+, D2+->D3+, D3+->D4+, D4+->D5+)
    - Group E: Forward- consecutive disks (D1-->D2-, D2-->D3-, D3-->D4-, D4-->D5-)

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize_per_group: Figure size tuple for each group figure

    Returns:
        Dictionary of matplotlib figures keyed by group name
    """
    figs = {}

    # Get basic arrays from data
    events = data["event"]
    is_barrel = data["isBarrel"]
    is_endcap = data["isEndcap"]
    stub_z = data["stub_z"]
    stub_r = data["stub_r"]
    layer = data["layer"]
    bend = data["bend"]
    dPhiDr = data["dPhiDr"]

    # Compute direction vectors for each stub (inner -> outer)
    dx = data["outer_x"] - data["inner_x"]
    dy = data["outer_y"] - data["inner_y"]

    # Normalize direction vectors
    dir_norm = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
    dir_norm[dir_norm == 0] = 1.0
    dx_normalized = dx / dir_norm
    dy_normalized = dy / dir_norm

    # Compute stub phi from global position (using stub_x, stub_y)
    stub_phi = np.arctan2(data["stub_y"], data["stub_x"])

    # Compute eta = z/r for each stub
    eta = stub_z / stub_r

    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def compute_layer_pair_geometric_metrics(mask1, mask2):
        """
        Compute geometric compatibility metrics for all stub pairs from two layer
        selections within the same event.

        Args:
            mask1: Boolean mask for inner layer selection
            mask2: Boolean mask for outer layer selection

        Returns:
            Dictionary with metric arrays
        """
        # Get indices for each layer
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return {
                'delta_eta': np.array([]),
                'abs_dot_product': np.array([]),
                'delta_phi_extrap': np.array([]),
                'bend_sign_product': np.array([])
            }

        # Get values for each layer
        events1 = events[idx1]
        events2 = events[idx2]
        eta1 = eta[idx1]
        eta2 = eta[idx2]
        dx1 = dx_normalized[idx1]
        dy1 = dy_normalized[idx1]
        dx2 = dx_normalized[idx2]
        dy2 = dy_normalized[idx2]
        phi1 = stub_phi[idx1]
        phi2 = stub_phi[idx2]
        r1 = stub_r[idx1]
        r2 = stub_r[idx2]
        dPhiDr1 = dPhiDr[idx1]
        bend1 = bend[idx1]
        bend2 = bend[idx2]

        # Find unique events that have stubs in both layers
        unique_events = np.intersect1d(np.unique(events1), np.unique(events2))

        delta_eta_list = []
        abs_dot_product_list = []
        delta_phi_extrap_list = []
        bend_sign_product_list = []

        for evt in unique_events:
            # Get values for this event in each layer
            evt_mask1 = events1 == evt
            evt_mask2 = events2 == evt

            eta1_evt = eta1[evt_mask1]
            eta2_evt = eta2[evt_mask2]
            dx1_evt = dx1[evt_mask1]
            dy1_evt = dy1[evt_mask1]
            dx2_evt = dx2[evt_mask2]
            dy2_evt = dy2[evt_mask2]
            phi1_evt = phi1[evt_mask1]
            phi2_evt = phi2[evt_mask2]
            r1_evt = r1[evt_mask1]
            r2_evt = r2[evt_mask2]
            dPhiDr1_evt = dPhiDr1[evt_mask1]
            bend1_evt = bend1[evt_mask1]
            bend2_evt = bend2[evt_mask2]

            # Compute metrics for all pairs
            for i in range(len(eta1_evt)):
                for j in range(len(eta2_evt)):
                    # a) Delta eta
                    delta_eta_list.append(eta2_evt[j] - eta1_evt[i])

                    # b) Absolute dot product of direction vectors
                    dot_prod = dx1_evt[i] * dx2_evt[j] + dy1_evt[i] * dy2_evt[j]
                    abs_dot_product_list.append(np.abs(dot_prod))

                    # c) Phi extrapolation residual
                    # Extrapolate phi1 to r2 using dPhiDr1
                    phi_pred = phi1_evt[i] + dPhiDr1_evt[i] * (r2_evt[j] - r1_evt[i])
                    delta_phi = phi2_evt[j] - phi_pred
                    # Normalize to [-pi, pi]
                    delta_phi = normalize_angle(delta_phi)
                    delta_phi_extrap_list.append(delta_phi)

                    # d) Bend sign product
                    bend_sign_product_list.append(bend1_evt[i] * bend2_evt[j])

        return {
            'delta_eta': np.array(delta_eta_list),
            'abs_dot_product': np.array(abs_dot_product_list),
            'delta_phi_extrap': np.array(delta_phi_extrap_list),
            'bend_sign_product': np.array(bend_sign_product_list)
        }

    # Get flat/tilted masks for barrel splitting
    is_flat = data["isFlat"]
    is_tilted = data["isTilted"]

    # Define layer pair groups with masks
    # Group A is split into three subgroups for flat/tilted barrel analysis:
    # - A1: Flat barrel -> Flat barrel
    # - A2: Tilted barrel -> Tilted barrel
    # - A3: Mixed (Flat <-> Tilted) transitions
    layer_pair_groups = {
        "A1_flat_flat": {
            "title": "Group A1: Flat Barrel -> Flat Barrel - Geometric Compatibility",
            "pairs": [
                ("L1f->L2f", is_barrel & is_flat & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3f", is_barrel & is_flat & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4f", is_barrel & is_flat & (layer == 3), is_barrel & is_flat & (layer == 4)),
                ("L4f->L5f", is_barrel & is_flat & (layer == 4), is_barrel & is_flat & (layer == 5)),
                ("L5f->L6f", is_barrel & is_flat & (layer == 5), is_barrel & is_flat & (layer == 6)),
            ]
        },
        "A2_tilted_tilted": {
            "title": "Group A2: Tilted Barrel -> Tilted Barrel - Geometric Compatibility",
            "pairs": [
                ("L1t->L2t", is_barrel & is_tilted & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L2t->L3t", is_barrel & is_tilted & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L3t->L4t", is_barrel & is_tilted & (layer == 3), is_barrel & is_tilted & (layer == 4)),
            ]
        },
        "A3_mixed_flat_tilted": {
            "title": "Group A3: Mixed Flat <-> Tilted Barrel - Geometric Compatibility",
            "pairs": [
                ("L1f->L2t", is_barrel & is_flat & (layer == 1), is_barrel & is_tilted & (layer == 2)),
                ("L1t->L2f", is_barrel & is_tilted & (layer == 1), is_barrel & is_flat & (layer == 2)),
                ("L2f->L3t", is_barrel & is_flat & (layer == 2), is_barrel & is_tilted & (layer == 3)),
                ("L2t->L3f", is_barrel & is_tilted & (layer == 2), is_barrel & is_flat & (layer == 3)),
                ("L3f->L4t", is_barrel & is_flat & (layer == 3), is_barrel & is_tilted & (layer == 4)),
                ("L3t->L4f", is_barrel & is_tilted & (layer == 3), is_barrel & is_flat & (layer == 4)),
            ]
        },
        "B1_flat_barrel_forward_plus": {
            "title": "Group B1: Flat Barrel to Forward+ (D1+) - Geometric Compatibility",
            "pairs": [
                ("L1f->D1+", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2f->D1+", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3f->D1+", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4f->D1+", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L5f->D1+", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L6f->D1+", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "B2_tilted_barrel_forward_plus": {
            "title": "Group B2: Tilted Barrel to Forward+ (D1+) - Geometric Compatibility",
            "pairs": [
                ("L1t->D1+", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L2t->D1+", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L3t->D1+", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z > 0) & (layer == 1)),
                ("L4t->D1+", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z > 0) & (layer == 1)),
            ]
        },
        "C1_flat_barrel_forward_minus": {
            "title": "Group C1: Flat Barrel to Forward- (D1-) - Geometric Compatibility",
            "pairs": [
                ("L1f->D1-", is_barrel & is_flat & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2f->D1-", is_barrel & is_flat & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3f->D1-", is_barrel & is_flat & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4f->D1-", is_barrel & is_flat & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L5f->D1-", is_barrel & is_flat & (layer == 5), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L6f->D1-", is_barrel & is_flat & (layer == 6), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "C2_tilted_barrel_forward_minus": {
            "title": "Group C2: Tilted Barrel to Forward- (D1-) - Geometric Compatibility",
            "pairs": [
                ("L1t->D1-", is_barrel & is_tilted & (layer == 1), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L2t->D1-", is_barrel & is_tilted & (layer == 2), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L3t->D1-", is_barrel & is_tilted & (layer == 3), is_endcap & (stub_z < 0) & (layer == 1)),
                ("L4t->D1-", is_barrel & is_tilted & (layer == 4), is_endcap & (stub_z < 0) & (layer == 1)),
            ]
        },
        "D_forward_plus_consecutive": {
            "title": "Group D: Forward+ Consecutive Disks - Geometric Compatibility",
            "pairs": [
                ("D1+->D2+", is_endcap & (stub_z > 0) & (layer == 1), is_endcap & (stub_z > 0) & (layer == 2)),
                ("D2+->D3+", is_endcap & (stub_z > 0) & (layer == 2), is_endcap & (stub_z > 0) & (layer == 3)),
                ("D3+->D4+", is_endcap & (stub_z > 0) & (layer == 3), is_endcap & (stub_z > 0) & (layer == 4)),
                ("D4+->D5+", is_endcap & (stub_z > 0) & (layer == 4), is_endcap & (stub_z > 0) & (layer == 5)),
            ]
        },
        "E_forward_minus_consecutive": {
            "title": "Group E: Forward- Consecutive Disks - Geometric Compatibility",
            "pairs": [
                ("D1-->D2-", is_endcap & (stub_z < 0) & (layer == 1), is_endcap & (stub_z < 0) & (layer == 2)),
                ("D2-->D3-", is_endcap & (stub_z < 0) & (layer == 2), is_endcap & (stub_z < 0) & (layer == 3)),
                ("D3-->D4-", is_endcap & (stub_z < 0) & (layer == 3), is_endcap & (stub_z < 0) & (layer == 4)),
                ("D4-->D5-", is_endcap & (stub_z < 0) & (layer == 4), is_endcap & (stub_z < 0) & (layer == 5)),
            ]
        },
    }

    # Define variables to plot with their properties
    # Format: (key, label, range, reference_lines)
    # reference_lines is a list of (value, label) tuples
    variables = [
        ('delta_eta', r'$\Delta\eta$ = $(z/r)_{outer} - (z/r)_{inner}$',
         (-0.5, 0.5), [(0, '0')]),
        ('abs_dot_product', r'$|d_1 \cdot d_2|$ (direction alignment)',
         (0, 1), [(0.9, '0.9'), (0.95, '0.95'), (1.0, '1.0')]),
        ('delta_phi_extrap', r'$\Delta\phi_{extrap}$ = $\phi_2 - (\phi_1 + dPhiDr_1 \cdot \Delta r)$ [rad]',
         (-0.5, 0.5), [(0, '0')]),
        ('bend_sign_product', r'$bend_1 \times bend_2$ [rad$^2$]',
         (-0.001, 0.001), [(0, '0')]),
    ]

    # Colors for different layer pair types
    colors = {
        "barrel_barrel": "blue",
        "barrel_fwd_plus": "red",
        "barrel_fwd_minus": "green",
        "flat_barrel_fwd_plus": "cyan",
        "flat_barrel_fwd_minus": "lightgreen",
        "tilted_barrel_fwd_plus": "gold",
        "tilted_barrel_fwd_minus": "goldenrod",
        "fwd_plus": "darkorange",
        "fwd_minus": "purple",
    }

    def get_pair_color(group_name):
        if "flat_flat" in group_name or "tilted_tilted" in group_name or "mixed_flat_tilted" in group_name:
            return colors["barrel_barrel"]
        elif "flat_barrel_forward_plus" in group_name:
            return colors["flat_barrel_fwd_plus"]
        elif "flat_barrel_forward_minus" in group_name:
            return colors["flat_barrel_fwd_minus"]
        elif "tilted_barrel_forward_plus" in group_name:
            return colors["tilted_barrel_fwd_plus"]
        elif "tilted_barrel_forward_minus" in group_name:
            return colors["tilted_barrel_fwd_minus"]
        elif "forward_plus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_plus"]
        elif "forward_minus" in group_name and "barrel" in group_name:
            return colors["barrel_fwd_minus"]
        elif "forward_plus" in group_name:
            return colors["fwd_plus"]
        else:
            return colors["fwd_minus"]

    # Process each group
    print("Computing geometric compatibility metrics between stubs from consecutive layers...")
    for group_name, group_info in layer_pair_groups.items():
        pairs = group_info["pairs"]
        n_pairs = len(pairs)
        n_vars = len(variables)

        # Create figure: one row per layer pair, one column per variable
        fig, axes = plt.subplots(n_pairs, n_vars, figsize=figsize_per_group)
        fig.suptitle(f"{group_info['title']}\n"
                     f"(Computed for all stub pairs from consecutive layers in same event)",
                     fontsize=12, y=1.02)

        color = get_pair_color(group_name)

        for pair_idx, (pair_name, mask1, mask2) in enumerate(pairs):
            print(f"  Processing {pair_name}...")
            metrics = compute_layer_pair_geometric_metrics(mask1, mask2)
            n_pairs_found = len(metrics['delta_eta'])
            print(f"    Found {n_pairs_found} stub pairs")

            for var_idx, (var_key, var_label, var_range, ref_lines) in enumerate(variables):
                # Handle single row case (axes is 1D)
                if n_pairs == 1:
                    ax = axes[var_idx]
                else:
                    ax = axes[pair_idx, var_idx]

                values = metrics[var_key]

                if len(values) > 0:
                    ax.hist(values, bins=100, range=var_range,
                            histtype="step", linewidth=1.5, color=color)

                    # Add reference lines
                    for ref_val, ref_label in ref_lines:
                        ax.axvline(ref_val, color="gray", linestyle="--", alpha=0.7)
                        ax.text(ref_val, ax.get_ylim()[1] * 0.95, ref_label,
                                fontsize=6, ha="center", va="top", color="gray")

                    # Add statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    stats_text = f"N={len(values)}\nmean={mean_val:.4f}\nstd={std_val:.4f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=7, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10)

                ax.set_xlabel(var_label, fontsize=8)
                ax.set_ylabel("Stub pairs", fontsize=8)
                ax.set_title(f"{pair_name}", fontsize=9)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figs[f"geom_compat_{group_name}"] = fig

    return figs


def plot_summary_stats(data):
    """Print summary statistics for the stub sample."""
    print("=" * 60)
    print("OT Stub Ntuple Summary Statistics")
    print("=" * 60)

    n_total = len(data["stubIndex"])
    n_barrel = np.sum(data["isBarrel"])
    n_endcap = np.sum(data["isEndcap"])
    n_forward = np.sum((data["isEndcap"]) & (data["stub_z"] > 0))
    n_backward = np.sum((data["isEndcap"]) & (data["stub_z"] < 0))
    n_tilted = np.sum(data["isTilted"])
    n_flat = np.sum((data["isBarrel"]) & (data["isFlat"]))
    n_ps = np.sum(data["isPS"])
    n_ss = n_total - n_ps

    print(f"Total stubs:     {n_total:>8}")
    print(f"  Barrel:        {n_barrel:>8} ({100*n_barrel/n_total:.1f}%)")
    print(f"    Flat:        {n_flat:>8}")
    print(f"    Tilted:      {n_tilted:>8}")
    print(f"  Endcap:        {n_endcap:>8} ({100*n_endcap/n_total:.1f}%)")
    print(f"    Forward:     {n_forward:>8}")
    print(f"    Backward:    {n_backward:>8}")
    print(f"  PS modules:    {n_ps:>8}")
    print(f"  SS modules:    {n_ss:>8}")
    print()

    # dPhiDr statistics
    barrel = select_barrel(data)
    endcap = select_endcap(data)

    print("dPhiDr statistics [rad/cm]:")
    print(f"  Barrel:  mean={np.mean(barrel['dPhiDr']):.6f}  std={np.std(barrel['dPhiDr']):.6f}")
    print(f"  Endcap:  mean={np.mean(endcap['dPhiDr']):.6f}  std={np.std(endcap['dPhiDr']):.6f}")
    print()

    # dr statistics
    print("dr_sensor statistics [cm]:")
    print(f"  Barrel:  mean={np.mean(barrel['dr_sensor']):.4f}  std={np.std(barrel['dr_sensor']):.4f}")
    print(f"  Endcap:  mean={np.mean(endcap['dr_sensor']):.6f}  std={np.std(endcap['dr_sensor']):.6f}")
    print()

    # dz statistics
    print("dz_sensor statistics [cm]:")
    print(f"  Barrel:  mean={np.mean(barrel['dz_sensor']):.4f}  std={np.std(barrel['dz_sensor']):.4f}")
    print(f"  Endcap:  mean={np.mean(endcap['dz_sensor']):.4f}  std={np.std(endcap['dz_sensor']):.4f}")
    print()

    # Eta consistency
    print("Eta consistency (z/r outer - inner):")
    print(f"  Barrel:  mean={np.mean(barrel['eta_diff']):.6f}  std={np.std(barrel['eta_diff']):.6f}")
    print(f"  Endcap:  mean={np.mean(endcap['eta_diff']):.6f}  std={np.std(endcap['eta_diff']):.6f}")
    print("=" * 60)


def plot_ca_tuning_variables(data, figsize=(18, 16)):
    """
    Plot per-layer distributions of CA cut tuning variables.

    This function produces histograms of quantities that correspond to the CA
    configuration cut values, organized by layer/disk. These plots help tune:

    Per-layer cuts (from layers table in caHitNtupletAlpakaPhase2OTStubs_cfi.py):
    - caDCACuts: DCA cut per layer (related to circle fit quality)
    - caThetaCuts: theta alignment cut per layer

    Per-layer-pair cuts (from layerPairs table):
    - phiCuts: phi cut per layer pair
    - minDZ, maxDZ: z difference cuts per layer pair
    - maxDR: dr cut per layer pair

    The function produces separate figures for:
    1. Barrel layers 1-6 (flat and tilted combined)
    2. Forward endcap disks 1-5 (positive z)
    3. Backward endcap disks 1-5 (negative z)

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        Dictionary of matplotlib figures keyed by region name
    """
    figs = {}

    # Check which columns are available (for backward compatibility)
    available_cols = set(data.keys())

    # Define the variables to plot and their properties
    # Format: (column_name, xlabel, barrel_range, endcap_range, scale_factor)
    variables = [
        ("dr_sensor", "dr_sensor [cm]", (-3, 3), (-0.5, 0.5), 1.0),
        ("dz_sensor", "dz_sensor [cm]", (-1, 1), (-5, 5), 1.0),
        ("dPhiDr", "dPhiDr [rad/cm]", (-0.01, 0.01), (-0.5, 0.5), 1.0),
        ("dPhiDz", "dPhiDz [rad/cm]", (-0.1, 0.1), (-0.01, 0.01), 1.0),
        ("eta_diff", "eta_diff (z/r outer - inner)", (-0.02, 0.02), (-0.02, 0.02), 1.0),
        ("stub_eta", "stub eta", (-3, 3), (-4, 4), 1.0),
    ]

    # Filter to only available variables
    variables = [(col, label, br, er, sf) for col, label, br, er, sf in variables
                 if col in available_cols]

    if not variables:
        print("Warning: No CA tuning variables found in ntuple")
        return figs

    n_vars = len(variables)

    # --- Barrel layers 1-6 ---
    fig_barrel, axes_barrel = plt.subplots(n_vars, 6, figsize=figsize)
    fig_barrel.suptitle("CA Tuning Variables - OT Barrel Layers 1-6 (CA layers 28-33)", fontsize=14, y=1.02)

    for layer in range(1, 7):
        # Select barrel stubs for this layer (layer field is 1-based in ntuple)
        mask = (data["isBarrel"] == True) & (data["layer"] == layer)
        n_stubs = np.sum(mask)

        for var_idx, (var_name, xlabel, barrel_range, _, scale) in enumerate(variables):
            ax = axes_barrel[var_idx, layer - 1] if n_vars > 1 else axes_barrel[layer - 1]

            if n_stubs > 0:
                values = data[var_name][mask] * scale

                ax.hist(values, bins=50, range=barrel_range,
                        histtype="step", linewidth=1.5, color="blue")

                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    stats_text = (f"N={len(values)}\nmean={mean_val:.4f}\n"
                                  f"std={std_val:.4f}\nmin={min_val:.4f}\nmax={max_val:.4f}")
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10)

            ax.set_xlabel(xlabel, fontsize=7)
            if layer == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"L{layer} (CA {27+layer}, N={n_stubs})", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.grid(True, alpha=0.3)

    fig_barrel.tight_layout()
    figs["barrel_layers"] = fig_barrel

    # --- Forward endcap disks 1-5 (positive z, CA layers 39-43) ---
    fig_forward, axes_forward = plt.subplots(n_vars, 5, figsize=(15, 16))
    fig_forward.suptitle("CA Tuning Variables - OT Forward Endcap Disks 1-5 (CA layers 39-43)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        # Select forward endcap stubs (positive z)
        mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["layer"] == disk)
        n_stubs = np.sum(mask)

        for var_idx, (var_name, xlabel, _, endcap_range, scale) in enumerate(variables):
            ax = axes_forward[var_idx, disk - 1] if n_vars > 1 else axes_forward[disk - 1]

            if n_stubs > 0:
                values = data[var_name][mask] * scale

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="green")

                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    stats_text = (f"N={len(values)}\nmean={mean_val:.4f}\n"
                                  f"std={std_val:.4f}\nmin={min_val:.4f}\nmax={max_val:.4f}")
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10)

            ax.set_xlabel(xlabel, fontsize=7)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk}+ (CA {38+disk}, N={n_stubs})", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.grid(True, alpha=0.3)

    fig_forward.tight_layout()
    figs["forward_disks"] = fig_forward

    # --- Backward endcap disks 1-5 (negative z, CA layers 34-38) ---
    fig_backward, axes_backward = plt.subplots(n_vars, 5, figsize=(15, 16))
    fig_backward.suptitle("CA Tuning Variables - OT Backward Endcap Disks 1-5 (CA layers 34-38)", fontsize=14, y=1.02)

    for disk in range(1, 6):
        # Select backward endcap stubs (negative z)
        mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["layer"] == disk)
        n_stubs = np.sum(mask)

        for var_idx, (var_name, xlabel, _, endcap_range, scale) in enumerate(variables):
            ax = axes_backward[var_idx, disk - 1] if n_vars > 1 else axes_backward[disk - 1]

            if n_stubs > 0:
                values = data[var_name][mask] * scale

                ax.hist(values, bins=50, range=endcap_range,
                        histtype="step", linewidth=1.5, color="red")

                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    stats_text = (f"N={len(values)}\nmean={mean_val:.4f}\n"
                                  f"std={std_val:.4f}\nmin={min_val:.4f}\nmax={max_val:.4f}")
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                            fontsize=6, verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10)

            ax.set_xlabel(xlabel, fontsize=7)
            if disk == 1:
                ax.set_ylabel("Stubs", fontsize=8)
            ax.set_title(f"D{disk}- (CA {33+disk}, N={n_stubs})", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.grid(True, alpha=0.3)

    fig_backward.tight_layout()
    figs["backward_disks"] = fig_backward

    # --- Summary figure with all regions overlaid ---
    fig_summary, axes_summary = plt.subplots(2, 3, figsize=(16, 10))
    fig_summary.suptitle("CA Tuning Variables Summary - All OT Regions", fontsize=14, y=1.02)

    summary_vars = [
        ("dr_sensor", "dr_sensor [cm]", (-3, 3)),
        ("dz_sensor", "dz_sensor [cm]", (-5, 5)),
        ("dPhiDr", "dPhiDr [rad/cm]", (-0.02, 0.02)),
        ("dPhiDz", "dPhiDz [rad/cm]", (-0.05, 0.05)),
        ("eta_diff", "eta_diff", (-0.02, 0.02)),
        ("stub_eta", "stub eta", (-4, 4)),
    ]

    # Filter to available
    summary_vars = [(col, label, rng) for col, label, rng in summary_vars if col in available_cols]

    for idx, (var_name, xlabel, plot_range) in enumerate(summary_vars):
        row = idx // 3
        col = idx % 3
        ax = axes_summary[row, col]

        # Barrel
        barrel_mask = data["isBarrel"] == True
        if np.sum(barrel_mask) > 0:
            ax.hist(data[var_name][barrel_mask], bins=80, range=plot_range,
                    histtype="step", linewidth=1.5, color="blue", alpha=0.8,
                    label=f"Barrel (N={np.sum(barrel_mask)})")

        # Forward endcap
        forward_mask = (data["isEndcap"] == True) & (data["stub_z"] > 0)
        if np.sum(forward_mask) > 0:
            ax.hist(data[var_name][forward_mask], bins=80, range=plot_range,
                    histtype="step", linewidth=1.5, color="green", alpha=0.8,
                    label=f"Fwd (N={np.sum(forward_mask)})")

        # Backward endcap
        backward_mask = (data["isEndcap"] == True) & (data["stub_z"] < 0)
        if np.sum(backward_mask) > 0:
            ax.hist(data[var_name][backward_mask], bins=80, range=plot_range,
                    histtype="step", linewidth=1.5, color="red", alpha=0.8,
                    label=f"Bwd (N={np.sum(backward_mask)})")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Stubs")
        ax.set_title(var_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Handle case where we have fewer than 6 variables
    for idx in range(len(summary_vars), 6):
        row = idx // 3
        col = idx % 3
        axes_summary[row, col].set_visible(False)

    fig_summary.tight_layout()
    figs["summary"] = fig_summary

    # --- Print statistics table for cut tuning ---
    print("\n" + "=" * 80)
    print("CA TUNING VARIABLE STATISTICS PER LAYER")
    print("Use these values to tune cuts in caHitNtupletAlpakaPhase2OTStubs_cfi.py")
    print("=" * 80)

    # Barrel layers
    print("\nOT BARREL LAYERS (CA layers 28-33):")
    print("-" * 80)
    print(f"{'Layer':<12} {'N':>8} {'dr_sensor':>12} {'dz_sensor':>12} {'dPhiDr':>12} {'eta_diff':>12}")
    print(f"{'':12} {'':>8} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12}")
    print("-" * 80)

    for layer in range(1, 7):
        mask = (data["isBarrel"] == True) & (data["layer"] == layer)
        n = np.sum(mask)
        if n > 0:
            dr_mean, dr_std = np.mean(data["dr_sensor"][mask]), np.std(data["dr_sensor"][mask])
            dz_mean, dz_std = np.mean(data["dz_sensor"][mask]), np.std(data["dz_sensor"][mask])
            dpdr_mean, dpdr_std = np.mean(data["dPhiDr"][mask]), np.std(data["dPhiDr"][mask])
            eta_mean, eta_std = np.mean(data["eta_diff"][mask]), np.std(data["eta_diff"][mask])
            print(f"L{layer} (CA {27+layer})  {n:>8} {dr_mean:>6.3f}/{dr_std:<5.3f} "
                  f"{dz_mean:>6.3f}/{dz_std:<5.3f} {dpdr_mean:>6.4f}/{dpdr_std:<5.4f} "
                  f"{eta_mean:>6.4f}/{eta_std:<5.4f}")
        else:
            print(f"L{layer} (CA {27+layer})  {n:>8} {'--':>12} {'--':>12} {'--':>12} {'--':>12}")

    # Forward disks
    print("\nOT FORWARD ENDCAP DISKS (CA layers 39-43, z > 0):")
    print("-" * 80)
    print(f"{'Disk':<12} {'N':>8} {'dr_sensor':>12} {'dz_sensor':>12} {'dPhiDz':>12} {'eta_diff':>12}")
    print(f"{'':12} {'':>8} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12}")
    print("-" * 80)

    for disk in range(1, 6):
        mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["layer"] == disk)
        n = np.sum(mask)
        if n > 0:
            dr_mean, dr_std = np.mean(data["dr_sensor"][mask]), np.std(data["dr_sensor"][mask])
            dz_mean, dz_std = np.mean(data["dz_sensor"][mask]), np.std(data["dz_sensor"][mask])
            dpdz_mean, dpdz_std = np.mean(data["dPhiDz"][mask]), np.std(data["dPhiDz"][mask])
            eta_mean, eta_std = np.mean(data["eta_diff"][mask]), np.std(data["eta_diff"][mask])
            print(f"D{disk}+ (CA {38+disk}) {n:>8} {dr_mean:>6.4f}/{dr_std:<5.4f} "
                  f"{dz_mean:>6.3f}/{dz_std:<5.3f} {dpdz_mean:>6.5f}/{dpdz_std:<5.5f} "
                  f"{eta_mean:>6.4f}/{eta_std:<5.4f}")
        else:
            print(f"D{disk}+ (CA {38+disk}) {n:>8} {'--':>12} {'--':>12} {'--':>12} {'--':>12}")

    # Backward disks
    print("\nOT BACKWARD ENDCAP DISKS (CA layers 34-38, z < 0):")
    print("-" * 80)
    print(f"{'Disk':<12} {'N':>8} {'dr_sensor':>12} {'dz_sensor':>12} {'dPhiDz':>12} {'eta_diff':>12}")
    print(f"{'':12} {'':>8} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12} {'mean/std':>12}")
    print("-" * 80)

    for disk in range(1, 6):
        mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["layer"] == disk)
        n = np.sum(mask)
        if n > 0:
            dr_mean, dr_std = np.mean(data["dr_sensor"][mask]), np.std(data["dr_sensor"][mask])
            dz_mean, dz_std = np.mean(data["dz_sensor"][mask]), np.std(data["dz_sensor"][mask])
            dpdz_mean, dpdz_std = np.mean(data["dPhiDz"][mask]), np.std(data["dPhiDz"][mask])
            eta_mean, eta_std = np.mean(data["eta_diff"][mask]), np.std(data["eta_diff"][mask])
            print(f"D{disk}- (CA {33+disk}) {n:>8} {dr_mean:>6.4f}/{dr_std:<5.4f} "
                  f"{dz_mean:>6.3f}/{dz_std:<5.3f} {dpdz_mean:>6.5f}/{dpdz_std:<5.5f} "
                  f"{eta_mean:>6.4f}/{eta_std:<5.4f}")
        else:
            print(f"D{disk}- (CA {33+disk}) {n:>8} {'--':>12} {'--':>12} {'--':>12} {'--':>12}")

    print("=" * 80 + "\n")

    return figs


def plot_ca_layer_pair_variables(data, figsize=(20, 16)):
    """
    Plot CA cut variable distributions organized by layer PAIR for doublet building.

    This function produces plots that help tune the per-layer-pair cuts in the CA
    configuration (layerPairs table in caHitNtupletAlpakaPhase2OTStubs_cfi.py).

    For each layer pair, the CA algorithm checks compatibility between an inner hit
    and an outer hit. The cuts from the layerPairs table are:
    - phiCuts: maximum dphi (in 1/65536 units of 2*pi) between inner and outer
    - minDZ, maxDZ: allowed range for (outer_z - inner_z)
    - maxDR: maximum |outer_r - inner_r|
    - minIn, maxIn: z limits for inner hit
    - minOut, maxOut: z limits for outer hit
    - ptCuts: minimum pT cut

    Since the ntuple contains single-stub information, we plot stub properties from
    the OUTER layer of each pair, which represent the "incoming" hit that would be
    connected from the inner layer. The key quantities are:
    - stub_z: z position (compare to minIn/maxIn for inner, minOut/maxOut for outer)
    - stub_r: r position (used to compute dR between layers)
    - dphi_raw: stub's internal dphi (not the doublet dphi)
    - dr_sensor, dz_sensor: stub's sensor separation (helps understand stub geometry)

    Layer pair organization:
    - OT barrel consecutive: L1->L2, L2->L3, L3->L4, L4->L5, L5->L6 (CA 28->29, 29->30, etc.)
    - OT barrel to forward disk: L1->D1+, L2->D1+, ..., L5->D1+ (CA 28->39, 29->39, etc.)
    - OT barrel to backward disk: L1->D1-, L2->D1-, ..., L5->D1- (CA 28->34, 29->34, etc.)
    - Forward disk consecutive: D1+->D2+, D2+->D3+, D3+->D4+, D4+->D5+ (CA 39->40, etc.)
    - Backward disk consecutive: D1-->D2-, D2-->D3-, D3-->D4-, D4-->D5- (CA 34->35, etc.)

    Args:
        data: Dictionary of numpy arrays from load_stubs()
        figsize: Figure size tuple

    Returns:
        Dictionary of matplotlib figures keyed by layer pair group name
    """
    figs = {}

    # Check available columns
    available_cols = set(data.keys())

    # CA layer index mapping for OT stubs:
    # Barrel L1-L6 -> CA 28-33
    # Forward D1-D5 -> CA 39-43
    # Backward D1-D5 -> CA 34-38

    # Define layer pairs with their CA cut values from caHitNtupletAlpakaPhase2OTStubs_cfi.py
    # Format: (inner_layer, outer_layer, inner_selector, outer_selector, ca_inner, ca_outer, cuts_dict)
    # cuts_dict contains: phiCut, minIn, maxIn, minOut, maxOut, maxDR, minDZ, maxDZ, ptCut

    # OT barrel consecutive layer pairs
    barrel_consecutive_pairs = [
        # L1->L2 (CA 28->29): phiCut=1100, minIn=-1200, maxIn=1200, minOut=-10000, maxOut=10000, maxDR=10000, minDZ=-50, maxDZ=50
        ("L1", "L2", 1, 2, 28, 29, {"phiCut": 1100, "minIn": -1200, "maxIn": 1200, "minDZ": -50, "maxDZ": 50}),
        # L2->L3 (CA 29->30): phiCut=1250, minIn=-1200, maxIn=1200, minDZ=-40, maxDZ=40
        ("L2", "L3", 2, 3, 29, 30, {"phiCut": 1250, "minIn": -1200, "maxIn": 1200, "minDZ": -40, "maxDZ": 40}),
        # L3->L4 (CA 30->31): phiCut=1250, minDZ=-40, maxDZ=40
        ("L3", "L4", 3, 4, 30, 31, {"phiCut": 1250, "minDZ": -40, "maxDZ": 40}),
        # L4->L5 (CA 31->32): phiCut=1250, minDZ=-40, maxDZ=40
        ("L4", "L5", 4, 5, 31, 32, {"phiCut": 1250, "minDZ": -40, "maxDZ": 40}),
        # L5->L6 (CA 32->33): phiCut=1250, minDZ=-40, maxDZ=40
        ("L5", "L6", 5, 6, 32, 33, {"phiCut": 1250, "minDZ": -40, "maxDZ": 40}),
    ]

    # Forward disk consecutive layer pairs (z > 0)
    forward_disk_pairs = [
        # D1+->D2+ (CA 39->40): phiCut=1500, minDZ=-40, maxDZ=40
        ("D1+", "D2+", 1, 2, 39, 40, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D2+->D3+ (CA 40->41)
        ("D2+", "D3+", 2, 3, 40, 41, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D3+->D4+ (CA 41->42)
        ("D3+", "D4+", 3, 4, 41, 42, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D4+->D5+ (CA 42->43)
        ("D4+", "D5+", 4, 5, 42, 43, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
    ]

    # Backward disk consecutive layer pairs (z < 0)
    backward_disk_pairs = [
        # D1-->D2- (CA 34->35): phiCut=1500, minDZ=-40, maxDZ=40
        ("D1-", "D2-", 1, 2, 34, 35, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D2-->D3- (CA 35->36)
        ("D2-", "D3-", 2, 3, 35, 36, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D3-->D4- (CA 36->37)
        ("D3-", "D4-", 3, 4, 36, 37, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
        # D4-->D5- (CA 37->38)
        ("D4-", "D5-", 4, 5, 37, 38, {"phiCut": 1500, "minDZ": -40, "maxDZ": 40}),
    ]

    # Barrel to forward disk pairs
    barrel_to_forward_pairs = [
        # L1->D1+ (CA 28->39): phiCut=1500, minIn=80, maxIn=10000, minOut=0, maxOut=80
        ("L1", "D1+", 1, 1, 28, 39, {"phiCut": 1500, "minIn": 80}),
        # L2->D1+ (CA 29->39)
        ("L2", "D1+", 2, 1, 29, 39, {"phiCut": 1500, "minIn": 80}),
        # L3->D1+ (CA 30->39)
        ("L3", "D1+", 3, 1, 30, 39, {"phiCut": 1500, "minIn": 80}),
        # L4->D1+ (CA 31->39)
        ("L4", "D1+", 4, 1, 31, 39, {"phiCut": 1500, "minIn": 80}),
        # L5->D1+ (CA 32->39)
        ("L5", "D1+", 5, 1, 32, 39, {"phiCut": 1500, "minIn": 80}),
    ]

    # Barrel to backward disk pairs
    barrel_to_backward_pairs = [
        # L1->D1- (CA 28->34): phiCut=1500, minIn=-1100, maxIn=-80
        ("L1", "D1-", 1, 1, 28, 34, {"phiCut": 1500, "maxIn": -80}),
        # L2->D1- (CA 29->34)
        ("L2", "D1-", 2, 1, 29, 34, {"phiCut": 1500, "maxIn": -80}),
        # L3->D1- (CA 30->34)
        ("L3", "D1-", 3, 1, 30, 34, {"phiCut": 1500, "maxIn": -80}),
        # L4->D1- (CA 31->34)
        ("L4", "D1-", 4, 1, 31, 34, {"phiCut": 1500, "maxIn": -80}),
        # L5->D1- (CA 32->34)
        ("L5", "D1-", 5, 1, 32, 34, {"phiCut": 1500, "maxIn": -80}),
    ]

    def select_barrel_layer(layer_idx):
        """Select barrel stubs for a given layer index (1-6)."""
        mask = (data["isBarrel"] == True) & (data["layer"] == layer_idx)
        return mask

    def select_forward_disk(disk_idx):
        """Select forward endcap (z>0) stubs for a given disk index (1-5)."""
        mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["layer"] == disk_idx)
        return mask

    def select_backward_disk(disk_idx):
        """Select backward endcap (z<0) stubs for a given disk index (1-5)."""
        mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["layer"] == disk_idx)
        return mask

    def get_selector(layer_name, layer_idx):
        """Get the appropriate selector function based on layer name."""
        if layer_name.startswith("L"):
            return select_barrel_layer(layer_idx)
        elif layer_name.endswith("+"):
            return select_forward_disk(layer_idx)
        else:  # ends with "-"
            return select_backward_disk(layer_idx)

    def plot_layer_pair_group(pairs, group_name, title, is_barrel_outer=True):
        """
        Create a figure for a group of layer pairs.

        For each pair, we show the z position and r position of the OUTER layer stubs,
        since these are what the CA algorithm would be checking against.
        """
        n_pairs = len(pairs)
        n_rows = 4  # stub_z, stub_r, dphi_raw, dz_sensor
        n_cols = n_pairs

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(f"{title}\nOuter layer stub distributions for CA doublet building", fontsize=12, y=1.02)

        for col_idx, (inner_name, outer_name, inner_idx, outer_idx, ca_inner, ca_outer, cuts) in enumerate(pairs):
            # Get outer layer data
            outer_mask = get_selector(outer_name, outer_idx)
            n_outer = np.sum(outer_mask)

            pair_label = f"{inner_name}->{outer_name}\n(CA {ca_inner}->{ca_outer})"

            # Row 0: stub_z distribution of outer layer
            ax = axes[0, col_idx] if n_cols > 1 else axes[0]
            if "stub_z" in available_cols and n_outer > 0:
                z_vals = data["stub_z"][outer_mask]
                z_range = (-300, 300) if is_barrel_outer else (-350, 350)
                ax.hist(z_vals, bins=60, range=z_range, histtype="step", linewidth=1.5, color="blue")

                # Add cut lines if available
                if "minIn" in cuts:
                    ax.axvline(cuts["minIn"], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"minIn={cuts['minIn']}")
                if "maxIn" in cuts:
                    ax.axvline(cuts["maxIn"], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"maxIn={cuts['maxIn']}")

                mean_z = np.mean(z_vals)
                std_z = np.std(z_vals)
                stats = f"N={n_outer}\nmean={mean_z:.1f}\nstd={std_z:.1f}"
                ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                if "minIn" in cuts or "maxIn" in cuts:
                    ax.legend(fontsize=6, loc="upper left")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

            ax.set_xlabel("stub_z [cm]", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Stubs", fontsize=9)
            ax.set_title(pair_label, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            # Row 1: stub_r distribution of outer layer
            ax = axes[1, col_idx] if n_cols > 1 else axes[1]
            if "stub_r" in available_cols and n_outer > 0:
                r_vals = data["stub_r"][outer_mask]
                r_range = (20, 120) if is_barrel_outer else (20, 120)
                ax.hist(r_vals, bins=50, range=r_range, histtype="step", linewidth=1.5, color="green")

                mean_r = np.mean(r_vals)
                std_r = np.std(r_vals)
                stats = f"N={n_outer}\nmean={mean_r:.1f}\nstd={std_r:.1f}"
                ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

            ax.set_xlabel("stub_r [cm]", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Stubs", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            # Row 2: dphi_raw distribution of outer layer
            ax = axes[2, col_idx] if n_cols > 1 else axes[2]
            if "dphi_raw" in available_cols and n_outer > 0:
                dphi_vals = data["dphi_raw"][outer_mask]
                dphi_range = (-0.05, 0.05)
                ax.hist(dphi_vals, bins=50, range=dphi_range, histtype="step", linewidth=1.5, color="purple")

                mean_dphi = np.mean(dphi_vals)
                std_dphi = np.std(dphi_vals)
                stats = f"N={n_outer}\nmean={mean_dphi:.4f}\nstd={std_dphi:.4f}"
                ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

            ax.set_xlabel("dphi_raw [rad]", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Stubs", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            # Row 3: dz_sensor distribution (for understanding doublet dZ)
            ax = axes[3, col_idx] if n_cols > 1 else axes[3]
            if "dz_sensor" in available_cols and n_outer > 0:
                dz_vals = data["dz_sensor"][outer_mask]
                dz_range = (-5, 5) if is_barrel_outer else (-5, 5)
                ax.hist(dz_vals, bins=50, range=dz_range, histtype="step", linewidth=1.5, color="orange")

                # Add minDZ/maxDZ cut lines if applicable
                if "minDZ" in cuts and cuts["minDZ"] > -100:
                    ax.axvline(cuts["minDZ"], color="red", linestyle="--", linewidth=1.5, alpha=0.8)
                if "maxDZ" in cuts and cuts["maxDZ"] < 100:
                    ax.axvline(cuts["maxDZ"], color="red", linestyle="--", linewidth=1.5, alpha=0.8)

                mean_dz = np.mean(dz_vals)
                std_dz = np.std(dz_vals)
                stats = f"N={n_outer}\nmean={mean_dz:.3f}\nstd={std_dz:.3f}"
                ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

            ax.set_xlabel("dz_sensor [cm]", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Stubs", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # Create figures for each layer pair group
    print("\nCreating CA layer pair variable plots...")

    # 1. Barrel consecutive pairs
    if len(barrel_consecutive_pairs) > 0:
        fig = plot_layer_pair_group(
            barrel_consecutive_pairs,
            "barrel_consecutive",
            "OT Barrel Consecutive Layer Pairs (L1->L2, ..., L5->L6)",
            is_barrel_outer=True
        )
        figs["barrel_consecutive"] = fig

    # 2. Forward disk consecutive pairs
    if len(forward_disk_pairs) > 0:
        fig = plot_layer_pair_group(
            forward_disk_pairs,
            "forward_consecutive",
            "OT Forward Endcap Consecutive Disk Pairs (D1+->D2+, ..., D4+->D5+)",
            is_barrel_outer=False
        )
        figs["forward_consecutive"] = fig

    # 3. Backward disk consecutive pairs
    if len(backward_disk_pairs) > 0:
        fig = plot_layer_pair_group(
            backward_disk_pairs,
            "backward_consecutive",
            "OT Backward Endcap Consecutive Disk Pairs (D1-->D2-, ..., D4-->D5-)",
            is_barrel_outer=False
        )
        figs["backward_consecutive"] = fig

    # 4. Barrel to forward disk pairs
    if len(barrel_to_forward_pairs) > 0:
        fig = plot_layer_pair_group(
            barrel_to_forward_pairs,
            "barrel_to_forward",
            "OT Barrel to Forward Disk Pairs (L1->D1+, ..., L5->D1+)",
            is_barrel_outer=False
        )
        figs["barrel_to_forward"] = fig

    # 5. Barrel to backward disk pairs
    if len(barrel_to_backward_pairs) > 0:
        fig = plot_layer_pair_group(
            barrel_to_backward_pairs,
            "barrel_to_backward",
            "OT Barrel to Backward Disk Pairs (L1->D1-, ..., L5->D1-)",
            is_barrel_outer=False
        )
        figs["barrel_to_backward"] = fig

    # --- Create a summary figure showing inter-layer dZ and dR distributions ---
    # This requires computing differences between consecutive events in the same layer pair
    # Since we don't have event-level pairing, we'll estimate from layer-to-layer z/r differences

    fig_summary, axes_summary = plt.subplots(2, 3, figsize=(18, 10))
    fig_summary.suptitle("CA Layer Pair Summary: Typical Z and R Separations Between Layers", fontsize=12, y=1.02)

    # Collect layer z and r statistics
    layer_stats = {}

    # Barrel layers
    for layer in range(1, 7):
        mask = (data["isBarrel"] == True) & (data["layer"] == layer)
        if np.sum(mask) > 0:
            layer_stats[f"L{layer}"] = {
                "z_mean": np.mean(data["stub_z"][mask]),
                "z_std": np.std(data["stub_z"][mask]),
                "r_mean": np.mean(data["stub_r"][mask]),
                "r_std": np.std(data["stub_r"][mask]),
            }

    # Forward disks
    for disk in range(1, 6):
        mask = (data["isEndcap"] == True) & (data["stub_z"] > 0) & (data["layer"] == disk)
        if np.sum(mask) > 0:
            layer_stats[f"D{disk}+"] = {
                "z_mean": np.mean(data["stub_z"][mask]),
                "z_std": np.std(data["stub_z"][mask]),
                "r_mean": np.mean(data["stub_r"][mask]),
                "r_std": np.std(data["stub_r"][mask]),
            }

    # Backward disks
    for disk in range(1, 6):
        mask = (data["isEndcap"] == True) & (data["stub_z"] < 0) & (data["layer"] == disk)
        if np.sum(mask) > 0:
            layer_stats[f"D{disk}-"] = {
                "z_mean": np.mean(data["stub_z"][mask]),
                "z_std": np.std(data["stub_z"][mask]),
                "r_mean": np.mean(data["stub_r"][mask]),
                "r_std": np.std(data["stub_r"][mask]),
            }

    # Plot 0,0: Barrel layer z positions
    ax = axes_summary[0, 0]
    barrel_layers = [f"L{i}" for i in range(1, 7) if f"L{i}" in layer_stats]
    if barrel_layers:
        z_means = [layer_stats[l]["z_mean"] for l in barrel_layers]
        z_stds = [layer_stats[l]["z_std"] for l in barrel_layers]
        x = range(len(barrel_layers))
        ax.errorbar(x, z_means, yerr=z_stds, fmt="o-", capsize=3, color="blue")
        ax.set_xticks(x)
        ax.set_xticklabels(barrel_layers)
        ax.set_ylabel("z position [cm]")
        ax.set_title("Barrel Layer Z Positions")
        ax.grid(True, alpha=0.3)

    # Plot 0,1: Barrel layer r positions
    ax = axes_summary[0, 1]
    if barrel_layers:
        r_means = [layer_stats[l]["r_mean"] for l in barrel_layers]
        r_stds = [layer_stats[l]["r_std"] for l in barrel_layers]
        ax.errorbar(x, r_means, yerr=r_stds, fmt="o-", capsize=3, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(barrel_layers)
        ax.set_ylabel("r position [cm]")
        ax.set_title("Barrel Layer R Positions")
        ax.grid(True, alpha=0.3)

        # Show approximate dR between consecutive layers
        if len(r_means) > 1:
            dr_text = "Approx dR:\n"
            for i in range(len(r_means) - 1):
                dr = r_means[i + 1] - r_means[i]
                dr_text += f"L{i+1}->L{i+2}: {dr:.1f} cm\n"
            ax.text(0.02, 0.98, dr_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Plot 0,2: Forward disk z positions
    ax = axes_summary[0, 2]
    fwd_disks = [f"D{i}+" for i in range(1, 6) if f"D{i}+" in layer_stats]
    if fwd_disks:
        z_means = [layer_stats[d]["z_mean"] for d in fwd_disks]
        z_stds = [layer_stats[d]["z_std"] for d in fwd_disks]
        x = range(len(fwd_disks))
        ax.errorbar(x, z_means, yerr=z_stds, fmt="o-", capsize=3, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(fwd_disks)
        ax.set_ylabel("z position [cm]")
        ax.set_title("Forward Disk Z Positions")
        ax.grid(True, alpha=0.3)

        # Show approximate dZ between consecutive disks
        if len(z_means) > 1:
            dz_text = "Approx dZ:\n"
            for i in range(len(z_means) - 1):
                dz = z_means[i + 1] - z_means[i]
                dz_text += f"D{i+1}+->D{i+2}+: {dz:.1f} cm\n"
            ax.text(0.02, 0.98, dz_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Plot 1,0: Backward disk z positions
    ax = axes_summary[1, 0]
    bwd_disks = [f"D{i}-" for i in range(1, 6) if f"D{i}-" in layer_stats]
    if bwd_disks:
        z_means = [layer_stats[d]["z_mean"] for d in bwd_disks]
        z_stds = [layer_stats[d]["z_std"] for d in bwd_disks]
        x = range(len(bwd_disks))
        ax.errorbar(x, z_means, yerr=z_stds, fmt="o-", capsize=3, color="red")
        ax.set_xticks(x)
        ax.set_xticklabels(bwd_disks)
        ax.set_ylabel("z position [cm]")
        ax.set_title("Backward Disk Z Positions")
        ax.grid(True, alpha=0.3)

        # Show approximate dZ between consecutive disks
        if len(z_means) > 1:
            dz_text = "Approx dZ:\n"
            for i in range(len(z_means) - 1):
                dz = z_means[i + 1] - z_means[i]
                dz_text += f"D{i+1}-->D{i+2}-: {dz:.1f} cm\n"
            ax.text(0.02, 0.02, dz_text, transform=ax.transAxes, fontsize=7,
                    verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Plot 1,1: Forward disk r positions
    ax = axes_summary[1, 1]
    if fwd_disks:
        r_means = [layer_stats[d]["r_mean"] for d in fwd_disks]
        r_stds = [layer_stats[d]["r_std"] for d in fwd_disks]
        x = range(len(fwd_disks))
        ax.errorbar(x, r_means, yerr=r_stds, fmt="o-", capsize=3, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(fwd_disks)
        ax.set_ylabel("r position [cm]")
        ax.set_title("Forward Disk R Positions")
        ax.grid(True, alpha=0.3)

    # Plot 1,2: Backward disk r positions
    ax = axes_summary[1, 2]
    if bwd_disks:
        r_means = [layer_stats[d]["r_mean"] for d in bwd_disks]
        r_stds = [layer_stats[d]["r_std"] for d in bwd_disks]
        x = range(len(bwd_disks))
        ax.errorbar(x, r_means, yerr=r_stds, fmt="o-", capsize=3, color="red")
        ax.set_xticks(x)
        ax.set_xticklabels(bwd_disks)
        ax.set_ylabel("r position [cm]")
        ax.set_title("Backward Disk R Positions")
        ax.grid(True, alpha=0.3)

    fig_summary.tight_layout()
    figs["layer_pair_summary"] = fig_summary

    # Print summary table
    print("\n" + "=" * 100)
    print("CA LAYER PAIR SUMMARY: TYPICAL Z AND R RANGES FOR DOUBLET BUILDING")
    print("Use these to verify/tune minIn, maxIn, minOut, maxOut, maxDR, minDZ, maxDZ cuts")
    print("=" * 100)

    print("\nBARREL LAYERS:")
    print("-" * 80)
    print(f"{'Layer':<10} {'CA':<6} {'Z mean':>10} {'Z std':>10} {'R mean':>10} {'R std':>10}")
    print("-" * 80)
    for layer in range(1, 7):
        key = f"L{layer}"
        if key in layer_stats:
            s = layer_stats[key]
            print(f"{key:<10} {27+layer:<6} {s['z_mean']:>10.1f} {s['z_std']:>10.1f} {s['r_mean']:>10.1f} {s['r_std']:>10.1f}")

    print("\nFORWARD ENDCAP DISKS (z > 0):")
    print("-" * 80)
    print(f"{'Disk':<10} {'CA':<6} {'Z mean':>10} {'Z std':>10} {'R mean':>10} {'R std':>10}")
    print("-" * 80)
    for disk in range(1, 6):
        key = f"D{disk}+"
        if key in layer_stats:
            s = layer_stats[key]
            print(f"{key:<10} {38+disk:<6} {s['z_mean']:>10.1f} {s['z_std']:>10.1f} {s['r_mean']:>10.1f} {s['r_std']:>10.1f}")

    print("\nBACKWARD ENDCAP DISKS (z < 0):")
    print("-" * 80)
    print(f"{'Disk':<10} {'CA':<6} {'Z mean':>10} {'Z std':>10} {'R mean':>10} {'R std':>10}")
    print("-" * 80)
    for disk in range(1, 6):
        key = f"D{disk}-"
        if key in layer_stats:
            s = layer_stats[key]
            print(f"{key:<10} {33+disk:<6} {s['z_mean']:>10.1f} {s['z_std']:>10.1f} {s['r_mean']:>10.1f} {s['r_std']:>10.1f}")

    print("\nCONSECUTIVE LAYER dZ AND dR ESTIMATES:")
    print("-" * 80)

    # Barrel consecutive
    print("Barrel consecutive pairs:")
    for i in range(5):
        l_in = f"L{i+1}"
        l_out = f"L{i+2}"
        if l_in in layer_stats and l_out in layer_stats:
            dz = layer_stats[l_out]["z_mean"] - layer_stats[l_in]["z_mean"]
            dr = layer_stats[l_out]["r_mean"] - layer_stats[l_in]["r_mean"]
            print(f"  {l_in}->{l_out} (CA {28+i}->{29+i}): dZ ~ {dz:.1f} cm, dR ~ {dr:.1f} cm")

    # Forward consecutive
    print("Forward disk consecutive pairs:")
    for i in range(4):
        d_in = f"D{i+1}+"
        d_out = f"D{i+2}+"
        if d_in in layer_stats and d_out in layer_stats:
            dz = layer_stats[d_out]["z_mean"] - layer_stats[d_in]["z_mean"]
            dr = layer_stats[d_out]["r_mean"] - layer_stats[d_in]["r_mean"]
            print(f"  {d_in}->{d_out} (CA {39+i}->{40+i}): dZ ~ {dz:.1f} cm, dR ~ {dr:.1f} cm")

    # Backward consecutive
    print("Backward disk consecutive pairs:")
    for i in range(4):
        d_in = f"D{i+1}-"
        d_out = f"D{i+2}-"
        if d_in in layer_stats and d_out in layer_stats:
            dz = layer_stats[d_out]["z_mean"] - layer_stats[d_in]["z_mean"]
            dr = layer_stats[d_out]["r_mean"] - layer_stats[d_in]["r_mean"]
            print(f"  {d_in}->{d_out} (CA {34+i}->{35+i}): dZ ~ {dz:.1f} cm, dR ~ {dr:.1f} cm")

    print("=" * 100 + "\n")

    return figs


def plot_kappa_comparison(data, figsize=(22, 20)):
    """
    Compare naive dPhiDr significance vs curvature-corrected kappa significance
    for barrel stub pairs on consecutive layers.

    For a helix in a uniform B field:
        |dPhiDr(r)| = kappa / sqrt(1 - (r*kappa)^2)

    where kappa = 1/(2*rho) = 0.3*B/(2*pT) is the radius-independent half-curvature.
    Raw dPhiDr increases systematically from inner to outer layers (for fixed pT),
    introducing a bias in naive |dPhiDr_i - dPhiDr_j| comparisons.

    Extracting kappa = |dPhiDr| / sqrt(1 + r^2 * dPhiDr^2) removes this bias exactly,
    centering the same-track distribution at zero for all layer pairs and pT values.

    This function shows:
      Row 0: Per-layer kappa distributions (barrel layers 1-6)
      Row 1: Naive vs kappa pairwise significance for flat barrel (per layer pair)
      Row 2: Naive vs kappa pairwise significance for tilted barrel (per layer pair)
      Row 3: Aggregate comparison (all flat pairs, all tilted pairs)
      Row 4: kappa vs stub_r scatter (should be horizontal bands for each pT)
    """
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle("Curvature-Corrected Stub Linking: kappa vs naive dPhiDr",
                 fontsize=14, fontweight="bold")

    # Precompute kappa and kappa_error for all stubs
    r = data["stub_r"]
    dPhiDr = data["dPhiDr"]
    dPhiDrErr = data["dPhiDrError"]
    kappa_all = compute_kappa(dPhiDr, r)
    kappa_err_all = compute_kappa_error(dPhiDrErr, dPhiDr, r)

    is_barrel = data["isBarrel"] == True
    is_flat = data["isFlat"] == True
    is_tilted = is_barrel & ~is_flat
    layer = data["layer"]

    # ========================================================================
    # Row 0: Per-layer kappa distributions
    # ========================================================================
    for col, l in enumerate(range(1, 7)):
        if col >= 5:
            break  # Only 5 columns available
        ax = axes[0, col]
        mask = is_barrel & (layer == l)
        if np.sum(mask) > 0:
            k = kappa_all[mask]
            d = np.abs(dPhiDr[mask])
            ax.hist(d, bins=100, range=(0, 0.006), histtype="step", linewidth=1.5,
                    color="blue", alpha=0.8, label="|dPhiDr|")
            ax.hist(k, bins=100, range=(0, 0.006), histtype="step", linewidth=1.5,
                    color="red", alpha=0.8, label=r"$\kappa$")
            ax.set_title(f"Barrel L{l}", fontsize=10)
            ax.legend(fontsize=7)
            stats = (f"N={np.sum(mask)}\n"
                     f"|dPhiDr| mean={np.mean(d):.5f}\n"
                     r"$\kappa$" + f" mean={np.mean(k):.5f}")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=6,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("rad/cm", fontsize=8)
        ax.set_ylabel("Stubs", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0, 4].set_visible(False)  # Only 5 columns; L6 not shown

    # ========================================================================
    # Helper: compute pairwise significance for both naive and kappa
    # ========================================================================
    def pairwise_significance(data, mask1, mask2, use_kappa=False):
        """Compute pairwise significance for same-event stub pairs."""
        events = data["event"]
        if use_kappa:
            val_arr = kappa_all
            err_arr = kappa_err_all
        else:
            val_arr = data["dPhiDr"]
            err_arr = data["dPhiDrError"]

        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            return np.array([])

        ev1, ev2 = events[idx1], events[idx2]
        v1, v2 = val_arr[idx1], val_arr[idx2]
        e1, e2 = err_arr[idx1], err_arr[idx2]

        unique_events = np.intersect1d(np.unique(ev1), np.unique(ev2))

        sig_list = []
        for evt in unique_events:
            m1 = ev1 == evt
            m2 = ev2 == evt
            d1, d2 = v1[m1], v2[m2]
            s1, s2 = e1[m1], e2[m2]
            for i in range(len(d1)):
                for j in range(len(d2)):
                    combined_err = np.sqrt(s1[i]**2 + s2[j]**2)
                    if combined_err > 0:
                        sig_list.append(np.abs(d1[i] - d2[j]) / combined_err)
        return np.array(sig_list)

    # ========================================================================
    # Row 1: Flat barrel - per consecutive layer pair (naive vs kappa)
    # ========================================================================
    flat_naive_all = []
    flat_kappa_all = []
    for col, l_inner in enumerate(range(1, 6)):
        ax = axes[1, col]
        m1 = is_barrel & is_flat & (layer == l_inner)
        m2 = is_barrel & is_flat & (layer == l_inner + 1)

        sig_naive = pairwise_significance(data, m1, m2, use_kappa=False)
        sig_kappa = pairwise_significance(data, m1, m2, use_kappa=True)

        if len(sig_naive) > 0:
            flat_naive_all.append(sig_naive)
        if len(sig_kappa) > 0:
            flat_kappa_all.append(sig_kappa)

        xrange = (0, 10)
        if len(sig_naive) > 0:
            ax.hist(sig_naive, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="blue", alpha=0.8, label="Naive dPhiDr")
        if len(sig_kappa) > 0:
            ax.hist(sig_kappa, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="red", alpha=0.8, label=r"$\kappa$-corrected")
        ax.axvline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"Flat L{l_inner}-L{l_inner+1}", fontsize=9)
        ax.set_xlabel("Significance", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Stats
        if len(sig_naive) > 0 and len(sig_kappa) > 0:
            stats = (f"Naive: {np.mean(sig_naive):.2f}+/-{np.std(sig_naive):.2f}\n"
                     rf"$\kappa$: {np.mean(sig_kappa):.2f}+/-{np.std(sig_kappa):.2f}")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=6,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ========================================================================
    # Row 2: Tilted barrel - per consecutive layer pair (naive vs kappa)
    # ========================================================================
    tilted_naive_all = []
    tilted_kappa_all = []
    for col, l_inner in enumerate(range(1, 4)):
        ax = axes[2, col]
        m1 = is_barrel & is_tilted & (layer == l_inner)
        m2 = is_barrel & is_tilted & (layer == l_inner + 1)

        sig_naive = pairwise_significance(data, m1, m2, use_kappa=False)
        sig_kappa = pairwise_significance(data, m1, m2, use_kappa=True)

        if len(sig_naive) > 0:
            tilted_naive_all.append(sig_naive)
        if len(sig_kappa) > 0:
            tilted_kappa_all.append(sig_kappa)

        xrange = (0, 10)
        if len(sig_naive) > 0:
            ax.hist(sig_naive, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="blue", alpha=0.8, label="Naive dPhiDr")
        if len(sig_kappa) > 0:
            ax.hist(sig_kappa, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="red", alpha=0.8, label=r"$\kappa$-corrected")
        ax.axvline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"Tilted L{l_inner}-L{l_inner+1}", fontsize=9)
        ax.set_xlabel("Significance", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        if len(sig_naive) > 0 and len(sig_kappa) > 0:
            stats = (f"Naive: {np.mean(sig_naive):.2f}+/-{np.std(sig_naive):.2f}\n"
                     rf"$\kappa$: {np.mean(sig_kappa):.2f}+/-{np.std(sig_kappa):.2f}")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=6,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Hide unused panels in row 2
    for col in range(3, 5):
        axes[2, col].set_visible(False)

    # ========================================================================
    # Row 3: Aggregate comparison and explanation
    # ========================================================================
    # [3,0]: All flat pairs aggregated
    ax = axes[3, 0]
    flat_naive_agg = np.concatenate(flat_naive_all) if flat_naive_all else np.array([])
    flat_kappa_agg = np.concatenate(flat_kappa_all) if flat_kappa_all else np.array([])
    if len(flat_naive_agg) > 0:
        ax.hist(flat_naive_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="blue", alpha=0.8, label="Naive dPhiDr")
    if len(flat_kappa_agg) > 0:
        ax.hist(flat_kappa_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="red", alpha=0.8, label=r"$\kappa$-corrected")
    ax.axvline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("All Flat Barrel Pairs", fontsize=10)
    ax.set_xlabel("Significance", fontsize=9)
    ax.set_ylabel("Pairs", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if len(flat_naive_agg) > 0 and len(flat_kappa_agg) > 0:
        pct_naive = np.sum(flat_naive_agg < 5.0) / len(flat_naive_agg) * 100
        pct_kappa = np.sum(flat_kappa_agg < 5.0) / len(flat_kappa_agg) * 100
        stats = (f"Naive: mean={np.mean(flat_naive_agg):.2f}, <5sig: {pct_naive:.1f}%\n"
                 rf"$\kappa$: mean={np.mean(flat_kappa_agg):.2f}, <5sig: {pct_kappa:.1f}%")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # [3,1]: All tilted pairs aggregated
    ax = axes[3, 1]
    tilted_naive_agg = np.concatenate(tilted_naive_all) if tilted_naive_all else np.array([])
    tilted_kappa_agg = np.concatenate(tilted_kappa_all) if tilted_kappa_all else np.array([])
    if len(tilted_naive_agg) > 0:
        ax.hist(tilted_naive_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="blue", alpha=0.8, label="Naive dPhiDr")
    if len(tilted_kappa_agg) > 0:
        ax.hist(tilted_kappa_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="red", alpha=0.8, label=r"$\kappa$-corrected")
    ax.axvline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("All Tilted Barrel Pairs", fontsize=10)
    ax.set_xlabel("Significance", fontsize=9)
    ax.set_ylabel("Pairs", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if len(tilted_naive_agg) > 0 and len(tilted_kappa_agg) > 0:
        pct_naive = np.sum(tilted_naive_agg < 5.0) / len(tilted_naive_agg) * 100
        pct_kappa = np.sum(tilted_kappa_agg < 5.0) / len(tilted_kappa_agg) * 100
        stats = (f"Naive: mean={np.mean(tilted_naive_agg):.2f}, <5sig: {pct_naive:.1f}%\n"
                 rf"$\kappa$: mean={np.mean(tilted_kappa_agg):.2f}, <5sig: {pct_kappa:.1f}%")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # [3,2]: Per-layer kappa means (should be constant across layers for same pT)
    ax = axes[3, 2]
    flat_means = []
    flat_stds = []
    naive_means = []
    naive_stds = []
    layers_found = []
    for l in range(1, 7):
        mask = is_barrel & is_flat & (layer == l)
        if np.sum(mask) > 10:
            layers_found.append(l)
            flat_means.append(np.mean(kappa_all[mask]))
            flat_stds.append(np.std(kappa_all[mask]))
            naive_means.append(np.mean(np.abs(dPhiDr[mask])))
            naive_stds.append(np.std(np.abs(dPhiDr[mask])))
    if layers_found:
        x = np.array(layers_found)
        ax.errorbar(x - 0.1, naive_means, yerr=naive_stds, fmt="o-", color="blue",
                     markersize=5, capsize=3, label="|dPhiDr| mean+/-std")
        ax.errorbar(x + 0.1, flat_means, yerr=flat_stds, fmt="s-", color="red",
                     markersize=5, capsize=3, label=r"$\kappa$ mean+/-std")
    ax.set_xlabel("Barrel Layer", fontsize=9)
    ax.set_ylabel("Value (rad/cm)", fontsize=9)
    ax.set_title(r"Per-Layer Mean: |dPhiDr| vs $\kappa$ (Flat)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [3,3]: Explanation text
    ax = axes[3, 3]
    ax.axis("off")
    explanation = r"""
    CURVATURE EXTRACTION ($\kappa$):

    For a helix: $|d\phi/dr| = \kappa / \sqrt{1 - (r\kappa)^2}$

    Raw dPhiDr increases with r (visible in
    row 0 and [3,2] blue points shifting up).

    Extracting:
      $\kappa$ = |dPhiDr| / $\sqrt{1 + r^2 \cdot dPhiDr^2}$

    gives a radius-independent invariant
    equal to $0.3 B / (2 p_T)$ for all layers.

    The $\kappa$ significance (red) should be
    narrower and more centered than naive
    dPhiDr significance (blue), especially
    for low-$p_T$ tracks on well-separated layers.

    Error propagation:
      $\sigma_\kappa = \sigma_{dPhiDr} / (1 + r^2 dPhiDr^2)^{3/2}$
    """
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    # [3,4]: unused
    axes[3, 4].set_visible(False)

    # ========================================================================
    # Row 4: kappa vs stub_r scatter (flat barrel, per layer)
    # ========================================================================
    # [4,0]: scatter of kappa vs r (should show horizontal bands per pT)
    ax = axes[4, 0]
    mask_flat_barrel = is_barrel & is_flat
    if np.sum(mask_flat_barrel) > 0:
        ax.scatter(r[mask_flat_barrel], kappa_all[mask_flat_barrel],
                   s=1, alpha=0.2, c="red", label=r"$\kappa$")
        ax.scatter(r[mask_flat_barrel], np.abs(dPhiDr[mask_flat_barrel]),
                   s=1, alpha=0.2, c="blue", label="|dPhiDr|")
    ax.set_xlabel("stub_r (cm)", fontsize=9)
    ax.set_ylabel("Value (rad/cm)", fontsize=9)
    ax.set_title(r"$\kappa$ vs |dPhiDr| vs Radius (Flat)", fontsize=10)
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(True, alpha=0.3)

    # [4,1]: same for tilted
    ax = axes[4, 1]
    mask_tilted_barrel = is_barrel & is_tilted
    if np.sum(mask_tilted_barrel) > 0:
        ax.scatter(r[mask_tilted_barrel], kappa_all[mask_tilted_barrel],
                   s=1, alpha=0.2, c="red", label=r"$\kappa$")
        ax.scatter(r[mask_tilted_barrel], np.abs(dPhiDr[mask_tilted_barrel]),
                   s=1, alpha=0.2, c="blue", label="|dPhiDr|")
    ax.set_xlabel("stub_r (cm)", fontsize=9)
    ax.set_ylabel("Value (rad/cm)", fontsize=9)
    ax.set_title(r"$\kappa$ vs |dPhiDr| vs Radius (Tilted)", fontsize=10)
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(True, alpha=0.3)

    # [4,2]: kappa_error vs r (to visualize error scaling)
    ax = axes[4, 2]
    if np.sum(mask_flat_barrel) > 0:
        valid = mask_flat_barrel & (dPhiDrErr > 0) & (dPhiDrErr < 1.0)
        if np.sum(valid) > 0:
            ax.scatter(r[valid], kappa_err_all[valid], s=1, alpha=0.2,
                       c="red", label=r"$\sigma_\kappa$")
            ax.scatter(r[valid], dPhiDrErr[valid], s=1, alpha=0.2,
                       c="blue", label=r"$\sigma_{dPhiDr}$")
    ax.set_xlabel("stub_r (cm)", fontsize=9)
    ax.set_ylabel("Error (rad/cm)", fontsize=9)
    ax.set_title(r"$\sigma_\kappa$ vs $\sigma_{dPhiDr}$ vs Radius", fontsize=10)
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(True, alpha=0.3)

    # [4,3]: pT estimate from kappa (kappa = 0.3*B/(2*pT) => pT = 0.3*B/(2*kappa))
    ax = axes[4, 3]
    valid_kappa = mask_flat_barrel & (kappa_all > 1e-6)
    if np.sum(valid_kappa) > 0:
        pt_from_kappa = 0.3 * 3.8 / (2.0 * kappa_all[valid_kappa])
        pt_est = data["ptEst"][valid_kappa]
        # Only show reasonable pT range
        pt_mask = (pt_from_kappa > 0) & (pt_from_kappa < 200) & (pt_est > 0) & (pt_est < 200)
        if np.sum(pt_mask) > 0:
            ax.scatter(pt_est[pt_mask], pt_from_kappa[pt_mask], s=1, alpha=0.2)
            ax.plot([0, 200], [0, 200], "r--", linewidth=1, alpha=0.5, label="y=x")
    ax.set_xlabel("ptEst from stub (GeV)", fontsize=9)
    ax.set_ylabel(r"$p_T$ from $\kappa$ (GeV)", fontsize=9)
    ax.set_title(r"$p_T$ Consistency: Stub vs $\kappa$", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [4,4]: unused
    axes[4, 4].set_visible(False)

    plt.tight_layout()
    return fig


def plot_forward_dPhiDz_significance(data, figsize=(22, 20)):
    """
    Pairwise |dPhiDz_i - dPhiDz_j| / combined_error for forward (endcap) stubs
    on consecutive disks.

    In the endcap, modules lie approximately in the r-phi plane and the two
    sensors are separated along z.  dPhiDz = dphi/dz is therefore the natural,
    well-conditioned curvature-like variable (analogous to dPhiDr for flat barrel).

    Layout:
      Row 0: Per-disk dPhiDz distributions (Fwd+, D1-D5)
      Row 1: Pairwise dPhiDz significance for Fwd+ consecutive disk pairs
      Row 2: Pairwise dPhiDz significance for Fwd- consecutive disk pairs
      Row 3: Aggregated comparison, per-disk mean dPhiDz, explanation
    """
    fig, axes = plt.subplots(4, 5, figsize=figsize)
    fig.suptitle("Forward (Endcap) Stub Linking: Pairwise |dPhiDz| Significance",
                 fontsize=14, fontweight="bold")

    is_endcap = data["isEndcap"] == True
    layer = data["layer"]
    stub_z = data["stub_z"]

    fwd_plus = is_endcap & (stub_z > 0)
    fwd_minus = is_endcap & (stub_z < 0)

    # ========================================================================
    # Row 0: Per-disk dPhiDz distributions (Fwd+)
    # ========================================================================
    for col, d in enumerate(range(1, 6)):
        ax = axes[0, col]
        mask = fwd_plus & (layer == d)
        n = np.sum(mask)
        if n > 0:
            vals = data["dPhiDz"][mask]
            ax.hist(vals, bins=100, range=(-0.01, 0.01), histtype="step",
                    linewidth=1.5, color="green",
                    label=f"Fwd+ D{d} (N={n})")
            stats = (f"N={n}\nmean={np.mean(vals):.6f}\n"
                     f"std={np.std(vals):.6f}")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"Fwd+ Disk {d}", fontsize=10)
        ax.set_xlabel("dPhiDz [rad/cm]", fontsize=8)
        ax.set_ylabel("Stubs", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 1: Fwd+ pairwise significance per consecutive disk pair
    # ========================================================================
    fwd_plus_sig_all = []
    for col, d_inner in enumerate(range(1, 5)):
        ax = axes[1, col]
        m1 = fwd_plus & (layer == d_inner)
        m2 = fwd_plus & (layer == d_inner + 1)

        sig = compute_pairwise_significance(data, m1, m2, "dPhiDz", "dPhiDzError")
        if len(sig) > 0:
            fwd_plus_sig_all.append(sig)

        xrange = (0, 10)
        if len(sig) > 0:
            n_pairs = len(sig)
            ax.hist(sig, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="green", alpha=0.8,
                    label=f"Fwd+ (N={n_pairs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=r"5$\sigma$ cut")
            within_5sigma = np.sum(sig < 5.0) / n_pairs * 100
            stats = (f"N={n_pairs}\nmean={np.mean(sig):.3f}\n"
                     f"std={np.std(sig):.3f}\n{within_5sigma:.1f}% < 5sig")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"Fwd+ D{d_inner}->D{d_inner+1}", fontsize=9)
        ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[1, 4].set_visible(False)

    # ========================================================================
    # Row 2: Fwd- pairwise significance per consecutive disk pair
    # ========================================================================
    fwd_minus_sig_all = []
    for col, d_inner in enumerate(range(1, 5)):
        ax = axes[2, col]
        m1 = fwd_minus & (layer == d_inner)
        m2 = fwd_minus & (layer == d_inner + 1)

        sig = compute_pairwise_significance(data, m1, m2, "dPhiDz", "dPhiDzError")
        if len(sig) > 0:
            fwd_minus_sig_all.append(sig)

        xrange = (0, 10)
        if len(sig) > 0:
            n_pairs = len(sig)
            ax.hist(sig, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="purple", alpha=0.8,
                    label=f"Fwd- (N={n_pairs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5,
                       alpha=0.7, label=r"5$\sigma$ cut")
            within_5sigma = np.sum(sig < 5.0) / n_pairs * 100
            stats = (f"N={n_pairs}\nmean={np.mean(sig):.3f}\n"
                     f"std={np.std(sig):.3f}\n{within_5sigma:.1f}% < 5sig")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"Fwd- D{d_inner}->D{d_inner+1}", fontsize=9)
        ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[2, 4].set_visible(False)

    # ========================================================================
    # Row 3: Aggregated comparison, per-disk means, explanation
    # ========================================================================
    # [3,0]: All Fwd+ pairs aggregated
    ax = axes[3, 0]
    fwd_plus_agg = np.concatenate(fwd_plus_sig_all) if fwd_plus_sig_all else np.array([])
    if len(fwd_plus_agg) > 0:
        ax.hist(fwd_plus_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="green", alpha=0.8,
                label=f"Fwd+ (N={len(fwd_plus_agg)})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        pct = np.sum(fwd_plus_agg < 5.0) / len(fwd_plus_agg) * 100
        stats = (f"N={len(fwd_plus_agg)}\nmean={np.mean(fwd_plus_agg):.3f}\n"
                 f"std={np.std(fwd_plus_agg):.3f}\n{pct:.1f}% < 5sig")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("All Fwd+ Disk Pairs", fontsize=10)
    ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=9)
    ax.set_ylabel("Pairs", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,1]: All Fwd- pairs aggregated
    ax = axes[3, 1]
    fwd_minus_agg = np.concatenate(fwd_minus_sig_all) if fwd_minus_sig_all else np.array([])
    if len(fwd_minus_agg) > 0:
        ax.hist(fwd_minus_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="purple", alpha=0.8,
                label=f"Fwd- (N={len(fwd_minus_agg)})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        pct = np.sum(fwd_minus_agg < 5.0) / len(fwd_minus_agg) * 100
        stats = (f"N={len(fwd_minus_agg)}\nmean={np.mean(fwd_minus_agg):.3f}\n"
                 f"std={np.std(fwd_minus_agg):.3f}\n{pct:.1f}% < 5sig")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("All Fwd- Disk Pairs", fontsize=10)
    ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=9)
    ax.set_ylabel("Pairs", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,2]: Fwd+ vs Fwd- overlay
    ax = axes[3, 2]
    if len(fwd_plus_agg) > 0:
        ax.hist(fwd_plus_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="green", alpha=0.8, label="Fwd+")
    if len(fwd_minus_agg) > 0:
        ax.hist(fwd_minus_agg, bins=100, range=(0, 10), histtype="step",
                linewidth=2, color="purple", alpha=0.8, label="Fwd-")
    ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_title("Fwd+ vs Fwd- Overlay", fontsize=10)
    ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=9)
    ax.set_ylabel("Pairs", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,3]: Per-disk mean |dPhiDz| for Fwd+
    ax = axes[3, 3]
    disk_labels = []
    fwd_means = []
    fwd_stds = []
    for d in range(1, 6):
        mask = fwd_plus & (layer == d)
        if np.sum(mask) > 10:
            disk_labels.append(d)
            vals = np.abs(data["dPhiDz"][mask])
            fwd_means.append(np.mean(vals))
            fwd_stds.append(np.std(vals))
    if disk_labels:
        x = np.array(disk_labels)
        ax.errorbar(x, fwd_means, yerr=fwd_stds, fmt="o-", color="green",
                     markersize=5, capsize=3, label="Fwd+ |dPhiDz| mean+/-std")
    ax.set_xlabel("Endcap Disk", fontsize=9)
    ax.set_ylabel("|dPhiDz| (rad/cm)", fontsize=9)
    ax.set_title("Per-Disk Mean |dPhiDz| (Fwd+)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [3,4]: Explanation text
    ax = axes[3, 4]
    ax.axis("off")
    explanation = (
        "FORWARD STUB LINKING (dPhiDz):\n\n"
        "Endcap modules lie in the r-phi plane\n"
        "with sensor separation along z.\n"
        "dPhiDz = dphi/dz is the natural variable\n"
        "(analogous to dPhiDr for flat barrel).\n\n"
        "Pairwise significance:\n"
        r"  |dPhiDz$_i$ - dPhiDz$_j$| / $\sigma_{comb}$" + "\n"
        r"  $\sigma_{comb}$ = $\sqrt{\sigma_i^2 + \sigma_j^2}$" + "\n\n"
        "For same-track stubs on consecutive disks,\n"
        "this should be small (peaked near 0).\n"
        "Different-track pairs should be broadly\n"
        "distributed.\n\n"
        "Note: dPhiDz depends on both pT and eta\n"
        "(unlike barrel dPhiDr which depends on\n"
        "pT only). A z-dependent correction\n"
        "analogous to kappa may be needed if\n"
        "systematic tails appear."
    )
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightgreen",
                      edgecolor="green", alpha=0.9))

    plt.tight_layout()
    return fig


def plot_forward_dPhiDz_diagnostic(data, figsize=(28, 30)):
    """
    Diagnostic comparing endcap pairwise significance using dPhiDr (kernel) vs dPhiDz (ntuplizer).

    This helps identify whether the FWD dPhiDz significance issue is due to:
    (a) The dPhiDz/dPhiDzError formula being wrong, or
    (b) Ghost stubs / physics effects affecting all endcap variables equally

    Layout (5 rows x 5 cols):
      Row 0: Fwd+ pairwise significance using KERNEL dPhiDr (for comparison)
      Row 1: Fwd+ pairwise significance using NTUPLIZER dPhiDz (the "bad" one)
      Row 2: Overlay comparison per disk pair + aggregated
      Row 3: Diagnostic: dz_sensor, stub multiplicity, dPhiDz vs dPhiDr scatter
      Row 4: Targeted diagnostics: value scatter, signed diff, pull, per-disk errors, dr_effective
    """
    fig, axes = plt.subplots(5, 5, figsize=figsize, constrained_layout=True)
    fig.suptitle("FWD dPhiDz Diagnostic: dPhiDr (kernel) vs dPhiDz (ntuplizer) Significance",
                 fontsize=14, fontweight="bold")

    is_endcap = data["isEndcap"] == True
    layer = data["layer"]
    stub_z = data["stub_z"]
    stub_r = data["stub_r"]
    events = data["event"]

    fwd_plus = is_endcap & (stub_z > 0)

    # ========================================================================
    # Row 0: Fwd+ pairwise significance using KERNEL dPhiDr
    # ========================================================================
    fwd_plus_dPhiDr_sig_all = []
    for col, d_inner in enumerate(range(1, 5)):
        ax = axes[0, col]
        m1 = fwd_plus & (layer == d_inner)
        m2 = fwd_plus & (layer == d_inner + 1)

        sig = compute_pairwise_significance(data, m1, m2, "dPhiDr", "dPhiDrError")
        if len(sig) > 0:
            fwd_plus_dPhiDr_sig_all.append(sig)

        xrange = (0, 10)
        if len(sig) > 0:
            n_pairs = len(sig)
            ax.hist(sig, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="blue", alpha=0.8,
                    label=f"dPhiDr (N={n_pairs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            within_5sigma = np.sum(sig < 5.0) / n_pairs * 100
            stats = (f"N={n_pairs}\nmean={np.mean(sig):.3f}\n"
                     f"std={np.std(sig):.3f}\n{within_5sigma:.1f}% < 5sig")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"dPhiDr: Fwd+ D{d_inner}->D{d_inner+1}", fontsize=9)
        ax.set_xlabel(r"|$\Delta$dPhiDr| / $\sigma_{comb}$", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Row 0, col 4: aggregated dPhiDr
    ax = axes[0, 4]
    agg_r = np.concatenate(fwd_plus_dPhiDr_sig_all) if fwd_plus_dPhiDr_sig_all else np.array([])
    if len(agg_r) > 0:
        ax.hist(agg_r, bins=80, range=(0, 10), histtype="step",
                linewidth=2, color="blue", alpha=0.8, label=f"All (N={len(agg_r)})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        pct = np.sum(agg_r < 5.0) / len(agg_r) * 100
        stats = (f"N={len(agg_r)}\nmean={np.mean(agg_r):.3f}\n"
                 f"std={np.std(agg_r):.3f}\n{pct:.1f}% < 5sig")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("dPhiDr: All Fwd+ Pairs", fontsize=10)
    ax.set_xlabel(r"|$\Delta$dPhiDr| / $\sigma_{comb}$", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 1: Fwd+ pairwise significance using NTUPLIZER dPhiDz
    # ========================================================================
    fwd_plus_dPhiDz_sig_all = []
    for col, d_inner in enumerate(range(1, 5)):
        ax = axes[1, col]
        m1 = fwd_plus & (layer == d_inner)
        m2 = fwd_plus & (layer == d_inner + 1)

        sig = compute_pairwise_significance(data, m1, m2, "dPhiDz", "dPhiDzError")
        if len(sig) > 0:
            fwd_plus_dPhiDz_sig_all.append(sig)

        xrange = (0, 10)
        if len(sig) > 0:
            n_pairs = len(sig)
            ax.hist(sig, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="green", alpha=0.8,
                    label=f"dPhiDz (N={n_pairs})")
            ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            within_5sigma = np.sum(sig < 5.0) / n_pairs * 100
            stats = (f"N={n_pairs}\nmean={np.mean(sig):.3f}\n"
                     f"std={np.std(sig):.3f}\n{within_5sigma:.1f}% < 5sig")
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"dPhiDz: Fwd+ D{d_inner}->D{d_inner+1}", fontsize=9)
        ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Row 1, col 4: aggregated dPhiDz
    ax = axes[1, 4]
    agg_z = np.concatenate(fwd_plus_dPhiDz_sig_all) if fwd_plus_dPhiDz_sig_all else np.array([])
    if len(agg_z) > 0:
        ax.hist(agg_z, bins=80, range=(0, 10), histtype="step",
                linewidth=2, color="green", alpha=0.8, label=f"All (N={len(agg_z)})")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        pct = np.sum(agg_z < 5.0) / len(agg_z) * 100
        stats = (f"N={len(agg_z)}\nmean={np.mean(agg_z):.3f}\n"
                 f"std={np.std(agg_z):.3f}\n{pct:.1f}% < 5sig")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("dPhiDz: All Fwd+ Pairs", fontsize=10)
    ax.set_xlabel(r"|$\Delta$dPhiDz| / $\sigma_{comb}$", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 2: Overlay comparison per disk pair
    # ========================================================================
    for col, d_inner in enumerate(range(1, 5)):
        ax = axes[2, col]
        m1 = fwd_plus & (layer == d_inner)
        m2 = fwd_plus & (layer == d_inner + 1)

        sig_r = compute_pairwise_significance(data, m1, m2, "dPhiDr", "dPhiDrError")
        sig_z = compute_pairwise_significance(data, m1, m2, "dPhiDz", "dPhiDzError")

        xrange = (0, 10)
        if len(sig_r) > 0:
            ax.hist(sig_r, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="blue", alpha=0.8, label="dPhiDr")
        if len(sig_z) > 0:
            ax.hist(sig_z, bins=80, range=xrange, histtype="step",
                    linewidth=1.5, color="green", alpha=0.8, label="dPhiDz")
        ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_title(f"Overlay: D{d_inner}->D{d_inner+1}", fontsize=9)
        ax.set_xlabel("Significance", fontsize=8)
        ax.set_ylabel("Pairs", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2, col 4: aggregated overlay
    ax = axes[2, 4]
    if len(agg_r) > 0:
        ax.hist(agg_r, bins=80, range=(0, 10), histtype="step",
                linewidth=2, color="blue", alpha=0.8, label="dPhiDr")
    if len(agg_z) > 0:
        ax.hist(agg_z, bins=80, range=(0, 10), histtype="step",
                linewidth=2, color="green", alpha=0.8, label="dPhiDz")
    ax.axvline(5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_title("All Fwd+ Pairs Overlay", fontsize=10)
    ax.set_xlabel("Significance", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 3: Diagnostic plots
    # ========================================================================
    # [3,0]: dz_sensor distribution for forward endcap
    ax = axes[3, 0]
    if "dz_sensor" in data and np.sum(fwd_plus) > 0:
        dz_vals = data["dz_sensor"][fwd_plus]
        ax.hist(dz_vals, bins=100, histtype="step", linewidth=1.5, color="green")
        stats = (f"N={len(dz_vals)}\nmean={np.mean(dz_vals):.5f}\n"
                 f"std={np.std(dz_vals):.5f}\n"
                 f"min={np.min(dz_vals):.5f}\nmax={np.max(dz_vals):.5f}")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("dz_sensor (Fwd+) [cm]", fontsize=9)
    ax.set_xlabel("dz_sensor [cm]", fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,1]: Stub multiplicity per event per disk (Fwd+)
    ax = axes[3, 1]
    mult_data = {}
    for d in range(1, 6):
        mask_d = fwd_plus & (layer == d)
        ev_d = events[mask_d]
        if len(ev_d) > 0:
            unique_evts, counts = np.unique(ev_d, return_counts=True)
            mult_data[d] = counts
    if mult_data:
        positions = []
        labels = []
        all_counts = []
        for d in sorted(mult_data.keys()):
            positions.append(d)
            labels.append(f"D{d}")
            all_counts.append(mult_data[d])
        bp = ax.boxplot(all_counts, positions=positions, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        for d in sorted(mult_data.keys()):
            c = mult_data[d]
            stats_txt = f"med={np.median(c):.0f}\nmean={np.mean(c):.1f}"
            ax.text(d, np.max(c) + 0.2, stats_txt, ha="center", fontsize=6)
    ax.set_title("Stubs/event/disk (Fwd+)", fontsize=9)
    ax.set_ylabel("Stubs per event", fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,2]: dPhiDz vs dPhiDr * (r/z) scatter for endcap
    # If dPhiDz = dphi/dz and dPhiDr = dphi/dr_eff, then dPhiDz * (z/r) should equal dPhiDr
    # (approximately, for endcap where dr_eff ≈ sep*r/z and dz ≈ sep)
    ax = axes[3, 2]
    if "dPhiDr" in data and "dPhiDz" in data and np.sum(fwd_plus) > 0:
        fwd_dPhiDr = data["dPhiDr"][fwd_plus]
        fwd_dPhiDz = data["dPhiDz"][fwd_plus]
        fwd_r = stub_r[fwd_plus]
        fwd_z = stub_z[fwd_plus]
        # dPhiDz * z/r should approximately equal dPhiDr
        dPhiDz_scaled = fwd_dPhiDz * fwd_z / fwd_r
        ax.scatter(fwd_dPhiDr, dPhiDz_scaled, s=0.5, alpha=0.3, color="green")
        # Add diagonal
        lim = max(np.abs(fwd_dPhiDr).max(), np.abs(dPhiDz_scaled).max()) if len(fwd_dPhiDr) > 0 else 0.01
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label="y=x")
        ax.set_xlabel("dPhiDr (kernel)", fontsize=8)
        ax.set_ylabel("dPhiDz * z/r", fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title("dPhiDz*z/r vs dPhiDr", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # [3,3]: dPhiDzError vs dPhiDrError * (r/z) scatter
    ax = axes[3, 3]
    if "dPhiDrError" in data and "dPhiDzError" in data and np.sum(fwd_plus) > 0:
        fwd_dPhiDrErr = data["dPhiDrError"][fwd_plus]
        fwd_dPhiDzErr = data["dPhiDzError"][fwd_plus]
        fwd_r = stub_r[fwd_plus]
        fwd_z = stub_z[fwd_plus]
        # dPhiDzError * z/r should approximately equal dPhiDrError
        dPhiDzErr_scaled = fwd_dPhiDzErr * fwd_z / fwd_r
        ax.scatter(fwd_dPhiDrErr, dPhiDzErr_scaled, s=0.5, alpha=0.3, color="green")
        lim = max(fwd_dPhiDrErr.max(), dPhiDzErr_scaled.max()) if len(fwd_dPhiDrErr) > 0 else 0.01
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label="y=x")
        ax.set_xlabel("dPhiDrError (kernel)", fontsize=8)
        ax.set_ylabel("dPhiDzError * z/r", fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title("Errors: dPhiDzErr*z/r vs dPhiDrErr", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # [3,4]: inner/outer xerrLocal for endcap
    ax = axes[3, 4]
    if "inner_xerrLocal" in data and "outer_xerrLocal" in data and np.sum(fwd_plus) > 0:
        inner_xe = data["inner_xerrLocal"][fwd_plus]
        outer_xe = data["outer_xerrLocal"][fwd_plus]
        inner_sigma = np.sqrt(inner_xe) * 1e4  # cm to um
        outer_sigma = np.sqrt(outer_xe) * 1e4  # cm to um
        ax.hist(inner_sigma, bins=100, range=(0, 100), histtype="step",
                linewidth=1.5, color="blue", label=f"inner (mean={np.mean(inner_sigma):.1f}um)")
        ax.hist(outer_sigma, bins=100, range=(0, 100), histtype="step",
                linewidth=1.5, color="red", label=f"outer (mean={np.mean(outer_sigma):.1f}um)")
    ax.set_title("xerrLocal sigma (Fwd+)", fontsize=9)
    ax.set_xlabel("sigma_x [um]", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Row 4: Targeted diagnostics for understanding pairwise significance
    # ========================================================================

    # --- [4,0]: 2D scatter of dPhiDz on D1 vs dPhiDz on D2 for same-event pairs ---
    ax = axes[4, 0]
    # Use D1->D2 as example pair
    m1_scatter = fwd_plus & (layer == 1)
    m2_scatter = fwd_plus & (layer == 2)
    pull_data_12 = compute_pairwise_signed_pull(data, m1_scatter, m2_scatter, "dPhiDz", "dPhiDzError")
    if len(pull_data_12["val1"]) > 0:
        v1_plot = pull_data_12["val1"]
        v2_plot = pull_data_12["val2"]
        ax.scatter(v1_plot, v2_plot, s=2, alpha=0.4, color="green")
        # Plot y=x diagonal
        all_vals = np.concatenate([v1_plot, v2_plot])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        margin = 0.1 * (vmax - vmin) if vmax > vmin else 0.001
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1.5, label="y=x")
        # Correlation coefficient
        if len(v1_plot) > 1:
            corr = np.corrcoef(v1_plot, v2_plot)[0, 1]
            ax.text(0.05, 0.95, f"N={len(v1_plot)}\ncorr={corr:.4f}",
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("dPhiDz on D1", fontsize=8)
        ax.set_ylabel("dPhiDz on D2", fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title("dPhiDz: D1 vs D2 (same-event)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- [4,1]: Signed pairwise difference distribution (all D_i -> D_{i+1}) ---
    ax = axes[4, 1]
    all_signed_diffs = []
    for d_inner in range(1, 5):
        m1_d = fwd_plus & (layer == d_inner)
        m2_d = fwd_plus & (layer == d_inner + 1)
        pd = compute_pairwise_signed_pull(data, m1_d, m2_d, "dPhiDz", "dPhiDzError")
        if len(pd["signed_diff"]) > 0:
            all_signed_diffs.append(pd["signed_diff"])
    if all_signed_diffs:
        agg_diff = np.concatenate(all_signed_diffs)
        # Determine symmetric range
        abs_max = np.percentile(np.abs(agg_diff), 99)
        xrange_diff = (-abs_max, abs_max)
        ax.hist(agg_diff, bins=80, range=xrange_diff, histtype="step",
                linewidth=1.5, color="green", alpha=0.8)
        mean_d = np.mean(agg_diff)
        std_d = np.std(agg_diff)
        median_d = np.median(agg_diff)
        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.axvline(mean_d, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"mean={mean_d:.2e}")
        stats = (f"N={len(agg_diff)}\nmean={mean_d:.3e}\n"
                 f"median={median_d:.3e}\nstd={std_d:.3e}")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=7)
    ax.set_title("Signed diff: dPhiDz_1 - dPhiDz_2 (all pairs)", fontsize=9)
    ax.set_xlabel(r"$\Delta$dPhiDz (signed)", fontsize=8)
    ax.set_ylabel("Pairs", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- [4,2]: Pull distribution with Gaussian overlay ---
    ax = axes[4, 2]
    all_pulls = []
    for d_inner in range(1, 5):
        m1_d = fwd_plus & (layer == d_inner)
        m2_d = fwd_plus & (layer == d_inner + 1)
        pd = compute_pairwise_signed_pull(data, m1_d, m2_d, "dPhiDz", "dPhiDzError")
        if len(pd["pull"]) > 0:
            all_pulls.append(pd["pull"])
    if all_pulls:
        agg_pull = np.concatenate(all_pulls)
        pull_range = (-10, 10)
        n_bins_pull = 80
        counts, bin_edges, _ = ax.hist(agg_pull, bins=n_bins_pull, range=pull_range,
                                       histtype="step", linewidth=1.5, color="green",
                                       alpha=0.8, label="Data")
        mean_p = np.mean(agg_pull)
        std_p = np.std(agg_pull)
        # Overlay Gaussian fit: N(mean_p, std_p)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]
        gauss = (len(agg_pull) * bin_width / (std_p * np.sqrt(2 * np.pi))
                 * np.exp(-0.5 * ((bin_centers - mean_p) / std_p)**2))
        ax.plot(bin_centers, gauss, 'r-', linewidth=2, alpha=0.8,
                label=f"Gauss: $\\mu$={mean_p:.2f}, $\\sigma$={std_p:.2f}")
        # Also overlay unit Gaussian for reference
        gauss_unit = (len(agg_pull) * bin_width / np.sqrt(2 * np.pi)
                      * np.exp(-0.5 * bin_centers**2))
        ax.plot(bin_centers, gauss_unit, 'b--', linewidth=1.5, alpha=0.6,
                label="N(0,1) expected")
        stats = (f"N={len(agg_pull)}\nmean={mean_p:.3f}\nstd={std_p:.3f}\n"
                 f"|mean|/std={abs(mean_p)/std_p:.3f}" if std_p > 0 else f"N={len(agg_pull)}")
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=6)
    ax.set_title("Pull: (dPhiDz_1 - dPhiDz_2) / $\\sigma_{comb}$", fontsize=9)
    ax.set_xlabel("Pull (signed)", fontsize=8)
    ax.set_ylabel("Pairs", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- [4,3]: Per-disk dPhiDrError distribution (kernel error) for Fwd+ ---
    ax = axes[4, 3]
    disk_colors = {1: "blue", 2: "orange", 3: "green", 4: "red", 5: "purple"}
    if "dPhiDrError" in data:
        for d in range(1, 6):
            mask_d = fwd_plus & (layer == d)
            n_d = np.sum(mask_d)
            if n_d > 0:
                err_vals = data["dPhiDrError"][mask_d]
                ax.hist(err_vals, bins=80, histtype="step", linewidth=1.5,
                        color=disk_colors[d], alpha=0.8,
                        label=f"D{d} (N={n_d}, mean={np.mean(err_vals):.2e})")
    ax.set_title("dPhiDrError per disk (Fwd+)", fontsize=9)
    ax.set_xlabel("dPhiDrError", fontsize=8)
    ax.set_ylabel("Stubs", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # --- [4,4]: Per-disk dr_effective distribution for Fwd+ ---
    ax = axes[4, 4]
    # dr_effective = separation * r / z for endcap
    if "dz_sensor" in data:
        for d in range(1, 6):
            mask_d = fwd_plus & (layer == d)
            n_d = np.sum(mask_d)
            if n_d > 0:
                sep = np.abs(data["dz_sensor"][mask_d])
                r_d = stub_r[mask_d]
                z_d = np.abs(stub_z[mask_d])
                dr_eff = sep * r_d / z_d
                ax.hist(dr_eff, bins=80, histtype="step", linewidth=1.5,
                        color=disk_colors[d], alpha=0.8,
                        label=f"D{d} (N={n_d}, mean={np.mean(dr_eff):.4f})")
    ax.set_title("dr_effective = sep*r/z per disk (Fwd+)", fontsize=9)
    ax.set_xlabel("dr_effective [cm]", fontsize=8)
    ax.set_ylabel("Stubs", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Save individual row PDFs for detailed inspection
    # ========================================================================
    row_names = [
        "fwd_diag_row0_dPhiDr_significance",
        "fwd_diag_row1_dPhiDz_significance",
        "fwd_diag_row2_overlay_comparison",
        "fwd_diag_row3_diagnostics",
        "fwd_diag_row4_pulls_and_errors",
    ]
    row_titles = [
        "dPhiDr (kernel) Significance",
        "dPhiDz (ntuplizer) Significance",
        "Overlay Comparison",
        "Diagnostics",
        "Pulls and Errors",
    ]

    # Save the full figure as a single PDF
    fig.savefig("fwd_diag_full.pdf", dpi=150, bbox_inches="tight")
    print("Saved fwd_diag_full.pdf")

    # Save each row as a separate PDF by hiding all other rows' axes
    for row_idx in range(5):
        # Track visibility state of all axes so we can restore it
        visibility = {}
        for r in range(5):
            for c in range(5):
                ax = axes[r, c]
                visibility[(r, c)] = ax.get_visible()
                if r != row_idx:
                    ax.set_visible(False)

        # Also hide the suptitle for individual row PDFs by saving its text and clearing it
        original_suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        fig.suptitle(row_titles[row_idx], fontsize=14, fontweight="bold")

        # Compute the bounding box that tightly encloses only the visible row
        # We use the axes positions to determine the row extent
        row_axes = [axes[row_idx, c] for c in range(5)]
        renderer = fig.canvas.get_renderer()

        # Get the extent of the row's axes in figure coordinates
        y_bottoms = []
        y_tops = []
        x_lefts = []
        x_rights = []
        for ax in row_axes:
            bbox = ax.get_tightbbox(renderer)
            if bbox is not None:
                # Convert from display to figure coordinates
                bbox_fig = bbox.transformed(fig.transFigure.inverted())
                y_bottoms.append(bbox_fig.y0)
                y_tops.append(bbox_fig.y1)
                x_lefts.append(bbox_fig.x0)
                x_rights.append(bbox_fig.x1)

        if y_bottoms:
            from matplotlib.transforms import Bbox as MplBbox
            # Add some padding and include space for the suptitle
            pad = 0.02
            # Include suptitle bbox if present
            suptitle_bbox = fig._suptitle.get_tightbbox(renderer) if fig._suptitle else None
            suptitle_top = 1.0
            if suptitle_bbox is not None:
                suptitle_fig = suptitle_bbox.transformed(fig.transFigure.inverted())
                suptitle_top = suptitle_fig.y1

            row_bbox = MplBbox.from_extents(
                max(0, min(x_lefts) - pad),
                max(0, min(y_bottoms) - pad),
                min(1, max(x_rights) + pad),
                min(1, max(max(y_tops), suptitle_top) + pad),
            )
            # Convert figure-fraction bbox to inches for savefig
            fig_w, fig_h = fig.get_size_inches()
            bbox_inches = MplBbox.from_extents(
                row_bbox.x0 * fig_w,
                row_bbox.y0 * fig_h,
                row_bbox.x1 * fig_w,
                row_bbox.y1 * fig_h,
            )

            fname = f"{row_names[row_idx]}.pdf"
            fig.savefig(fname, dpi=150, bbox_inches=bbox_inches)
            print(f"Saved {fname}")

        # Restore visibility and suptitle
        for r in range(5):
            for c in range(5):
                axes[r, c].set_visible(visibility[(r, c)])
        fig.suptitle(original_suptitle, fontsize=14, fontweight="bold")

    return fig


def plot_pairwise_significance_by_transition(data, figsize=(22, 12)):
    """
    Pairwise dPhiDr and dPhiDz significance broken down by module transition type.

    For each pair of stubs in the same event (i < j, to avoid duplicates and
    self-pairs), the module transition type is determined from the two stubs:
      - "Barrel Flat" if isBarrel==1 and isFlat==1
      - "Barrel Tilted" if isBarrel==1 and isFlat==0
      - "Disk" (endcap) if isBarrel==0

    The six transition categories (order doesn't matter within a pair):
      flat-flat, flat-tilted, tilted-tilted, flat-disk, tilted-disk, disk-disk

    Each panel shows overlaid histograms of:
      - dPhiDr pairwise significance (blue)
      - dPhiDz pairwise significance (red, semi-transparent)
    with mean/sigma annotation and a 5-sigma reference line.

    Layout: 2 rows x 3 columns (one panel per transition type).
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Pairwise Significance by Module Transition Type\n"
                 r"(|val$_i$ - val$_j$| / $\sqrt{\sigma_i^2 + \sigma_j^2}$, same-event pairs with i<j)",
                 fontsize=13, fontweight="bold")

    # Classify each stub into one of three module types:
    #   0 = Barrel Flat, 1 = Barrel Tilted, 2 = Disk (endcap)
    is_barrel = data["isBarrel"] == True
    is_flat = data["isFlat"] == True
    stub_type = np.full(len(is_barrel), 2, dtype=np.int8)  # default: disk
    stub_type[is_barrel & is_flat] = 0   # barrel flat
    stub_type[is_barrel & ~is_flat] = 1  # barrel tilted

    type_labels = {0: "Barrel Flat", 1: "Barrel Tilted", 2: "Disk"}

    # Transition categories: (type_a, type_b) with type_a <= type_b
    # Maps to (row, col) in the 2x3 grid
    transition_config = {
        (0, 0): {"label": "Flat - Flat", "pos": (0, 0)},
        (0, 1): {"label": "Flat - Tilted", "pos": (0, 1)},
        (1, 1): {"label": "Tilted - Tilted", "pos": (0, 2)},
        (0, 2): {"label": "Flat - Disk", "pos": (1, 0)},
        (1, 2): {"label": "Tilted - Disk", "pos": (1, 1)},
        (2, 2): {"label": "Disk - Disk", "pos": (1, 2)},
    }

    # Collect significance values per transition type
    sig_r_by_trans = {k: [] for k in transition_config}
    sig_z_by_trans = {k: [] for k in transition_config}

    events = data["event"]
    dPhiDr_arr = data["dPhiDr"]
    dPhiDrErr_arr = data["dPhiDrError"]
    dPhiDz_arr = data["dPhiDz"]
    dPhiDzErr_arr = data["dPhiDzError"]

    unique_events = np.unique(events)

    for evt in unique_events:
        evt_mask = events == evt
        idx = np.where(evt_mask)[0]
        n = len(idx)
        if n < 2:
            continue

        types_evt = stub_type[idx]
        dr_vals = dPhiDr_arr[idx]
        dr_errs = dPhiDrErr_arr[idx]
        dz_vals = dPhiDz_arr[idx]
        dz_errs = dPhiDzErr_arr[idx]

        for i in range(n):
            for j in range(i + 1, n):
                ta, tb = types_evt[i], types_evt[j]
                key = (min(ta, tb), max(ta, tb))

                # dPhiDr significance
                cerr_r = np.sqrt(dr_errs[i]**2 + dr_errs[j]**2)
                if cerr_r > 0:
                    sig_r = np.abs(dr_vals[i] - dr_vals[j]) / cerr_r
                    sig_r_by_trans[key].append(sig_r)

                # dPhiDz significance
                cerr_z = np.sqrt(dz_errs[i]**2 + dz_errs[j]**2)
                if cerr_z > 0:
                    sig_z = np.abs(dz_vals[i] - dz_vals[j]) / cerr_z
                    sig_z_by_trans[key].append(sig_z)

    # Plot each transition type
    xrange = (0, 15)
    nbins = 100
    for trans_key, config in transition_config.items():
        row, col = config["pos"]
        ax = axes[row, col]
        label = config["label"]

        sr = np.array(sig_r_by_trans[trans_key])
        sz = np.array(sig_z_by_trans[trans_key])

        has_r = len(sr) > 0
        has_z = len(sz) > 0

        if has_r:
            ax.hist(sr, bins=nbins, range=xrange, histtype="stepfilled",
                    linewidth=1.5, color="blue", alpha=0.4,
                    label=f"dPhiDr (N={len(sr)})")
            ax.hist(sr, bins=nbins, range=xrange, histtype="step",
                    linewidth=1.5, color="blue", alpha=0.9)
        if has_z:
            ax.hist(sz, bins=nbins, range=xrange, histtype="stepfilled",
                    linewidth=1.5, color="red", alpha=0.3,
                    label=f"dPhiDz (N={len(sz)})")
            ax.hist(sz, bins=nbins, range=xrange, histtype="step",
                    linewidth=1.5, color="red", alpha=0.9)

        ax.axvline(5.0, color="gray", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=r"5$\sigma$ cut")

        # Stats annotation
        stats_lines = [f"{label}"]
        if has_r:
            pct_r = np.sum(sr < 5.0) / len(sr) * 100
            stats_lines.append(
                f"dPhiDr: mean={np.mean(sr):.2f}, std={np.std(sr):.2f}\n"
                f"  {pct_r:.1f}% < 5sig")
        if has_z:
            pct_z = np.sum(sz < 5.0) / len(sz) * 100
            stats_lines.append(
                f"dPhiDz: mean={np.mean(sz):.2f}, std={np.std(sz):.2f}\n"
                f"  {pct_z:.1f}% < 5sig")
        if not has_r and not has_z:
            stats_lines.append("No pairs")
        ax.text(0.95, 0.95, "\n".join(stats_lines), transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        n_pairs = max(len(sr), len(sz))
        ax.set_title(f"{label}  (N={n_pairs} pairs)", fontsize=10)
        ax.set_xlabel(r"|$\Delta$val| / $\sigma_{comb}$", fontsize=9)
        ax.set_ylabel("Pairs", fontsize=9)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save outputs
    fig.savefig("pairwise_significance_by_transition.png", dpi=150, bbox_inches="tight")
    fig.savefig("pairwise_significance_by_transition.pdf", dpi=150, bbox_inches="tight")
    print("Saved pairwise_significance_by_transition.png")
    print("Saved pairwise_significance_by_transition.pdf")

    return fig


def make_all_plots(filename=DEFAULT_FILE, treename=DEFAULT_TREE, save_prefix=None):
    """
    Generate all analysis plots.

    Args:
        filename: Path to ROOT file
        treename: Name of TTree in file
        save_prefix: If provided, save plots with this prefix
    """
    print(f"Loading stubs from {filename}...")
    data = load_stubs(filename, treename)
    print(f"Loaded {len(data['stubIndex'])} stubs")

    plot_summary_stats(data)

    figs = []

    print("Creating dPhiDr comparison plot...")
    fig1 = plot_dPhiDr_comparison(data)
    figs.append(("dPhiDr_comparison", fig1))

    print("Creating dPhiDz comparison plot...")
    fig2 = plot_dPhiDz_comparison(data)
    figs.append(("dPhiDz_comparison", fig2))

    print("Creating eta consistency plot...")
    fig3 = plot_eta_consistency(data)
    figs.append(("eta_consistency", fig3))

    print("Creating bend and pT plot...")
    fig4 = plot_bend_and_pt(data)
    figs.append(("bend_pt", fig4))

    print("Creating geometry plot...")
    fig5 = plot_geometry(data)
    figs.append(("geometry", fig5))

    print("Creating sensor separation details plot...")
    fig6 = plot_sensor_separation_details(data)
    figs.append(("sensor_separation", fig6))

    print("Creating all distance definitions plot...")
    fig_distances = plot_all_distance_definitions(data)
    figs.append(("all_distance_definitions", fig_distances))

    print("Creating deltaPhi distributions plot...")
    fig_deltaphi = plot_deltaPhi_distributions(data)
    figs.append(("deltaPhi_distributions", fig_deltaphi))

    print("Creating per-layer distribution plots...")
    layer_figs = make_layer_plots(data)
    for name, fig in layer_figs.items():
        figs.append((f"layer_{name}", fig))

    print("Creating width distribution plots...")
    width_figs = make_width_plots(data)
    for name, fig in width_figs.items():
        figs.append((f"width_{name}", fig))

    print("Creating CA compatibility cut variables plot...")
    fig_compat = plot_compatibility_cuts(data)
    figs.append(("compatibility_cuts", fig_compat))

    print("Creating layer vs width scatter plots...")
    fig_lw = plot_layer_vs_width(data)
    figs.append(("layer_vs_width", fig_lw))

    print("Creating layer vs width_raw scatter plots...")
    fig_lwr = plot_layer_vs_width_raw(data)
    figs.append(("layer_vs_width_raw", fig_lwr))

    print("Creating layer vs width comparison plots...")
    fig_lwc = plot_layer_vs_width_comparison(data)
    figs.append(("layer_vs_width_comparison", fig_lwc))

    print("Creating XY angle plot...")
    fig_xy = plot_xy_angle(data)
    figs.append(("xy_angle", fig_xy))

    print("Creating consecutive layer dot product, dZ, and dR plots...")
    dotprod_result = plot_consecutive_layer_dotproduct(data)
    figs.append(("consecutive_layer_dotproduct", dotprod_result["figure_dotproduct"]))
    figs.append(("consecutive_layer_dZ", dotprod_result["figure_dz"]))
    figs.append(("consecutive_layer_dR", dotprod_result["figure_dr"]))

    print("Creating proposed compatibility quantities plot...")
    fig_proposed = plot_proposed_compatibility(data)
    figs.append(("proposed_compatibility", fig_proposed))

    print("Creating cuts per layer pair plots...")
    layer_pair_figs = plot_cuts_per_layer_pair(data)
    for name, fig in layer_pair_figs.items():
        figs.append((f"cuts_layer_pair_{name}", fig))

    print("Creating CA tuning variable plots...")
    ca_tuning_figs = plot_ca_tuning_variables(data)
    for name, fig in ca_tuning_figs.items():
        figs.append((f"ca_tuning_{name}", fig))

    print("Creating CA layer pair variable plots...")
    ca_layer_pair_figs = plot_ca_layer_pair_variables(data)
    for name, fig in ca_layer_pair_figs.items():
        figs.append((f"ca_layer_pair_{name}", fig))

    print("Creating kappa (curvature) comparison plots...")
    fig_kappa = plot_kappa_comparison(data)
    figs.append(("kappa_comparison", fig_kappa))

    print("Creating forward dPhiDz pairwise significance plots...")
    fig_fwd = plot_forward_dPhiDz_significance(data)
    figs.append(("forward_dPhiDz_significance", fig_fwd))

    print("Creating forward dPhiDz diagnostic plots...")
    fig_fwd_diag = plot_forward_dPhiDz_diagnostic(data)
    figs.append(("forward_dPhiDz_diagnostic", fig_fwd_diag))

    print("Creating pairwise significance by module transition type plots...")
    fig_trans = plot_pairwise_significance_by_transition(data)
    figs.append(("pairwise_significance_by_transition", fig_trans))

    if save_prefix:
        for name, fig in figs:
            fname = f"{save_prefix}_{name}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")

    plt.show()
    return data, figs


# Run if executed as script
if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    data, figs = make_all_plots(filename)
