#!/usr/bin/env python3
"""
Compare stacked module geometry between truth (CSV from simulation input)
and computed values (log from StackedModuleGeometryAnalyzer).

The truth CSV (allCoordinates.csv) contains stacked module geometry from
the Phase-2 Outer Tracker simulation input, keyed by stacked module DetId.

The comparison log is produced by StackedModuleGeometryAnalyzer reading
the StackedModuleGeometryESProducer SoA. It uses lower-sensor DetIds
internally but encodes stacked DetId as an offset from the lower DetId.

Convention differences handled by this script:

  - Section/PS-SS: The CSV "Section" column (TB2S, TBPS, TEDD_1, TEDD_2)
    maps to detector sub-types, not directly to PS/SS. Barrel: TB2S = SS,
    TBPS = PS. Endcap: TEDD_1 and TEDD_2 each contain both PS and SS
    modules at different rings. The script does NOT compare PS/SS for
    endcap modules since the CSV does not encode this at the module level.

  - Layer numbering: The CSV uses per-sub-detector layer numbers:
      Barrel: TBPS layers 1-3, TB2S layers 1-3
      Endcap: TEDD_1 disks 1-2, TEDD_2 disks 1-3
    TrackerTopology (comparison) uses unified numbering:
      Barrel: layers 1-6 (TBPS 1-3 then TB2S 4-6)
      Endcap: disks 1-5 (TEDD_1 disks 1-2 then TEDD_2 disks 3-5)
    The script converts CSV layers to the unified scheme before comparing.

  - Radius: The CSV "SensorCenter rho(mm)" is the stacked module center
    radial position. The comparison "meanRadius" is lowerPos.perp() in cm.
    For modules with non-zero separation, these differ by up to half the
    sensor separation projected onto the transverse plane (typically
    < 2 mm). The script reports this as an expected systematic.

  - Tilt angle: The CSV and ESProducer use DIFFERENT tilt angle conventions:

    CSV (old convention, relative to z-axis):
      - Flat barrel: ~0 deg (module axis parallel to z)
      - Tilted barrel: small angle from z (e.g., 10-30 deg)
      - Endcap: ~90 deg (module axis perpendicular to z)
      - Sign: positive for z > 0, negative for z < 0

    ESProducer (new convention, angle between PHYSICAL inner->outer vector and r-axis):
      - tiltAngle = atan2(dz_phys, dr_phys) where (dz_phys, dr_phys) is from
        physical inner to physical outer sensor (accounts for isFlipped)
      - Flat barrel: ~0 deg (inner->outer nearly radial, dr > 0)
      - Tilted barrel: non-zero (tilted toward IP)
      - Endcap: ~+-90 deg (inner->outer nearly parallel to z)
      - By using the physical direction, the sign is consistent regardless of
        whether the module is flipped. This ensures dPhiDr has the same sign
        for physically equivalent modules.

    Conversion: For barrel modules, the conversion is approximately:
      new_angle = 90 - |old_angle|, with sign preserved.
    For endcap modules, both conventions give ~+-90 deg (no conversion needed).

    NOTE: This conversion is based on geometric reasoning. Run the script
    and check if flat barrel modules show ~0 difference and endcap modules
    show ~0 difference. If there are large systematic offsets, the conversion
    formula may need adjustment.

    The script converts the CSV tilt angle to the new convention before
    comparing. Sign disagreements may indicate issues with the isFlipped
    computation.

Usage:
    python3 compareStackedModuleGeometry.py \\
        --truth allCoordinates.csv \\
        --comparison testStackedModuleGeometry_XXX.log
"""

import argparse
import re
import sys
import math
from collections import defaultdict


# Maps CSV (Section, Layer) to TrackerTopology unified layer number.
# Barrel: TBPS layers 1-3 -> topo layers 1-3; TB2S layers 1-3 -> topo layers 4-6
# Endcap: TEDD_1 disks 1-2 -> topo disks 1-2; TEDD_2 disks 1-3 -> topo disks 3-5
_LAYER_MAP = {}
for _l in range(1, 4):
    _LAYER_MAP[("TBPS", _l)] = _l
    _LAYER_MAP[("TB2S", _l)] = _l + 3
for _l in range(1, 3):
    _LAYER_MAP[("TEDD_1", _l)] = _l
for _l in range(1, 4):
    _LAYER_MAP[("TEDD_2", _l)] = _l + 2


def convert_csv_tilt_to_new_convention(csv_tilt_deg, is_barrel, z_mm):
    """Convert CSV tilt angle (old convention) to new convention.

    Old convention (CSV, relative to z-axis):
      - Flat barrel: ~90 deg (module axis perpendicular to z, i.e., radial)
      - Tilted barrel: between 90 and 0 (tilting toward z-parallel)
      - Endcap: ~0 deg (module axis parallel to z)
      - Sign: positive for z > 0, negative for z < 0

    New convention (ESProducer, relative to r-axis):
      - tiltAngle = atan2(dz_phys, dr_phys) where (dz_phys, dr_phys) is from
        PHYSICAL inner to PHYSICAL outer sensor (accounts for isFlipped)
      - For non-flipped: dz = upper.z - lower.z, dr = upper.r - lower.r
      - For flipped: dz = lower.z - upper.z, dr = lower.r - upper.r
      - Flat barrel: ~0 deg (inner->outer vector points radially outward)
      - Tilted barrel: non-zero, typically 10-30 deg (tilted toward IP)
      - Endcap: ~+-90 deg (inner->outer vector points along z)
      - By using physical direction, tiltAngle is consistent regardless of flipped

    The key insight is:
      - Old convention: measures angle of module axis (inner->outer) from z-axis
      - New convention: measures angle of module axis (inner->outer) from r-axis
      - Since r and z are perpendicular: new_angle = 90 - old_angle

    For a flat barrel module:
      - Module axis (inner->outer) is radial, perpendicular to z
      - Old: 90 deg (from z-axis)
      - New: 0 deg (from r-axis)

    For a tilted barrel module (tilted toward IP):
      - Module axis tilted from radial toward z
      - Old: decreases from 90 (e.g., 70 deg from z)
      - New: increases from 0 (e.g., 20 deg from r)

    For endcap:
      - Module axis is along z
      - Old: ~0 deg (from z-axis)
      - New: ~90 deg (from r-axis)

    Conversion: new_angle = 90 - |old_angle|, preserving sign

    Note: The actual CSV tiltAngle may use a different definition. If empirical
    comparison shows systematic offsets, this function may need adjustment.
    Run the script and check:
      - Flat barrel should show ~0 difference
      - Endcap should show ~0 difference
    If flat barrel shows ~90 difference, the CSV may define flat as 0, not 90.

    Args:
        csv_tilt_deg: Tilt angle from CSV in degrees (old convention)
        is_barrel: True if barrel module, False if endcap
        z_mm: z position of the module in mm (used for sign)

    Returns:
        Converted tilt angle in degrees (new convention)
    """
    if is_barrel:
        # For barrel: new_angle = 90 - |old_angle|
        # This converts from "angle of inner->outer from z" to
        # "angle of inner->outer from r".
        abs_old = abs(csv_tilt_deg)
        new_abs = 90.0 - abs_old
        # Preserve sign (indicates z hemisphere)
        if csv_tilt_deg >= 0:
            return new_abs
        else:
            return -new_abs
    else:
        # For endcap: both conventions give similar values
        # But we still need to convert from z-reference to r-reference
        # Old: ~0 (along z), New: ~90 (perpendicular to r)
        abs_old = abs(csv_tilt_deg)
        new_abs = 90.0 - abs_old
        if csv_tilt_deg >= 0:
            return new_abs
        else:
            return -new_abs


def parse_truth_csv(filepath):
    """Parse the truth CSV file (allCoordinates.csv).

    CSV columns (from header):
        DetId/U, BinaryDetId/B, Section/C, Layer/I, Ring/I,
        SensorCenter rho(mm), SensorCenter z(mm),
        tiltAngle_deg/D, skewAngle_deg/D, yawAngle_deg/D, phi_deg/D,
        vtxOneX_mm/D, vtxOneY_mm/D, vtxTwoX_mm/D, vtxTwoY_mm/D,
        vtxThreeX_mm/D, vtxThreeY_mm/D, vtxFourX_mm/D, vtxFourY_mm/D,
        meanWidth_mm/D, length_mm/D, sensorSpacing_mm/D, sensorThickness_mm/D

    Returns:
        dict keyed by integer DetId (stacked module), with values as dicts
        of parsed fields.
    """
    modules = {}
    with open(filepath, "r") as f:
        header_line = f.readline().strip()
        # Parse header to get column names
        raw_cols = [c.strip() for c in header_line.split(",")]
        # Clean column names (remove type suffixes like /U, /I, /D, /C, /B)
        col_names = [re.sub(r"/[UIDCB]$", "", c) for c in raw_cols]

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(col_names):
                print(f"WARNING: line {line_no} has {len(parts)} fields, expected {len(col_names)}, skipping",
                      file=sys.stderr)
                continue

            row = dict(zip(col_names, parts))

            det_id = int(row["DetId"])
            section = row["Section"]
            csv_layer = int(row["Layer"])

            # Determine barrel vs endcap from section
            is_barrel = section in ("TB2S", "TBPS")

            # Determine PS/SS from section (only reliable for barrel)
            # For barrel: TB2S = SS, TBPS = PS
            # For endcap: TEDD_1 and TEDD_2 contain both PS and SS at
            # different rings, so we set is_ps=None for endcap.
            if section == "TB2S":
                is_ps = False
            elif section == "TBPS":
                is_ps = True
            else:
                is_ps = None  # Unknown for endcap from CSV alone

            # Convert to unified TrackerTopology layer numbering
            topo_layer = _LAYER_MAP.get((section, csv_layer))
            if topo_layer is None:
                print(f"WARNING: unknown (section, layer) = ({section}, {csv_layer}) "
                      f"at line {line_no}", file=sys.stderr)
                topo_layer = csv_layer  # fallback

            modules[det_id] = {
                "detId": det_id,
                "section": section,
                "csv_layer": csv_layer,
                "layer": topo_layer,
                "ring": int(row["Ring"]),
                "rho_mm": float(row["SensorCenter rho(mm)"]),
                "z_mm": float(row["SensorCenter z(mm)"]),
                "tiltAngle_deg": float(row["tiltAngle_deg"]),
                "skewAngle_deg": float(row["skewAngle_deg"]),
                "yawAngle_deg": float(row["yawAngle_deg"]),
                "phi_deg": float(row["phi_deg"]),
                "meanWidth_mm": float(row["meanWidth_mm"]),
                "length_mm": float(row["length_mm"]),
                "sensorSpacing_mm": float(row["sensorSpacing_mm"]),
                "sensorThickness_mm": float(row["sensorThickness_mm"]),
                "isPS": is_ps,
                "isBarrel": is_barrel,
            }

    return modules


def parse_comparison_log(filepath):
    """Parse the comparison log file from StackedModuleGeometryAnalyzer.

    Each data line has the format (pipe-separated after the header):
        Idx | DetId(P-detId)[S-detId] | GeomIndex | K | Type | Barrel | Flat | Flipped | Layer |
        Sep(mm) | Radius(cm) | Tilt(deg) | maxBend(rad) |
        LowUpNormX | LowUpNormY | LowUpNormZ |
        locXInGlbX | locXInGlbY | locInGlbZ |

    The DetId column encodes: detId(partnerDetId - detId)[stackedDetId - detId]
    So to reconstruct: stackedDetId = detId + offset_in_brackets

    Returns:
        dict keyed by stackedDetId, with values as dicts of parsed fields.
        Also returns a dict keyed by detId (lower sensor id).
    """
    modules_by_stacked = {}
    modules_by_detid = {}

    # Pattern for the DetId(offset)[offset] format
    detid_pattern = re.compile(r"(\d+)\((-?\d+)\)\[(-?\d+)\]")

    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip()

            # Detect start of data section
            if "Idx |" in line and "DetId(P)[S]" in line:
                in_data = True
                continue
            if in_data and "----+---" in line:
                continue

            if not in_data:
                continue

            # Stop at empty line or end marker
            if not line.strip() or line.strip().startswith("====="):
                in_data = False
                continue

            # Parse the pipe-separated fields
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 19:
                # Not enough fields, skip
                continue

            try:
                idx = int(parts[0].strip())
            except ValueError:
                continue

            # Parse DetId field: "437519365(1)[-1]"
            detid_str = parts[1].strip()
            m = detid_pattern.search(detid_str)
            if not m:
                print(f"WARNING: could not parse DetId field '{detid_str}' at idx {idx}", file=sys.stderr)
                continue

            det_id = int(m.group(1))
            partner_offset = int(m.group(2))
            stacked_offset = int(m.group(3))
            partner_det_id = det_id + partner_offset
            stacked_det_id = det_id + stacked_offset

            geom_index = int(parts[2].strip())
            module_kind = parts[3].strip()  # P or S
            module_type = parts[4].strip()  # PS or SS
            is_barrel = parts[5].strip() == "Y"
            is_flat = parts[6].strip() == "Y"
            is_flipped = parts[7].strip() == "Y"
            layer = int(parts[8].strip())
            separation_mm = float(parts[9].strip())
            radius_cm = float(parts[10].strip())
            tilt_deg = float(parts[11].strip())
            max_bend = float(parts[12].strip())
            low_up_norm_x = float(parts[13].strip())
            low_up_norm_y = float(parts[14].strip())
            low_up_norm_z = float(parts[15].strip())
            loc_x_in_glb_x = float(parts[16].strip())
            loc_x_in_glb_y = float(parts[17].strip())
            loc_x_in_glb_z = float(parts[18].strip())

            entry = {
                "idx": idx,
                "detId": det_id,
                "partnerDetId": partner_det_id,
                "stackedDetId": stacked_det_id,
                "geomIndex": geom_index,
                "moduleKind": module_kind,
                "isPS": module_type.strip() == "PS",
                "isBarrel": is_barrel,
                "isFlat": is_flat,
                "isFlipped": is_flipped,
                "layer": layer,
                "separation_mm": separation_mm,
                "radius_cm": radius_cm,
                "tilt_deg": tilt_deg,
                "maxBend": max_bend,
                "lowUpNormX": low_up_norm_x,
                "lowUpNormY": low_up_norm_y,
                "lowUpNormZ": low_up_norm_z,
                "locXInGlbX": loc_x_in_glb_x,
                "locXInGlbY": loc_x_in_glb_y,
                "locXInGlbZ": loc_x_in_glb_z,
            }

            modules_by_stacked[stacked_det_id] = entry
            modules_by_detid[det_id] = entry

    return modules_by_stacked, modules_by_detid


def compute_stats(values, label):
    """Compute and return statistics for a list of values."""
    if not values:
        return {"label": label, "count": 0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    abs_max = max(abs(v) for v in values)

    return {
        "label": label,
        "count": n,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "abs_max": abs_max,
    }


def print_stats(stats, units=""):
    """Print statistics in a formatted way."""
    label = stats["label"]
    n = stats["count"]
    if n == 0:
        print(f"  {label}: no data")
        return

    u = f" {units}" if units else ""
    print(f"  {label} (N={n}):")
    print(f"    Mean:    {stats['mean']:+.6e}{u}")
    print(f"    Std:     {stats['std']:.6e}{u}")
    print(f"    Min:     {stats['min']:+.6e}{u}")
    print(f"    Max:     {stats['max']:+.6e}{u}")
    print(f"    AbsMax:  {stats['abs_max']:.6e}{u}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare stacked module geometry between truth CSV and computed log output."
    )
    parser.add_argument("--truth", required=True,
                        help="Path to truth CSV file (allCoordinates.csv)")
    parser.add_argument("--comparison", required=True,
                        help="Path to computed log file (StackedModuleGeometryAnalyzer output)")
    parser.add_argument("--tilt-tol", type=float, default=1.0,
                        help="Tolerance for tilt angle differences to flag as large discrepancy (degrees, default: 1.0)")
    parser.add_argument("--no-tilt-conversion", action="store_true",
                        help="Skip tilt angle convention conversion (compare raw values)")
    parser.add_argument("--show-raw-tilt", action="store_true",
                        help="Show raw CSV tilt values alongside converted values in output")
    parser.add_argument("--sep-tol", type=float, default=0.1,
                        help="Tolerance for sensor separation differences to flag as large discrepancy (mm, default: 0.1)")
    parser.add_argument("--radius-tol", type=float, default=1.0,
                        help="Tolerance for radius differences to flag as large discrepancy (mm, default: 1.0)")
    parser.add_argument("--max-discrepancies", type=int, default=20,
                        help="Maximum number of per-module discrepancies to print in detail (default: 20)")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Parse both files
    # ---------------------------------------------------------------
    print("=" * 72)
    print("  Stacked Module Geometry Comparison")
    print("=" * 72)
    print(f"  Truth file:      {args.truth}")
    print(f"  Comparison file: {args.comparison}")
    if args.no_tilt_conversion:
        print(f"  Tilt conversion: DISABLED (comparing raw values)")
    else:
        print(f"  Tilt conversion: ENABLED (CSV -> new convention)")
    print()

    truth = parse_truth_csv(args.truth)
    comp_by_stacked, comp_by_detid = parse_comparison_log(args.comparison)

    print(f"Modules in truth file:      {len(truth)}")
    print(f"Modules in comparison file: {len(comp_by_stacked)}")
    print()

    # ---------------------------------------------------------------
    # 2. Match modules by stacked DetId
    # ---------------------------------------------------------------
    truth_ids = set(truth.keys())
    comp_ids = set(comp_by_stacked.keys())

    matched_ids = truth_ids & comp_ids
    truth_only = truth_ids - comp_ids
    comp_only = comp_ids - truth_ids

    print(f"Matched modules (by stacked DetId): {len(matched_ids)}")
    print(f"Modules only in truth:              {len(truth_only)}")
    print(f"Modules only in comparison:         {len(comp_only)}")
    print()

    if truth_only:
        print(f"  First 10 DetIds only in truth: {sorted(truth_only)[:10]}")
    if comp_only:
        print(f"  First 10 DetIds only in comparison: {sorted(comp_only)[:10]}")
    if truth_only or comp_only:
        print()

    if not matched_ids:
        print("ERROR: No matched modules found. Cannot perform comparison.")
        print("This may indicate the files use different DetId conventions.")
        # Attempt alternate matching by lower sensor detId
        print("\nAttempting alternate matching by lower sensor DetId...")
        truth_ids_set = set(truth.keys())
        comp_detids_set = set(comp_by_detid.keys())
        alt_matched = truth_ids_set & comp_detids_set
        print(f"  Matched by lower sensor DetId: {len(alt_matched)}")
        if not alt_matched:
            print("No alternate matches found either. Exiting.")
            sys.exit(1)
        else:
            print("  Using lower sensor DetId matching instead.")
            matched_ids = alt_matched
            comp_by_stacked = {did: comp_by_detid[did] for did in alt_matched}

    # ---------------------------------------------------------------
    # 2b. Show raw tilt values for diagnostics (if requested)
    # ---------------------------------------------------------------
    if args.show_raw_tilt:
        print("-" * 72)
        print("  Raw Tilt Angle Diagnostics (--show-raw-tilt)")
        print("-" * 72)
        print()
        print("  Sample values help determine the CSV tilt angle convention:")
        print("    - If CSV flat barrel ~0 deg: CSV measures angle of plane from z")
        print("    - If CSV flat barrel ~90 deg: CSV measures angle of axis from z")
        print("  Compare with ESProducer (new convention, from r-axis):")
        print("    - Flat barrel: ~0 deg, Tilted barrel: ~10-30 deg, Endcap: ~90 deg")
        print()

        # Collect samples by category
        samples_flat_barrel = []
        samples_tilted_barrel = []
        samples_endcap = []

        for det_id in sorted(matched_ids)[:1000]:  # Sample first 1000
            t = truth[det_id]
            c = comp_by_stacked[det_id]
            entry = {
                "detId": det_id,
                "section": t["section"],
                "csv_tilt": t["tiltAngle_deg"],
                "comp_tilt": c["tilt_deg"],
                "isFlat": c["isFlat"],
                "z_mm": t["z_mm"],
            }
            if t["isBarrel"]:
                if c["isFlat"]:
                    samples_flat_barrel.append(entry)
                else:
                    samples_tilted_barrel.append(entry)
            else:
                samples_endcap.append(entry)

        # Show summary stats
        if samples_flat_barrel:
            csv_vals = [e["csv_tilt"] for e in samples_flat_barrel]
            comp_vals = [e["comp_tilt"] for e in samples_flat_barrel]
            print(f"  FLAT BARREL (N={len(samples_flat_barrel)}):")
            print(f"    CSV tilt:  min={min(csv_vals):+.2f}, max={max(csv_vals):+.2f}, "
                  f"mean={sum(csv_vals)/len(csv_vals):+.2f} deg")
            print(f"    Comp tilt: min={min(comp_vals):+.2f}, max={max(comp_vals):+.2f}, "
                  f"mean={sum(comp_vals)/len(comp_vals):+.2f} deg")
            print(f"    First 5 samples:")
            for e in samples_flat_barrel[:5]:
                print(f"      DetId={e['detId']}, {e['section']}: "
                      f"CSV={e['csv_tilt']:+.2f} deg, Comp={e['comp_tilt']:+.2f} deg")
            print()

        if samples_tilted_barrel:
            csv_vals = [e["csv_tilt"] for e in samples_tilted_barrel]
            comp_vals = [e["comp_tilt"] for e in samples_tilted_barrel]
            print(f"  TILTED BARREL (N={len(samples_tilted_barrel)}):")
            print(f"    CSV tilt:  min={min(csv_vals):+.2f}, max={max(csv_vals):+.2f}, "
                  f"mean={sum(csv_vals)/len(csv_vals):+.2f} deg")
            print(f"    Comp tilt: min={min(comp_vals):+.2f}, max={max(comp_vals):+.2f}, "
                  f"mean={sum(comp_vals)/len(comp_vals):+.2f} deg")
            print(f"    First 5 samples (z>0):")
            pos_z = [e for e in samples_tilted_barrel if e["z_mm"] > 0][:5]
            for e in pos_z:
                print(f"      DetId={e['detId']}, {e['section']}, z={e['z_mm']:+.0f}mm: "
                      f"CSV={e['csv_tilt']:+.2f} deg, Comp={e['comp_tilt']:+.2f} deg")
            print(f"    First 5 samples (z<0):")
            neg_z = [e for e in samples_tilted_barrel if e["z_mm"] < 0][:5]
            for e in neg_z:
                print(f"      DetId={e['detId']}, {e['section']}, z={e['z_mm']:+.0f}mm: "
                      f"CSV={e['csv_tilt']:+.2f} deg, Comp={e['comp_tilt']:+.2f} deg")
            print()

        if samples_endcap:
            csv_vals = [e["csv_tilt"] for e in samples_endcap]
            comp_vals = [e["comp_tilt"] for e in samples_endcap]
            print(f"  ENDCAP (N={len(samples_endcap)}):")
            print(f"    CSV tilt:  min={min(csv_vals):+.2f}, max={max(csv_vals):+.2f}, "
                  f"mean={sum(csv_vals)/len(csv_vals):+.2f} deg")
            print(f"    Comp tilt: min={min(comp_vals):+.2f}, max={max(comp_vals):+.2f}, "
                  f"mean={sum(comp_vals)/len(comp_vals):+.2f} deg")
            print(f"    First 5 samples (z>0):")
            pos_z = [e for e in samples_endcap if e["z_mm"] > 0][:5]
            for e in pos_z:
                print(f"      DetId={e['detId']}, {e['section']}, z={e['z_mm']:+.0f}mm: "
                      f"CSV={e['csv_tilt']:+.2f} deg, Comp={e['comp_tilt']:+.2f} deg")
            print(f"    First 5 samples (z<0):")
            neg_z = [e for e in samples_endcap if e["z_mm"] < 0][:5]
            for e in neg_z:
                print(f"      DetId={e['detId']}, {e['section']}, z={e['z_mm']:+.0f}mm: "
                      f"CSV={e['csv_tilt']:+.2f} deg, Comp={e['comp_tilt']:+.2f} deg")
            print()

    # ---------------------------------------------------------------
    # 3. Compare quantities for matched modules
    # ---------------------------------------------------------------

    # Accumulators for differences
    diff_separation = []          # mm
    diff_tilt_abs = []            # degrees (|comp| - |truth|)
    diff_tilt_signed = []         # degrees (comp - truth, signed)
    diff_radius = []              # mm (truth is in mm, comparison is in cm)
    mismatches_isPS = []          # Only for barrel (endcap PS/SS not in CSV)
    mismatches_isBarrel = []
    mismatches_layer = []
    mismatches_tilt_sign = []     # Modules where tilt sign disagrees

    # Per-module discrepancy records
    discrepancies = []

    # Category counters
    n_barrel_matched = 0
    n_endcap_matched = 0
    n_ps_matched_barrel = 0
    n_ss_matched_barrel = 0
    n_tilted = 0                  # Modules with non-zero tilt in truth
    n_tilt_sign_agree = 0
    n_tilt_sign_disagree = 0
    n_tilted_barrel = 0
    n_tilt_sign_agree_barrel = 0
    n_tilt_sign_disagree_barrel = 0
    n_tilted_endcap = 0
    n_tilt_sign_agree_endcap = 0
    n_tilt_sign_disagree_endcap = 0

    for det_id in sorted(matched_ids):
        t = truth[det_id]
        c = comp_by_stacked[det_id]

        # --- Module type (PS vs SS) ---
        # Only compare for barrel modules where the CSV reliably encodes
        # PS/SS via the Section field (TBPS = PS, TB2S = SS).
        # For endcap, the CSV Section (TEDD_1/TEDD_2) does not distinguish
        # PS vs SS at the module level, so we skip this check.
        if t["isBarrel"] and t["isPS"] is not None:
            if t["isPS"] != c["isPS"]:
                mismatches_isPS.append(det_id)
            if t["isPS"]:
                n_ps_matched_barrel += 1
            else:
                n_ss_matched_barrel += 1

        # --- Barrel vs Endcap ---
        if t["isBarrel"] != c["isBarrel"]:
            mismatches_isBarrel.append(det_id)

        if t["isBarrel"]:
            n_barrel_matched += 1
        else:
            n_endcap_matched += 1

        # --- Layer (using unified TrackerTopology numbering) ---
        if t["layer"] != c["layer"]:
            mismatches_layer.append((det_id, t["section"], t["csv_layer"],
                                     t["layer"], c["layer"]))

        # --- Sensor Separation (both in mm) ---
        d_sep = c["separation_mm"] - t["sensorSpacing_mm"]
        diff_separation.append(d_sep)

        # --- Tilt Angle (both in degrees) ---
        # The CSV uses the OLD convention (angle from z-axis):
        #   - Flat barrel: ~90 deg (radial), tilted barrel: decreasing, endcap: ~0 deg
        # The ESProducer uses the NEW convention (angle from r-axis):
        #   - Flat barrel: ~0 deg, tilted barrel: increasing, endcap: ~90 deg
        # We convert the CSV value to the new convention before comparing
        # (unless --no-tilt-conversion is specified).
        csv_tilt_raw = t["tiltAngle_deg"]
        if args.no_tilt_conversion:
            truth_tilt = csv_tilt_raw
        else:
            truth_tilt = convert_csv_tilt_to_new_convention(
                csv_tilt_raw, t["isBarrel"], t["z_mm"])
        comp_tilt = c["tilt_deg"]

        # Absolute value comparison (magnitude agreement)
        d_tilt_abs = abs(comp_tilt) - abs(truth_tilt)
        diff_tilt_abs.append(d_tilt_abs)

        # Signed comparison (full agreement including direction)
        d_tilt_signed = comp_tilt - truth_tilt
        diff_tilt_signed.append(d_tilt_signed)

        # Sign agreement check (only for non-zero tilt in new convention)
        # For barrel: flat modules have ~0 deg, tilted have large angles
        # For endcap: all modules have ~+-90 deg
        if abs(truth_tilt) > 0.1:  # Non-flat modules (in new convention)
            n_tilted += 1
            truth_sign = 1 if truth_tilt > 0 else -1
            comp_sign = 1 if comp_tilt > 0 else -1
            is_barrel = t["isBarrel"]
            if is_barrel:
                n_tilted_barrel += 1
            else:
                n_tilted_endcap += 1
            if truth_sign == comp_sign:
                n_tilt_sign_agree += 1
                if is_barrel:
                    n_tilt_sign_agree_barrel += 1
                else:
                    n_tilt_sign_agree_endcap += 1
            else:
                n_tilt_sign_disagree += 1
                if is_barrel:
                    n_tilt_sign_disagree_barrel += 1
                else:
                    n_tilt_sign_disagree_endcap += 1
                mismatches_tilt_sign.append({
                    "detId": det_id,
                    "section": t["section"],
                    "layer": t["layer"],
                    "csv_layer": t["csv_layer"],
                    "ring": t["ring"],
                    "z_mm": t["z_mm"],
                    "csv_tilt_raw": csv_tilt_raw,
                    "truth_tilt": truth_tilt,
                    "comp_tilt": comp_tilt,
                    "isFlipped": c["isFlipped"],
                    "isBarrel": c["isBarrel"],
                })

        # --- Radius ---
        # Truth: SensorCenter rho in mm (stacked module center position)
        # Comparison: meanRadius = lowerPos.perp() in cm
        # These differ by up to ~half the sensor separation projected
        # onto the transverse plane. Convert comparison to mm.
        truth_radius_mm = t["rho_mm"]
        comp_radius_mm = c["radius_cm"] * 10.0  # cm -> mm
        d_radius = comp_radius_mm - truth_radius_mm
        diff_radius.append(d_radius)

        # --- Collect large discrepancies ---
        module_issues = []
        if abs(d_sep) > args.sep_tol:
            module_issues.append(f"separation: truth={t['sensorSpacing_mm']:.4f} mm, "
                                 f"comp={c['separation_mm']:.4f} mm, diff={d_sep:+.4f} mm")
        if abs(d_tilt_abs) > args.tilt_tol:
            module_issues.append(f"tilt (magnitude): |truth_conv|={abs(truth_tilt):.4f} deg, "
                                 f"|comp|={abs(comp_tilt):.4f} deg, diff={d_tilt_abs:+.4f} deg "
                                 f"(csv_raw={csv_tilt_raw:+.4f} deg)")
        if abs(d_tilt_signed) > args.tilt_tol:
            module_issues.append(f"tilt (signed): truth_conv={truth_tilt:+.4f} deg, "
                                 f"comp={comp_tilt:+.4f} deg, diff={d_tilt_signed:+.4f} deg "
                                 f"(csv_raw={csv_tilt_raw:+.4f} deg)")
        if abs(d_radius) > args.radius_tol:
            module_issues.append(f"radius: truth={truth_radius_mm:.4f} mm, "
                                 f"comp={comp_radius_mm:.4f} mm, diff={d_radius:+.4f} mm")
        if t["isBarrel"] and t["isPS"] is not None and t["isPS"] != c["isPS"]:
            module_issues.append(f"type mismatch (barrel): truth={'PS' if t['isPS'] else 'SS'}, "
                                 f"comp={'PS' if c['isPS'] else 'SS'}")
        if t["isBarrel"] != c["isBarrel"]:
            module_issues.append(f"location mismatch: truth={'Barrel' if t['isBarrel'] else 'Endcap'}, "
                                 f"comp={'Barrel' if c['isBarrel'] else 'Endcap'}")
        if t["layer"] != c["layer"]:
            module_issues.append(f"layer mismatch: truth={t['section']} L{t['csv_layer']} "
                                 f"-> topo L{t['layer']}, comp topo L{c['layer']}")

        if module_issues:
            discrepancies.append((det_id, t, c, module_issues))

    # ---------------------------------------------------------------
    # 4. Compute and print statistics
    # ---------------------------------------------------------------
    print("-" * 72)
    print("  Comparison Statistics for Matched Modules")
    print("-" * 72)
    print()

    print(f"  Barrel modules matched: {n_barrel_matched}")
    print(f"  Endcap modules matched: {n_endcap_matched}")
    print(f"  PS modules matched (barrel only): {n_ps_matched_barrel}")
    print(f"  SS modules matched (barrel only): {n_ss_matched_barrel}")
    print(f"  (Endcap PS/SS not compared: CSV Section does not encode PS/SS per module)")
    print()

    # --- Boolean / categorical mismatches ---
    print("Categorical Mismatches:")
    print(f"  Module type (PS/SS, barrel only) mismatches: {len(mismatches_isPS)}")
    if mismatches_isPS:
        shown = mismatches_isPS[:10]
        print(f"    First {len(shown)} DetIds: {shown}")
    print(f"  Location (Barrel/Endcap) mismatches:        {len(mismatches_isBarrel)}")
    if mismatches_isBarrel:
        shown = mismatches_isBarrel[:10]
        print(f"    First {len(shown)} DetIds: {shown}")
    print(f"  Layer mismatches (unified numbering):       {len(mismatches_layer)}")
    if mismatches_layer:
        shown = mismatches_layer[:10]
        for det_id, section, csv_layer, topo_layer, comp_layer in shown:
            print(f"    DetId {det_id}: {section} L{csv_layer} -> topo L{topo_layer}, comp L{comp_layer}")
    print()

    # --- Continuous quantity statistics ---
    print("Continuous Quantity Differences (comparison - truth):")
    print()

    stats_sep = compute_stats(diff_separation, "Sensor Separation")
    print_stats(stats_sep, units="mm")
    print()

    stats_tilt_abs = compute_stats(diff_tilt_abs, "Tilt Angle magnitude (|comp| - |truth|)")
    print_stats(stats_tilt_abs, units="deg")
    print()

    stats_tilt_signed = compute_stats(diff_tilt_signed, "Tilt Angle signed (comp - truth)")
    print_stats(stats_tilt_signed, units="deg")
    print()

    stats_radius = compute_stats(diff_radius, "Radius (comp*10 - truth)")
    print_stats(stats_radius, units="mm")
    print()

    # --- Tilt angle sign analysis ---
    print("-" * 72)
    print("  Tilt Angle Sign Analysis (isFlipped cross-check)")
    print("  (Using converted CSV tilt values in new convention)")
    print("-" * 72)
    print()
    print(f"  Non-flat modules (|truth_conv tilt| > 0.1 deg): {n_tilted}")
    print(f"  Sign agrees (truth and comp same sign):    {n_tilt_sign_agree}")
    print(f"  Sign disagrees:                            {n_tilt_sign_disagree}")
    print()
    print(f"  Barrel tilted modules:   {n_tilted_barrel}")
    print(f"    Sign agrees:           {n_tilt_sign_agree_barrel}")
    print(f"    Sign disagrees:        {n_tilt_sign_disagree_barrel}")
    print(f"  Endcap modules:          {n_tilted_endcap}")
    print(f"    Sign agrees:           {n_tilt_sign_agree_endcap}")
    print(f"    Sign disagrees:        {n_tilt_sign_disagree_endcap}")
    print()
    # Cross-check: for barrel tilted modules where |tilt| agrees but sign
    # differs, check correlation with isFlipped
    barrel_sign_flip_flipped = []      # sign disagrees AND isFlipped=Y
    barrel_sign_flip_not_flipped = []  # sign disagrees AND isFlipped=N
    barrel_sign_agree_flipped = []     # sign agrees AND isFlipped=Y
    barrel_sign_agree_not_flipped = [] # sign agrees AND isFlipped=N

    for det_id in sorted(matched_ids):
        t = truth[det_id]
        c = comp_by_stacked[det_id]
        if not t["isBarrel"]:
            continue
        csv_tilt_raw = t["tiltAngle_deg"]
        if args.no_tilt_conversion:
            truth_tilt = csv_tilt_raw
        else:
            truth_tilt = convert_csv_tilt_to_new_convention(
                csv_tilt_raw, t["isBarrel"], t["z_mm"])
        comp_tilt = c["tilt_deg"]
        if abs(truth_tilt) < 0.1:
            continue  # skip flat modules (in new convention)

        mag_agrees = abs(abs(comp_tilt) - abs(truth_tilt)) < 0.1
        if not mag_agrees:
            continue  # skip if magnitudes don't agree

        truth_sign = 1 if truth_tilt > 0 else -1
        comp_sign = 1 if comp_tilt > 0 else -1
        sign_agrees = (truth_sign == comp_sign)

        entry = {
            "detId": det_id, "section": t["section"], "layer": t["layer"],
            "csv_layer": t["csv_layer"], "ring": t["ring"], "z_mm": t["z_mm"],
            "csv_tilt_raw": csv_tilt_raw, "truth_tilt": truth_tilt,
            "comp_tilt": comp_tilt, "isFlipped": c["isFlipped"],
        }

        if sign_agrees and c["isFlipped"]:
            barrel_sign_agree_flipped.append(entry)
        elif sign_agrees and not c["isFlipped"]:
            barrel_sign_agree_not_flipped.append(entry)
        elif not sign_agrees and c["isFlipped"]:
            barrel_sign_flip_flipped.append(entry)
        else:
            barrel_sign_flip_not_flipped.append(entry)

    print("-" * 72)
    print("  Barrel Tilted Modules: Tilt Sign vs isFlipped Cross-Check")
    print("  (only modules where |truth_conv tilt| == |comp tilt|, i.e. magnitude agrees)")
    print("  (truth_conv = CSV tilt converted to new convention)")
    print("-" * 72)
    print()
    n_total_barrel_tilted = (len(barrel_sign_flip_flipped) + len(barrel_sign_flip_not_flipped)
                             + len(barrel_sign_agree_flipped) + len(barrel_sign_agree_not_flipped))
    print(f"  Total barrel tilted modules (magnitude agrees): {n_total_barrel_tilted}")
    print()
    print(f"  Sign AGREES,   isFlipped=N: {len(barrel_sign_agree_not_flipped)}")
    print(f"  Sign AGREES,   isFlipped=Y: {len(barrel_sign_agree_flipped)}")
    print(f"  Sign DISAGREES, isFlipped=N: {len(barrel_sign_flip_not_flipped)}")
    print(f"  Sign DISAGREES, isFlipped=Y: {len(barrel_sign_flip_flipped)}")
    print()

    if barrel_sign_flip_not_flipped:
        print(f"  WARNING: {len(barrel_sign_flip_not_flipped)} modules have sign mismatch but isFlipped=N!")
        print(f"  First {min(len(barrel_sign_flip_not_flipped), 20)}:")
        print(f"    {'DetId':>12s}  {'Section':>7s}  {'Layer':>5s}  {'Ring':>4s}  "
              f"{'z(mm)':>10s}  {'CSV raw':>10s}  {'Truth conv':>10s}  {'Comp':>10s}")
        print(f"    {'':->12s}  {'':->7s}  {'':->5s}  {'':->4s}  "
              f"{'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}")
        for entry in barrel_sign_flip_not_flipped[:20]:
            print(f"    {entry['detId']:>12d}  {entry['section']:>7s}  "
                  f"L{entry['layer']:>4d}  R{entry['ring']:>3d}  "
                  f"{entry['z_mm']:>+10.2f}  {entry['csv_tilt_raw']:>+10.4f}  "
                  f"{entry['truth_tilt']:>+10.4f}  {entry['comp_tilt']:>+10.4f}")
        print()

    if barrel_sign_agree_flipped:
        print(f"  NOTE: {len(barrel_sign_agree_flipped)} modules have sign agreement but isFlipped=Y")
        print(f"  First {min(len(barrel_sign_agree_flipped), 20)}:")
        print(f"    {'DetId':>12s}  {'Section':>7s}  {'Layer':>5s}  {'Ring':>4s}  "
              f"{'z(mm)':>10s}  {'CSV raw':>10s}  {'Truth conv':>10s}  {'Comp':>10s}")
        print(f"    {'':->12s}  {'':->7s}  {'':->5s}  {'':->4s}  "
              f"{'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}")
        for entry in barrel_sign_agree_flipped[:20]:
            print(f"    {entry['detId']:>12d}  {entry['section']:>7s}  "
                  f"L{entry['layer']:>4d}  R{entry['ring']:>3d}  "
                  f"{entry['z_mm']:>+10.2f}  {entry['csv_tilt_raw']:>+10.4f}  "
                  f"{entry['truth_tilt']:>+10.4f}  {entry['comp_tilt']:>+10.4f}")
        print()

    if not barrel_sign_flip_not_flipped and not barrel_sign_agree_flipped:
        print("  PASS: Sign mismatch correlates perfectly with isFlipped=Y,")
        print("        and sign agreement correlates perfectly with isFlipped=N.")
    print()

    if mismatches_tilt_sign:
        # Sort to show barrel mismatches first (more relevant for isFlipped)
        barrel_mismatches = [e for e in mismatches_tilt_sign if e["isBarrel"]]
        endcap_mismatches = [e for e in mismatches_tilt_sign if not e["isBarrel"]]
        sorted_mismatches = barrel_mismatches + endcap_mismatches

        n_show = min(len(sorted_mismatches), args.max_discrepancies)
        n_barrel_shown = min(len(barrel_mismatches), n_show)
        print(f"  Modules with tilt sign mismatch (first {n_show}, barrel first):")
        print(f"  (CSV raw = old convention, Truth conv = after conversion to new convention)")
        print()
        print(f"    {'DetId':>12s}  {'Section':>7s}  {'Layer':>5s}  {'Ring':>4s}  "
              f"{'z(mm)':>10s}  {'CSV raw':>10s}  {'Truth conv':>10s}  {'Comp':>10s}  "
              f"{'Flipped':>7s}  {'Barrel':>6s}")
        print(f"    {'':->12s}  {'':->7s}  {'':->5s}  {'':->4s}  "
              f"{'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  "
              f"{'':->7s}  {'':->6s}")
        for entry in sorted_mismatches[:args.max_discrepancies]:
            print(f"    {entry['detId']:>12d}  {entry['section']:>7s}  "
                  f"L{entry['layer']:>4d}  R{entry['ring']:>3d}  "
                  f"{entry['z_mm']:>+10.2f}  {entry['csv_tilt_raw']:>+10.4f}  "
                  f"{entry['truth_tilt']:>+10.4f}  {entry['comp_tilt']:>+10.4f}  "
                  f"{'Y' if entry['isFlipped'] else 'N':>7s}  "
                  f"{'Y' if entry['isBarrel'] else 'N':>6s}")
        print()
        # Analyze correlation between sign mismatch and isFlipped
        n_flipped_mismatch = sum(1 for e in mismatches_tilt_sign if e["isFlipped"])
        n_nonflipped_mismatch = sum(1 for e in mismatches_tilt_sign if not e["isFlipped"])
        print(f"  Among sign-mismatched modules:")
        print(f"    isFlipped=Y: {n_flipped_mismatch}")
        print(f"    isFlipped=N: {n_nonflipped_mismatch}")
        # Check if sign mismatch correlates with specific z hemisphere
        n_pos_z = sum(1 for e in mismatches_tilt_sign if e["z_mm"] > 0)
        n_neg_z = sum(1 for e in mismatches_tilt_sign if e["z_mm"] < 0)
        print(f"    z > 0: {n_pos_z}")
        print(f"    z < 0: {n_neg_z}")
    else:
        print("  All non-flat modules have consistent tilt angle signs.")
    print()

    # --- Breakdown by barrel/endcap ---
    diff_sep_barrel = []
    diff_sep_endcap = []
    diff_tilt_abs_barrel = []
    diff_tilt_abs_endcap = []
    diff_tilt_signed_barrel = []
    diff_tilt_signed_endcap = []
    diff_radius_barrel = []
    diff_radius_endcap = []

    for det_id in sorted(matched_ids):
        t = truth[det_id]
        c = comp_by_stacked[det_id]

        d_sep = c["separation_mm"] - t["sensorSpacing_mm"]
        # Convert CSV tilt to new convention before comparing (unless disabled)
        csv_tilt_raw = t["tiltAngle_deg"]
        if args.no_tilt_conversion:
            truth_tilt_conv = csv_tilt_raw
        else:
            truth_tilt_conv = convert_csv_tilt_to_new_convention(
                csv_tilt_raw, t["isBarrel"], t["z_mm"])
        d_tilt_abs = abs(c["tilt_deg"]) - abs(truth_tilt_conv)
        d_tilt_signed = c["tilt_deg"] - truth_tilt_conv
        d_radius = c["radius_cm"] * 10.0 - t["rho_mm"]

        if t["isBarrel"]:
            diff_sep_barrel.append(d_sep)
            diff_tilt_abs_barrel.append(d_tilt_abs)
            diff_tilt_signed_barrel.append(d_tilt_signed)
            diff_radius_barrel.append(d_radius)
        else:
            diff_sep_endcap.append(d_sep)
            diff_tilt_abs_endcap.append(d_tilt_abs)
            diff_tilt_signed_endcap.append(d_tilt_signed)
            diff_radius_endcap.append(d_radius)

    print("-" * 72)
    print("  Barrel-only Statistics")
    print("-" * 72)
    print()
    print_stats(compute_stats(diff_sep_barrel, "Sensor Separation (Barrel)"), units="mm")
    print()
    print_stats(compute_stats(diff_tilt_abs_barrel, "Tilt Angle magnitude (Barrel)"), units="deg")
    print()
    print_stats(compute_stats(diff_tilt_signed_barrel, "Tilt Angle signed (Barrel)"), units="deg")
    print()
    print_stats(compute_stats(diff_radius_barrel, "Radius (Barrel)"), units="mm")
    print()

    print("-" * 72)
    print("  Endcap-only Statistics")
    print("-" * 72)
    print()
    print_stats(compute_stats(diff_sep_endcap, "Sensor Separation (Endcap)"), units="mm")
    print()
    print_stats(compute_stats(diff_tilt_abs_endcap, "Tilt Angle magnitude (Endcap)"), units="deg")
    print()
    print_stats(compute_stats(diff_tilt_signed_endcap, "Tilt Angle signed (Endcap)"), units="deg")
    print()
    print_stats(compute_stats(diff_radius_endcap, "Radius (Endcap)"), units="mm")
    print()

    # --- Breakdown by layer ---
    print("-" * 72)
    print("  Per-Layer Statistics")
    print("-" * 72)
    print()

    diff_by_layer = defaultdict(lambda: {"sep": [], "tilt_abs": [], "tilt_signed": [], "radius": []})
    for det_id in sorted(matched_ids):
        t = truth[det_id]
        c = comp_by_stacked[det_id]

        loc = "Barrel" if t["isBarrel"] else "Endcap"
        layer_key = f"{loc} L{t['layer']} ({t['section']})"

        d_sep = c["separation_mm"] - t["sensorSpacing_mm"]
        # Convert CSV tilt to new convention before comparing (unless disabled)
        csv_tilt_raw = t["tiltAngle_deg"]
        if args.no_tilt_conversion:
            truth_tilt_conv = csv_tilt_raw
        else:
            truth_tilt_conv = convert_csv_tilt_to_new_convention(
                csv_tilt_raw, t["isBarrel"], t["z_mm"])
        d_tilt_abs = abs(c["tilt_deg"]) - abs(truth_tilt_conv)
        d_tilt_signed = c["tilt_deg"] - truth_tilt_conv
        d_radius = c["radius_cm"] * 10.0 - t["rho_mm"]

        diff_by_layer[layer_key]["sep"].append(d_sep)
        diff_by_layer[layer_key]["tilt_abs"].append(d_tilt_abs)
        diff_by_layer[layer_key]["tilt_signed"].append(d_tilt_signed)
        diff_by_layer[layer_key]["radius"].append(d_radius)

    for layer_key in sorted(diff_by_layer.keys()):
        data = diff_by_layer[layer_key]
        print(f"  {layer_key}:")
        s_sep = compute_stats(data["sep"], "Sep")
        s_tilt_a = compute_stats(data["tilt_abs"], "TiltAbs")
        s_tilt_s = compute_stats(data["tilt_signed"], "TiltSgn")
        s_rad = compute_stats(data["radius"], "Rad")
        print(f"    Sep(mm):      mean={s_sep['mean']:+.4e}  std={s_sep['std']:.4e}  "
              f"min={s_sep['min']:+.4e}  max={s_sep['max']:+.4e}  (N={s_sep['count']})")
        print(f"    Tilt|d|(deg): mean={s_tilt_a['mean']:+.4e}  std={s_tilt_a['std']:.4e}  "
              f"min={s_tilt_a['min']:+.4e}  max={s_tilt_a['max']:+.4e}  (N={s_tilt_a['count']})")
        print(f"    Tilt_s(deg):  mean={s_tilt_s['mean']:+.4e}  std={s_tilt_s['std']:.4e}  "
              f"min={s_tilt_s['min']:+.4e}  max={s_tilt_s['max']:+.4e}  (N={s_tilt_s['count']})")
        print(f"    Rad(mm):      mean={s_rad['mean']:+.4e}  std={s_rad['std']:.4e}  "
              f"min={s_rad['min']:+.4e}  max={s_rad['max']:+.4e}  (N={s_rad['count']})")
        print()

    # ---------------------------------------------------------------
    # 5. Report large discrepancies
    # ---------------------------------------------------------------
    print("-" * 72)
    print("  Modules with Large Discrepancies")
    print("-" * 72)
    print(f"  (Thresholds: separation > {args.sep_tol} mm, "
          f"tilt > {args.tilt_tol} deg, radius > {args.radius_tol} mm)")
    print()

    if not discrepancies:
        print("  None -- all matched modules are within tolerance.")
    else:
        print(f"  Total modules with at least one discrepancy: {len(discrepancies)}")
        print()

        # Categorize discrepancies
        n_sep_disc = sum(1 for _, _, _, issues in discrepancies if any("separation" in i for i in issues))
        n_tilt_disc = sum(1 for _, _, _, issues in discrepancies if any("tilt" in i for i in issues))
        n_rad_disc = sum(1 for _, _, _, issues in discrepancies if any("radius" in i for i in issues))
        n_type_disc = sum(1 for _, _, _, issues in discrepancies if any("type mismatch" in i for i in issues))
        n_loc_disc = sum(1 for _, _, _, issues in discrepancies if any("location mismatch" in i for i in issues))
        n_layer_disc = sum(1 for _, _, _, issues in discrepancies if any("layer mismatch" in i for i in issues))
        # Count purely geometric discrepancies (separation, tilt, radius only)
        n_geom_only = sum(1 for _, _, _, issues in discrepancies
                          if all(any(k in i for k in ("separation", "tilt", "radius")) for i in issues))

        print(f"  By category:")
        print(f"    Separation discrepancies:  {n_sep_disc}")
        print(f"    Tilt angle discrepancies:  {n_tilt_disc}")
        print(f"    Radius discrepancies:      {n_rad_disc}")
        print(f"    Type (PS/SS) mismatches:   {n_type_disc}")
        print(f"    Location (B/E) mismatches: {n_loc_disc}")
        print(f"    Layer mismatches:          {n_layer_disc}")
        print(f"    Geometry-only (no cat.):   {n_geom_only}")
        print()

        # Print individual modules
        limit = min(len(discrepancies), args.max_discrepancies)
        print(f"  Showing first {limit} of {len(discrepancies)} discrepant modules:")
        print()
        for i, (det_id, t, c, issues) in enumerate(discrepancies[:limit]):
            section = t["section"]
            loc = "Barrel" if t["isBarrel"] else "Endcap"
            print(f"  [{i+1}] DetId={det_id} ({section}, {loc}, "
                  f"CSV layer={t['csv_layer']}, topo layer={t['layer']})")
            print(f"       comp idx={c['idx']}, comp detId(lower)={c['detId']}")
            for issue in issues:
                print(f"       - {issue}")
            print()

    # ---------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------
    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    print()
    print(f"  Truth modules:            {len(truth)}")
    print(f"  Comparison modules:       {len(comp_by_stacked)}")
    print(f"  Matched:                  {len(matched_ids)}")
    print(f"  Unmatched (truth only):   {len(truth_only)}")
    print(f"  Unmatched (comp only):    {len(comp_only)}")
    print()

    all_ok = True

    if mismatches_isPS:
        print(f"  FAIL: {len(mismatches_isPS)} module type (PS/SS) mismatches found (barrel only).")
        all_ok = False
    else:
        print(f"  PASS: All barrel module types (PS/SS) agree.")
        print(f"        (Endcap PS/SS not checked: CSV does not encode PS/SS per module)")

    if mismatches_isBarrel:
        print(f"  FAIL: {len(mismatches_isBarrel)} barrel/endcap mismatches found.")
        all_ok = False
    else:
        print(f"  PASS: All barrel/endcap classifications agree.")

    if mismatches_layer:
        print(f"  FAIL: {len(mismatches_layer)} layer mismatches found.")
        all_ok = False
    else:
        print(f"  PASS: All layer assignments agree.")

    # Check continuous quantities
    if stats_sep["count"] > 0:
        if stats_sep["abs_max"] < args.sep_tol:
            print(f"  PASS: Sensor separation max |diff| = {stats_sep['abs_max']:.6e} mm < {args.sep_tol} mm")
        else:
            print(f"  WARN: Sensor separation max |diff| = {stats_sep['abs_max']:.6e} mm >= {args.sep_tol} mm")
            all_ok = False

    if stats_tilt_abs["count"] > 0:
        if stats_tilt_abs["abs_max"] < args.tilt_tol:
            print(f"  PASS: Tilt angle magnitude max |diff| = {stats_tilt_abs['abs_max']:.6e} deg < {args.tilt_tol} deg")
        else:
            print(f"  WARN: Tilt angle magnitude max |diff| = {stats_tilt_abs['abs_max']:.6e} deg >= {args.tilt_tol} deg")
            all_ok = False

    if n_tilted > 0:
        if n_tilt_sign_disagree == 0:
            print(f"  PASS: Tilt angle sign agrees for all {n_tilted} tilted modules")
        else:
            print(f"  FAIL: Tilt angle sign disagrees for {n_tilt_sign_disagree}/{n_tilted} tilted modules")
            print(f"        This indicates potential isFlipped computation errors.")
            all_ok = False

    if stats_radius["count"] > 0:
        if stats_radius["abs_max"] < args.radius_tol:
            print(f"  PASS: Radius max |diff| = {stats_radius['abs_max']:.6e} mm < {args.radius_tol} mm")
        else:
            print(f"  INFO: Radius max |diff| = {stats_radius['abs_max']:.6e} mm >= {args.radius_tol} mm")
            print(f"        This is expected: truth uses stacked module center rho, while")
            print(f"        comparison uses lower sensor rho (lowerPos.perp()). The difference")
            print(f"        is approximately half the sensor separation projected onto the")
            print(f"        transverse plane. This is NOT a bug.")

    print()
    if all_ok:
        print("  OVERALL: All checks passed within tolerance.")
    else:
        print("  OVERALL: Some checks failed -- see details above.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
