#!/usr/bin/env python3
"""
Parse a CA (Cellular Automaton) tracking log file and produce a clean summary.

This script reads the verbose debug output from the Phase-2 CA tracking
(CAHitNtupletGenerator) and summarizes it event-by-event, showing:
  - Doublets created per layer pair
  - Triplet/cell formation
  - Fishbone cleaning kills
  - Track (n-tuplet) information
  - Overall tracking pipeline summary

Usage:
    python parseCATrackingLog.py <logfile> [options]

Examples:
    python parseCATrackingLog.py tracking.log
    python parseCATrackingLog.py tracking.log --event 1
    python parseCATrackingLog.py tracking.log --event 1 --verbose
    python parseCATrackingLog.py tracking.log --suggest-cuts
    python parseCATrackingLog.py tracking.log --suggest-cuts --config-file path/to/config.py
    python parseCATrackingLog.py tracking.log --suggest-cuts --safety-margin 0.15
    python parseCATrackingLog.py tracking.log --event 1 --suggest-cuts --verbose
    python parseCATrackingLog.py tracking.log --summary-only
"""

import argparse
import re
import sys
from collections import defaultdict, OrderedDict


# ---------------------------------------------------------------------------
# Regular expressions for the various log line formats
# ---------------------------------------------------------------------------

RE_BEGIN_EVENT = re.compile(
    r"Begin processing the (\d+)\w+ record\. Run (\d+), Event (\d+), LumiSection (\d+)"
)

RE_LAYER_START_PIXEL = re.compile(
    r"Pixel LayerStart: CA layer (\d+) at subdetector layer (\d+) starts at module (\d+) and is (barrel|not barrel)"
)

RE_LAYER_START_OT = re.compile(
    r"OT LayerStart: CA layer (\d+) starts at module (\d+) \((barrel|endcap) layer (\d+)\)"
)

RE_BUILDING_DOUBLETS = re.compile(
    r"building Doublets out of (\d+) Hits"
)

RE_MAX_NUM_DOUBLETS = re.compile(
    r"maxNumDoublets\s*=\s*(\d+)\s+cc\.metadata\(\)\.size\(\)\s*=\s*(\d+)\s+"
    r"ll\.metadata\(\)\.size\(\)\s*=\s*(\d+)\s+"
    r"cellZ0Cut_\s*=\s*([\d.]+)\s+ptmin_\s*=\s*([\d.]+)"
)

RE_DOUBLET = re.compile(
    r"doublet:\s*(\d+)\s+layerPair:\s*(\d+)\s+inner:\s*(\d+)\s+outer:\s*(\d+)\s+i:\s*(\d+)\s+oi:\s*(\d+)"
)

RE_PAIR_LAYER_ID = re.compile(
    r"pairLayerId\s*=\s*(\d+)\s+i\s*=\s*(\d+)\s+inner\s*=\s*(\d+)\s+outer\s*=\s*(\d+)"
)

RE_KILLED_1 = re.compile(r"Killed here 1\b")

RE_KILLED_2 = re.compile(
    r"Killed here 2 --> valInner:\s*([-\d.]+)\s+\[index:\s*(\d+)\],\s*"
    r"minInner:\s*([-\d.]+),\s*maxInner:\s*([-\d.]+)"
)

RE_KILLED_4_SIMPLE = re.compile(r"Killed here 4$")
RE_KILLED_4_DETAIL = re.compile(
    r"Killed here 4 --> mo:\s*(\d+)\s*>=\s*numberOfModules:\s*(\d+)"
)

RE_KILLED_5_SIMPLE = re.compile(r"Killed here 5$")
RE_KILLED_5_VAL_OUTER = re.compile(
    r"Killed here 5 --> valOuter:\s*([-\d.]+)\s+\[index:\s*(\d+)\],\s*"
    r"minOuter:\s*([-\d.]+),\s*maxOuter:\s*([-\d.]+)"
)
RE_KILLED_5_DZ = re.compile(
    r"Killed here 5 --> dz:\s*([-\d.]+)\s+\[index:\s*(\d+)\],\s*"
    r"minDZ:\s*([-\d.]+),\s*maxDZ:\s*([-\d.]+)"
)

RE_KILLED_6_DETAIL = re.compile(
    r"Killed here 6 --> idphi:\s*(\d+),\s*iphicut:\s*(\d+)"
)
RE_KILLED_6_SIMPLE = re.compile(r"Killed here 6$")

RE_KILLED_7 = re.compile(r"Killed here 7\b")

RE_KILLED_8 = re.compile(r"Killed here 8\b")

RE_KILLED_9 = re.compile(
    r"Killed here 9: stub direction inconsistent "
    r"\(dphidr_doublet=([-\d.]+),\s*dphidr_stub=([-\d.]+),\s*error=([-\d.]+)\)"
)

RE_KILLED_10 = re.compile(
    r"Killed here 10: stub sigma cut "
    r"\(sig=([-\d.]+)\s*>\s*cut=([-\d.]+),\s*barrel_i=(\d+)\s+barrel_o=(\d+)\)"
)

RE_TRIPLET = re.compile(
    r"Triplet no\.\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)\s+(\d+)\)\s*-\s*(\d+)\s+(\d+)\s*->\s*"
    r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
)

# TripletCheck debug output: comprehensive triplet formation check
# Format: TripletCheck;cellIdx;otherCell;outerPair;innerPair;layer1;layer2;layer3;
#         hasSS;thetaVal;thetaThresh;thetaCut;curv;dcaVal;dcaThresh;dcaCut;hardCurv;
#         thetaPass;dcaPass;stubType1;stubType2;stubType3;overallPass
RE_TRIPLET_CHECK = re.compile(
    r"TripletCheck;(\d+);(\d+);(\d+);(\d+);(\d+);(\d+);(\d+);(\d+);"
    r"([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);"
    r"(\d+);(\d+);(\w+);(\w+);(\w+);(\d+)"
)

# ThetaCheck debug output from areAlignedRZ function
# Format: ThetaCheck;r1;z1;ri;zi;ro;zo;lhs;rhs;thetaCut;ptmin;pass
RE_THETA_CHECK = re.compile(
    r"ThetaCheck;([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);"
    r"([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);(\d+)"
)

# DCACheck debug output from dcaCut function
# Format: DCACheck;x1;y1;x2;y2;x3;y3;curvature;maxCurv;dca;dcaThresh;dcaCut;curvPass;dcaPass;overallPass
RE_DCA_CHECK = re.compile(
    r"DCACheck;([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);"
    r"([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);([-\d.e+]+);(\d+);(\d+);(\d+)"
)

RE_FILLING_CELL = re.compile(
    r"filling cell no\.\s*(\d+)\s+(\d+):\s*(\d+)\s*->\s*(\d+)"
)

RE_FISHBONE_KILL = re.compile(
    r"cell\s+(\d+)\s+kill\s+(\d+)\s+cos\s*=\s*([-\d.]+)\s+n1\s*=\s*([-\d.]+)\s+n2\s*=\s*([-\d.]+)\s+(same|diff)"
)

RE_KERNEL_CONNECT = re.compile(r"Kernel_connect -> Done!")
RE_FISHBONE_DONE = re.compile(r"Early fishbone -> Done!")

RE_STARTING_NTUPLETS = re.compile(
    r"starting producing ntuplets from (\d+) cells and (\d+) triplets"
)

RE_TRACK = re.compile(
    r"track n\.\s*(\d+)\s+nhits\s+(\d+)\s+with cells:\s*([\d\s\-]+)"
)

RE_NTUPLET_DOUBLET = re.compile(
    r"Doublet no\.\s*(\d+)\s+(\d+)\s+doubletId:\s*(\d+)\s*->\s*(\d+)\s+"
    r"\(isKilled\s+(\d+)\)\s+\((\d+),(\d+)\)\s*->\s*\((\d+),(\d+)\)\s+(\d+)\s+(\d+)"
)

RE_LAYER_PAIR_DOIT = re.compile(
    r"LayerPairId\s+(\d+)\s+doit\s*\?\s*(\d+)\s+From cell\s+(\d+)\s+with nNeighbors\s*=\s*(\d+)"
)

RE_FILL_NLAYERS = re.compile(
    r"Kernel_fillNLayers\s+(\d+)\s+(\d+)\s+(\d+)\s*-\s*(\d+)\s+(\d+)"
)

RE_NSIZES = re.compile(
    r"nSizes:([\d;]+)"
)

RE_TOTAL_TRACKS = re.compile(
    r"Total tracks found:\s*(\d+)"
)

RE_CA_SUMMARY = re.compile(r"={5,}\s*CA Tracking Summary\s*={5,}")

RE_START_TUPLE_BUILDING = re.compile(r"start tuple building\. N hits (\d+)")

RE_FILL_DOUBLETS_HISTO = re.compile(r"FillDoubletsHisto\s*->\s*done!")

RE_EVENT_SUMMARY = re.compile(r"=== EVENT SUMMARY")

RE_ETA_SUMMARY = re.compile(
    r"Eta ([><]) 0:\s*(\d+) tracks,\s*(\d+) with issues "
    r"\((\d+) null hits,\s*(\d+) z mismatches\)"
)


# ---------------------------------------------------------------------------
# Data structures for holding parsed event data
# ---------------------------------------------------------------------------

class EventData:
    """Container for all parsed data from a single event."""

    def __init__(self, record_number, run, event, lumi):
        self.record_number = record_number
        self.run = run
        self.event = event
        self.lumi = lumi

        # Doublet building
        self.n_hits = 0
        self.n_actual_pairs = 0
        self.max_num_doublets = 0
        self.cell_z0_cut = 0.0
        self.pt_min = 0.0

        # Doublets per layer pair: {pair_id: [(doublet_idx, inner_layer, outer_layer, i, oi), ...]}
        self.doublets_by_pair = defaultdict(list)
        # Layer pair definitions: {pair_id: (inner_layer, outer_layer)}
        self.pair_layer_map = OrderedDict()

        # Kill statistics during doublet building
        self.kills = defaultdict(int)  # {kill_code: count}
        self.kill_details = defaultdict(list)  # {kill_code: [(pair_id, inner_layer, outer_layer, detail_str), ...]}
        self.kills_by_pair = defaultdict(lambda: defaultdict(int))  # {pair_id: {kill_code: count}}

        # Structured kill data for cut suggestions: each entry records the
        # cut name, the actual value that was tested, and the threshold(s).
        # {kill_code: [(pair_id, inner_layer, outer_layer, cut_name,
        #               actual_value, threshold_low, threshold_high), ...]}
        self.kill_cut_values = defaultdict(list)

        # Triplets / cells
        self.triplets = []  # [(triplet_no, doublet_a, doublet_b, layer_info), ...]
        self.cells = []  # [(cell_no, status, inner_doublet, outer_doublet), ...]

        # Fishbone cleaning
        self.fishbone_kills = []  # [(killer_cell, killed_cell, cos, n1, n2, same_or_diff), ...]

        # n-tuplet / track information
        self.n_cells_for_ntuplets = 0
        self.n_triplets_for_ntuplets = 0
        self.tracks = []  # [(track_idx, nhits, [cell_ids]), ...]

        # Track quality
        self.track_nlayers = []  # [(track_idx, total_tracks, nlayers, ...), ...]

        # nSizes line: nHits;nDoublets;nCells;nTriplets;nTracks_raw;nTracks_final;maxDoublets
        self.n_sizes = None

        # Final summary
        self.total_tracks = 0

        # Event summary (barrel tracks)
        self.eta_pos_tracks = 0
        self.eta_neg_tracks = 0
        self.eta_pos_issues = 0
        self.eta_neg_issues = 0

        # Triplet formation debug data (CA_DEBUG)
        # triplet_checks: list of dicts with detailed triplet check info
        self.triplet_checks = []
        # triplet_check_summary: {(layer1, layer2, layer3): {passed: int, failed_theta: int, failed_dca: int}}
        self.triplet_check_summary = defaultdict(lambda: {'passed': 0, 'failed_theta': 0, 'failed_dca': 0, 'failed_both': 0})
        # theta_checks: detailed theta alignment checks
        self.theta_checks = []
        # dca_checks: detailed DCA checks
        self.dca_checks = []


def parse_log(filepath):
    """
    Parse the entire log file and return:
      - header_info: dict of global configuration data
      - layer_map: dict mapping CA layer -> (subsystem, subdet_layer, module_start, is_barrel)
      - events: list of EventData objects
    """
    header_info = {}
    layer_map = {}
    events = []
    current_event = None

    # Track which phase we are in
    in_header = True
    # Track the last pairLayerId context for associating kills with layer pairs
    last_pair_context = None  # (pair_id, inner_layer, outer_layer)

    with open(filepath, 'r') as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.rstrip('\n')

            # ----- Global header parsing -----
            if in_header:
                m = RE_LAYER_START_PIXEL.search(line)
                if m:
                    ca_layer = int(m.group(1))
                    subdet_layer = int(m.group(2))
                    module_start = int(m.group(3))
                    is_barrel = (m.group(4) == "barrel")
                    layer_map[ca_layer] = ("Pixel", subdet_layer, module_start, is_barrel)
                    continue

                m = RE_LAYER_START_OT.search(line)
                if m:
                    ca_layer = int(m.group(1))
                    module_start = int(m.group(2))
                    subtype = m.group(3)
                    subdet_layer = int(m.group(4))
                    is_barrel = (subtype == "barrel")
                    layer_map[ca_layer] = ("OT", subdet_layer, module_start, is_barrel)
                    continue

                if "No. Layers to be used" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        header_info["n_layers"] = int(parts[1].strip())
                    continue

                if "No. Pairs to be used" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        header_info["n_pairs"] = int(parts[1].strip())
                    continue

            # ----- Event boundary detection -----
            m = RE_BEGIN_EVENT.search(line)
            if m:
                in_header = False
                if current_event is not None:
                    events.append(current_event)
                record_number = int(m.group(1))
                run = int(m.group(2))
                event = int(m.group(3))
                lumi = int(m.group(4))
                current_event = EventData(record_number, run, event, lumi)
                last_pair_context = None
                continue

            if current_event is None:
                continue

            # ----- Doublet building -----
            m = RE_BUILDING_DOUBLETS.search(line)
            if m:
                current_event.n_hits = int(m.group(1))
                continue

            if "nActualPairs" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    try:
                        current_event.n_actual_pairs = int(parts[1].strip())
                    except ValueError:
                        pass
                continue

            m = RE_MAX_NUM_DOUBLETS.search(line)
            if m:
                current_event.max_num_doublets = int(m.group(1))
                current_event.cell_z0_cut = float(m.group(4))
                current_event.pt_min = float(m.group(5))
                continue

            # ----- Pair layer mapping -----
            m = RE_PAIR_LAYER_ID.search(line)
            if m:
                pair_id = int(m.group(1))
                inner_layer = int(m.group(3))
                outer_layer = int(m.group(4))
                last_pair_context = (pair_id, inner_layer, outer_layer)
                if pair_id not in current_event.pair_layer_map:
                    current_event.pair_layer_map[pair_id] = (inner_layer, outer_layer)
                continue

            # ----- Doublets -----
            m = RE_DOUBLET.search(line)
            if m:
                doublet_idx = int(m.group(1))
                pair_id = int(m.group(2))
                inner_layer = int(m.group(3))
                outer_layer = int(m.group(4))
                i_val = int(m.group(5))
                oi_val = int(m.group(6))
                current_event.doublets_by_pair[pair_id].append(
                    (doublet_idx, inner_layer, outer_layer, i_val, oi_val)
                )
                # Also populate the pair->layer mapping
                if pair_id not in current_event.pair_layer_map:
                    current_event.pair_layer_map[pair_id] = (inner_layer, outer_layer)
                continue

            # ----- Kill codes during doublet formation -----
            # Kill code 1: invalid detector index
            if RE_KILLED_1.search(line):
                current_event.kills[1] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[1].append((pid, il, ol, "invalid detector index"))
                current_event.kills_by_pair[pid][1] += 1
                continue

            # Kill code 2: inner value out of range
            m = RE_KILLED_2.search(line)
            if m:
                current_event.kills[2] += 1
                val_inner = float(m.group(1))
                min_inner = float(m.group(3))
                max_inner = float(m.group(4))
                detail = (
                    f"valInner={m.group(1)} [idx:{m.group(2)}] "
                    f"range=[{m.group(3)}, {m.group(4)}]"
                )
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[2].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][2] += 1
                # Determine which bound was violated
                if val_inner < min_inner:
                    current_event.kill_cut_values[2].append(
                        (pid, il, ol, "minInner", val_inner, min_inner, None))
                else:
                    current_event.kill_cut_values[2].append(
                        (pid, il, ol, "maxInner", val_inner, None, max_inner))
                continue

            # Kill code 4: module index invalid or cluster size cut
            m = RE_KILLED_4_DETAIL.search(line)
            if m:
                current_event.kills[4] += 1
                detail = f"mo={m.group(1)} >= numberOfModules={m.group(2)}"
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[4].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][4] += 1
                continue

            if RE_KILLED_4_SIMPLE.search(line):
                current_event.kills[4] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[4].append((pid, il, ol, "cluster size cut"))
                current_event.kills_by_pair[pid][4] += 1
                continue

            # Kill code 5: outer value / dz / z0 cut
            m = RE_KILLED_5_VAL_OUTER.search(line)
            if m:
                current_event.kills[5] += 1
                val_outer = float(m.group(1))
                min_outer = float(m.group(3))
                max_outer = float(m.group(4))
                detail = (
                    f"valOuter={m.group(1)} [idx:{m.group(2)}] "
                    f"range=[{m.group(3)}, {m.group(4)}]"
                )
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[5].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][5] += 1
                if val_outer < min_outer:
                    current_event.kill_cut_values[5].append(
                        (pid, il, ol, "minOuter", val_outer, min_outer, None))
                else:
                    current_event.kill_cut_values[5].append(
                        (pid, il, ol, "maxOuter", val_outer, None, max_outer))
                continue

            m = RE_KILLED_5_DZ.search(line)
            if m:
                current_event.kills[5] += 1
                dz_val = float(m.group(1))
                min_dz = float(m.group(3))
                max_dz = float(m.group(4))
                detail = (
                    f"dz={m.group(1)} [idx:{m.group(2)}] "
                    f"range=[{m.group(3)}, {m.group(4)}]"
                )
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[5].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][5] += 1
                if dz_val < min_dz:
                    current_event.kill_cut_values[5].append(
                        (pid, il, ol, "minDZ", dz_val, min_dz, None))
                else:
                    current_event.kill_cut_values[5].append(
                        (pid, il, ol, "maxDZ", dz_val, None, max_dz))
                continue

            if RE_KILLED_5_SIMPLE.search(line):
                current_event.kills[5] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[5].append((pid, il, ol, "z0 cut"))
                current_event.kills_by_pair[pid][5] += 1
                # z0 cut has no detailed values in the simple form
                current_event.kill_cut_values[5].append(
                    (pid, il, ol, "z0Cut", None, None, None))
                continue

            # Kill code 6: phi cut (idphi > iphicut)
            m = RE_KILLED_6_DETAIL.search(line)
            if m:
                current_event.kills[6] += 1
                idphi = int(m.group(1))
                iphicut_val = int(m.group(2))
                detail = f"idphi={m.group(1)} > iphicut={m.group(2)}"
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[6].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][6] += 1
                current_event.kill_cut_values[6].append(
                    (pid, il, ol, "phiCut", idphi, None, iphicut_val))
                continue

            if RE_KILLED_6_SIMPLE.search(line):
                current_event.kills[6] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[6].append((pid, il, ol, "phi cut"))
                current_event.kills_by_pair[pid][6] += 1
                continue

            # Kill code 7: z-size cut
            if RE_KILLED_7.search(line):
                current_event.kills[7] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[7].append((pid, il, ol, "z-size cut"))
                current_event.kills_by_pair[pid][7] += 1
                continue

            # Kill code 8: pt cut
            if RE_KILLED_8.search(line):
                current_event.kills[8] += 1
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[8].append((pid, il, ol, "pt cut"))
                current_event.kills_by_pair[pid][8] += 1
                continue

            # Kill code 9: stub direction inconsistent
            m = RE_KILLED_9.search(line)
            if m:
                current_event.kills[9] += 1
                dphidr_doublet = float(m.group(1))
                dphidr_stub = float(m.group(2))
                dphidr_error = float(m.group(3))
                detail = (
                    f"dphidr_doublet={m.group(1)} dphidr_stub={m.group(2)} "
                    f"error={m.group(3)}"
                )
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[9].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][9] += 1
                # The cut is: |dphidr_doublet - dphidr_stub| > nSigma * error
                # actual = nsigma used, threshold = 5.0 (hardcoded nSigma)
                nsigma_actual = (abs(dphidr_doublet - dphidr_stub) / dphidr_error
                                 if dphidr_error > 0 else float('inf'))
                current_event.kill_cut_values[9].append(
                    (pid, il, ol, "stubDirNSigma", nsigma_actual, None, 5.0))
                continue

            # Kill code 10: stub sigma cut
            m = RE_KILLED_10.search(line)
            if m:
                current_event.kills[10] += 1
                significance = float(m.group(1))
                sigma_cut = float(m.group(2))
                detail = (
                    f"sig={m.group(1)} > cut={m.group(2)} "
                    f"barrel_i={m.group(3)} barrel_o={m.group(4)}"
                )
                pid, il, ol = last_pair_context if last_pair_context else (-1, -1, -1)
                current_event.kill_details[10].append((pid, il, ol, detail))
                current_event.kills_by_pair[pid][10] += 1
                current_event.kill_cut_values[10].append(
                    (pid, il, ol, "stubSigmaCut", significance, None, sigma_cut))
                continue

            # ----- Triplets -----
            m = RE_TRIPLET.search(line)
            if m:
                triplet_no = int(m.group(1))
                cut1 = float(m.group(2))
                cut2 = float(m.group(3))
                pair_a = int(m.group(4))
                pair_b = int(m.group(5))
                inner_doublet = int(m.group(6))
                outer_doublet = int(m.group(7))
                layer_info = (int(m.group(8)), int(m.group(9)),
                              int(m.group(10)), int(m.group(11)))
                current_event.triplets.append(
                    (triplet_no, cut1, cut2, pair_a, pair_b,
                     inner_doublet, outer_doublet, layer_info)
                )
                continue

            # ----- Triplet Check (CA_DEBUG) -----
            m = RE_TRIPLET_CHECK.search(line)
            if m:
                check = {
                    'outer_cell': int(m.group(1)),
                    'inner_cell': int(m.group(2)),
                    'outer_pair': int(m.group(3)),
                    'inner_pair': int(m.group(4)),
                    'layer1': int(m.group(5)),
                    'layer2': int(m.group(6)),
                    'layer3': int(m.group(7)),
                    'has_ss_stub': int(m.group(8)) == 1,
                    'theta_value': float(m.group(9)),
                    'theta_threshold': float(m.group(10)),
                    'theta_cut': float(m.group(11)),
                    'curvature': float(m.group(12)),
                    'dca_value': float(m.group(13)),
                    'dca_threshold': float(m.group(14)),
                    'dca_cut': float(m.group(15)),
                    'hard_curv_cut': float(m.group(16)),
                    'theta_passed': int(m.group(17)) == 1,
                    'dca_passed': int(m.group(18)) == 1,
                    'stub_type1': m.group(19),
                    'stub_type2': m.group(20),
                    'stub_type3': m.group(21),
                    'overall_passed': int(m.group(22)) == 1,
                }
                current_event.triplet_checks.append(check)
                # Update summary
                layers = (check['layer1'], check['layer2'], check['layer3'])
                if check['overall_passed']:
                    current_event.triplet_check_summary[layers]['passed'] += 1
                elif not check['theta_passed'] and not check['dca_passed']:
                    current_event.triplet_check_summary[layers]['failed_both'] += 1
                elif not check['theta_passed']:
                    current_event.triplet_check_summary[layers]['failed_theta'] += 1
                else:
                    current_event.triplet_check_summary[layers]['failed_dca'] += 1
                continue

            # ----- Theta Check (CA_DEBUG) -----
            m = RE_THETA_CHECK.search(line)
            if m:
                check = {
                    'r1': float(m.group(1)),
                    'z1': float(m.group(2)),
                    'ri': float(m.group(3)),
                    'zi': float(m.group(4)),
                    'ro': float(m.group(5)),
                    'zo': float(m.group(6)),
                    'lhs': float(m.group(7)),
                    'rhs': float(m.group(8)),
                    'theta_cut': float(m.group(9)),
                    'ptmin': float(m.group(10)),
                    'passed': int(m.group(11)) == 1,
                }
                current_event.theta_checks.append(check)
                continue

            # ----- DCA Check (CA_DEBUG) -----
            m = RE_DCA_CHECK.search(line)
            if m:
                check = {
                    'x1': float(m.group(1)),
                    'y1': float(m.group(2)),
                    'x2': float(m.group(3)),
                    'y2': float(m.group(4)),
                    'x3': float(m.group(5)),
                    'y3': float(m.group(6)),
                    'curvature': float(m.group(7)),
                    'max_curv': float(m.group(8)),
                    'dca': float(m.group(9)),
                    'dca_threshold': float(m.group(10)),
                    'dca_cut': float(m.group(11)),
                    'curv_passed': int(m.group(12)) == 1,
                    'dca_passed': int(m.group(13)) == 1,
                    'overall_passed': int(m.group(14)) == 1,
                }
                current_event.dca_checks.append(check)
                continue

            # ----- Filling cells -----
            m = RE_FILLING_CELL.search(line)
            if m:
                cell_no = int(m.group(1))
                status = int(m.group(2))
                inner_d = int(m.group(3))
                outer_d = int(m.group(4))
                current_event.cells.append((cell_no, status, inner_d, outer_d))
                continue

            # ----- Fishbone kills -----
            m = RE_FISHBONE_KILL.search(line)
            if m:
                killer = int(m.group(1))
                killed = int(m.group(2))
                cos_val = float(m.group(3))
                n1 = float(m.group(4))
                n2 = float(m.group(5))
                same_diff = m.group(6)
                current_event.fishbone_kills.append(
                    (killer, killed, cos_val, n1, n2, same_diff)
                )
                continue

            # ----- n-tuplet building -----
            m = RE_STARTING_NTUPLETS.search(line)
            if m:
                current_event.n_cells_for_ntuplets = int(m.group(1))
                current_event.n_triplets_for_ntuplets = int(m.group(2))
                continue

            # ----- Tracks -----
            m = RE_TRACK.search(line)
            if m:
                track_idx = int(m.group(1))
                nhits = int(m.group(2))
                cell_str = m.group(3).strip()
                cell_ids = [int(c.strip()) for c in cell_str.split('-') if c.strip()]
                current_event.tracks.append((track_idx, nhits, cell_ids))
                continue

            # ----- Track n-layers info -----
            m = RE_FILL_NLAYERS.search(line)
            if m:
                track_idx = int(m.group(1))
                total = int(m.group(2))
                nlayers = int(m.group(3))
                current_event.track_nlayers.append(
                    (track_idx, total, nlayers)
                )
                continue

            # ----- nSizes -----
            m = RE_NSIZES.search(line)
            if m:
                current_event.n_sizes = [int(x) for x in m.group(1).split(';')]
                continue

            # ----- Total tracks -----
            m = RE_TOTAL_TRACKS.search(line)
            if m:
                current_event.total_tracks = int(m.group(1))
                continue

            # ----- Event summary (barrel) -----
            m = RE_ETA_SUMMARY.search(line)
            if m:
                direction = m.group(1)
                n_tracks = int(m.group(2))
                n_issues = int(m.group(3))
                if direction == '>':
                    current_event.eta_pos_tracks = n_tracks
                    current_event.eta_pos_issues = n_issues
                else:
                    current_event.eta_neg_tracks = n_tracks
                    current_event.eta_neg_issues = n_issues
                continue

    # Don't forget the last event
    if current_event is not None:
        events.append(current_event)

    return header_info, layer_map, events


# ---------------------------------------------------------------------------
# OT Stub Formation log parsing
# ---------------------------------------------------------------------------

# Module index ranges (from OT Module Configuration header)
OT_BARREL_START = 4000
OT_BACKWARD_START = 11288
OT_FORWARD_START = 14244
OT_TOTAL_MODULES = 17200


class OTStubEventData:
    """Container for OT stub formation data from a single event."""

    def __init__(self, record_number, run, event, lumi):
        self.record_number = record_number
        self.run = run
        self.event = event
        self.lumi = lumi

        # OT RecHits
        self.n_ot_rechits = 0

        # Bend-based stubs (GPU)
        self.n_bend_stubs = 0
        self.bend_stubs = []  # list of stub dicts

        # VectorHits-style stubs (GPU)
        self.n_vh_style_stubs = 0
        self.vh_style_stubs = []  # list of stub dicts

        # VectorHits (CPU reference)
        self.n_vh_accepted = 0
        self.n_vh_rejected = 0
        self.n_vh_output = 0  # vectorhit in output count

        # Per-module stub counts from CountStubs kernel
        self.module_stub_counts = {}  # {module_idx: (n_stubs, maxBend)}


RE_OT_BEGIN_EVENT = re.compile(
    r"Begin processing the (\d+)\w+ record\. Run (\d+), Event (\d+), LumiSection (\d+)"
)
RE_OT_TOTAL_HITS = re.compile(r"Total OT hits processed:\s*(\d+)")
RE_OT_TOTAL_STUBS_FORMED = re.compile(r"Total stubs formed:\s*(\d+)")
RE_OT_RECHITS_TOTAL = re.compile(r"Total number of hits:\s*(\d+)")
RE_OT_STUBS_TOTAL = re.compile(r"Total number of stubs:\s*(\d+)")
RE_OT_STUB_DETAIL = re.compile(
    r"Stub\s+(\d+):\s+pos=\(\s*([-\d.e+]+),\s*([-\d.e+]+),\s*([-\d.e+]+)\)\s+"
    r"r=\s*([-\d.e+]+)\s+iphi=\s*([-\d]+)\s+"
    r"localPos=\(\s*([-\d.e+]+),\s*([-\d.e+]+)\)\s+"
    r"bend=\s*([-\d.e+]+)\s+ptEst=\s*([-\d.e+]+)\s+"
    r"dPhiDr=\s*([-\d.e+]+)\s+dPhiDrErr=\s*([-\d.e+]+)\s+"
    r"detIdx=\s*(\d+)\s+innerHit=\s*(\d+)\s+outerHit=\s*(\d+)\s+"
    r"type=\s*(\d+)\s+flags=(\d+)"
)
RE_COUNT_STUBS_RESULT = re.compile(
    r"CountStubs: Module (\d+) counted (\d+) stubs \(maxBend=([\d.e+-]+)\)"
)
RE_VH_ACCEPT = re.compile(r"accepting VH!")
RE_VH_REJECT = re.compile(r"rejecting VH:")
RE_VH_OUTPUT = re.compile(r"vectorhit in output\s+VectorHit create in the DetId#:\s*(\d+)")
RE_OT_MODULE_CONFIG = re.compile(
    r"OT Barrel: (\d+) modules \(CA index (\d+)-(\d+)\)"
)


def _classify_detidx(detIdx):
    """Classify a detIdx into an OT region string."""
    if detIdx < OT_BARREL_START:
        return None  # pixel, shouldn't happen for OT stubs
    elif detIdx < OT_BACKWARD_START:
        return "barrel"
    elif detIdx < OT_FORWARD_START:
        return "backward"
    else:
        return "forward"


def _detidx_to_ot_layer(detIdx, layer_starts):
    """Map a detIdx to an OT layer name using the CA layer start boundaries.

    layer_starts is a list of (ca_layer, module_start, is_barrel, subsys, subdet_layer)
    sorted by module_start, for OT layers only.
    """
    if detIdx < OT_BARREL_START:
        return None
    for i in range(len(layer_starts) - 1, -1, -1):
        if detIdx >= layer_starts[i][1]:
            ca_layer, _, is_barrel, _, subdet_layer = layer_starts[i]
            return ca_layer
    return None


def parse_ot_stub_log(filepath):
    """Parse the OT stub formation log file.

    Returns a list of OTStubEventData objects, one per event.
    """
    events = []
    current_event = None

    # State machine for which section we're in
    in_bend_stubs = False
    in_vh_style_stubs = False
    in_rechits = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # Event boundary
            m = RE_OT_BEGIN_EVENT.search(line)
            if m:
                if current_event is not None:
                    events.append(current_event)
                record_number = int(m.group(1))
                run = int(m.group(2))
                event = int(m.group(3))
                lumi = int(m.group(4))
                current_event = OTStubEventData(record_number, run, event, lumi)
                in_bend_stubs = False
                in_vh_style_stubs = False
                in_rechits = False
                continue

            if current_event is None:
                continue

            # VectorHit accept/reject (CPU reference)
            if RE_VH_ACCEPT.search(line):
                current_event.n_vh_accepted += 1
                continue
            if RE_VH_REJECT.search(line):
                current_event.n_vh_rejected += 1
                continue

            # VectorHit output count
            if RE_VH_OUTPUT.search(line):
                current_event.n_vh_output += 1
                continue

            # CountStubs per-module results
            m = RE_COUNT_STUBS_RESULT.search(line)
            if m:
                mod_idx = int(m.group(1))
                n_stubs = int(m.group(2))
                max_bend = float(m.group(3))
                current_event.module_stub_counts[mod_idx] = (n_stubs, max_bend)
                continue

            # OT Stub Formation Summary
            m = RE_OT_TOTAL_HITS.search(line)
            if m:
                current_event.n_ot_rechits = int(m.group(1))
                continue

            # Section markers
            if "=== Bend-Based Stubs ===" in line:
                in_bend_stubs = True
                in_vh_style_stubs = False
                in_rechits = False
                continue
            if "=== VectorHits-Style Stubs ===" in line:
                in_vh_style_stubs = True
                in_bend_stubs = False
                in_rechits = False
                continue
            if "=== OT RecHits ===" in line:
                in_rechits = True
                in_bend_stubs = False
                in_vh_style_stubs = False
                continue

            # Total number of stubs (in bend or VH-style section)
            m = RE_OT_STUBS_TOTAL.search(line)
            if m:
                count = int(m.group(1))
                if in_bend_stubs:
                    current_event.n_bend_stubs = count
                elif in_vh_style_stubs:
                    current_event.n_vh_style_stubs = count
                continue

            # Total number of hits (in rechits section)
            m = RE_OT_RECHITS_TOTAL.search(line)
            if m and in_rechits:
                current_event.n_ot_rechits = int(m.group(1))
                continue

            # Individual stub detail
            m = RE_OT_STUB_DETAIL.search(line)
            if m:
                stub = {
                    'idx': int(m.group(1)),
                    'x': float(m.group(2)),
                    'y': float(m.group(3)),
                    'z': float(m.group(4)),
                    'r': float(m.group(5)),
                    'bend': float(m.group(9)),
                    'ptEst': float(m.group(10)),
                    'dPhiDr': float(m.group(11)),
                    'dPhiDrErr': float(m.group(12)),
                    'detIdx': int(m.group(13)),
                    'innerHit': int(m.group(14)),
                    'outerHit': int(m.group(15)),
                    'type': int(m.group(16)),
                    'flags': m.group(17),
                    'region': _classify_detidx(int(m.group(13))),
                }
                if in_bend_stubs:
                    current_event.bend_stubs.append(stub)
                elif in_vh_style_stubs:
                    current_event.vh_style_stubs.append(stub)
                continue

    if current_event is not None:
        events.append(current_event)

    return events


def _build_ot_layer_starts(layer_map):
    """Build sorted list of OT layer start boundaries from the CA layer map."""
    ot_layers = []
    for ca_layer, (subsys, subdet_layer, mod_start, is_barrel) in layer_map.items():
        if subsys == "OT":
            ot_layers.append((ca_layer, mod_start, is_barrel, subsys, subdet_layer))
    ot_layers.sort(key=lambda x: x[1])
    return ot_layers


def _stubs_by_ot_layer(stubs, layer_starts, layer_map):
    """Group stubs by OT CA layer.

    Returns an OrderedDict: {ca_layer: [stub, ...]} sorted by ca_layer.
    """
    by_layer = defaultdict(list)
    for s in stubs:
        ca_layer = _detidx_to_ot_layer(s['detIdx'], layer_starts)
        if ca_layer is not None:
            by_layer[ca_layer].append(s)
    return OrderedDict(sorted(by_layer.items()))


# ---------------------------------------------------------------------------
# Layer name helper
# ---------------------------------------------------------------------------

def _endcap_sign(layer_id, subsys, layer_map):
    """Determine +/- z sign for an endcap CA layer.

    Convention based on CMS Phase-2 CA layer ordering:
    - Pixel: forward (+z) endcap layers come first, backward (-z) second
    - OT: backward (-z) endcap layers come first, forward (+z) second
    """
    endcap_layers = sorted(
        lid for lid, (s, _, _, is_b) in layer_map.items()
        if s == subsys and not is_b
    )
    if not endcap_layers:
        return ""
    mid = len(endcap_layers) // 2
    first_half = set(endcap_layers[:mid])
    if subsys == "Pixel":
        return "+" if layer_id in first_half else "-"
    else:
        return "-" if layer_id in first_half else "+"


def layer_name(layer_id, layer_map):
    """Return a human-readable name for a CA layer, including +/- for endcap."""
    if layer_id in layer_map:
        subsys, subdet_layer, _, is_barrel = layer_map[layer_id]
        if is_barrel:
            return f"{subsys}B{subdet_layer}"
        else:
            sign = _endcap_sign(layer_id, subsys, layer_map)
            return f"{subsys}E{sign}{subdet_layer}"
    return f"L{layer_id}"


def is_ot_layer(layer_id, layer_map):
    """Return True if the CA layer belongs to the Outer Tracker."""
    if layer_id in layer_map:
        return layer_map[layer_id][0] == "OT"
    return False


def build_doublet_layers(ev):
    """Build a dict mapping doublet index -> (inner_layer, outer_layer)."""
    doublet_layers = {}
    for pair_id, doublets in ev.doublets_by_pair.items():
        for (doublet_idx, inner_layer, outer_layer, i, oi) in doublets:
            doublet_layers[doublet_idx] = (inner_layer, outer_layer)
    return doublet_layers


def get_track_layers(cell_ids, doublet_layers):
    """Extract the ordered layer sequence from a chain of cell (doublet) IDs.

    Returns a list of CA layer IDs traversed by the track, or None if
    any cell ID cannot be resolved.
    """
    if not cell_ids:
        return None
    first = cell_ids[0]
    if first not in doublet_layers:
        return None
    layers = [doublet_layers[first][0], doublet_layers[first][1]]
    for cid in cell_ids[1:]:
        if cid not in doublet_layers:
            return None
        layers.append(doublet_layers[cid][1])
    return layers


def count_hits(layers, layer_map):
    """Count hits with OT stubs expanded to 2 hits each.

    Returns (n_pixel, n_ot_stubs, expanded_total).
    """
    n_pixel = sum(1 for l in layers if not is_ot_layer(l, layer_map))
    n_ot = sum(1 for l in layers if is_ot_layer(l, layer_map))
    return n_pixel, n_ot, n_pixel + n_ot * 2


# ---------------------------------------------------------------------------
# Cut suggestion analysis
# ---------------------------------------------------------------------------

# Maps cut_name to whether the cut is an upper bound (actual > threshold)
# or a lower bound (actual < threshold).
# For upper-bound cuts, suggest increasing the threshold.
# For lower-bound cuts, suggest decreasing the threshold.
CUT_BOUND_TYPE = {
    "minInner": "lower",    # actual < minInner  -> suggest decreasing minInner
    "maxInner": "upper",    # actual > maxInner  -> suggest increasing maxInner
    "minOuter": "lower",    # actual < minOuter  -> suggest decreasing minOuter
    "maxOuter": "upper",    # actual > maxOuter  -> suggest increasing maxOuter
    "minDZ": "lower",       # dz < minDZ         -> suggest decreasing minDZ
    "maxDZ": "upper",       # dz > maxDZ         -> suggest increasing maxDZ
    "phiCut": "upper",      # idphi > iphicut    -> suggest increasing phiCut
    "stubDirNSigma": "upper",   # nsigma > 5     -> suggest increasing nSigma
    "stubSigmaCut": "upper",    # sig > cut      -> suggest increasing stubSigmaCut
    "z0Cut": None,          # no numeric detail available
}

# Human-readable parameter names for the Python configuration
CUT_CONFIG_NAMES = {
    "minInner": "minInZ/minInR (inner coordinate lower bound)",
    "maxInner": "maxInZ/maxInR (inner coordinate upper bound)",
    "minOuter": "minOutZ/minOutR (outer coordinate lower bound)",
    "maxOuter": "maxOutZ/maxOutR (outer coordinate upper bound)",
    "minDZ": "minDZ (signed dz lower bound)",
    "maxDZ": "maxDZ (signed dz upper bound)",
    "phiCut": "phiCuts (phi difference cut)",
    "stubDirNSigma": "nSigma in stub direction check (hardcoded=5.0)",
    "stubSigmaCut": "stubSigmaCut (stub-stub sigma cut)",
    "z0Cut": "cellZ0Cut (z0 compatibility cut)",
}

# Mapping from cut_name to layerPairs table column index
# layerPairs columns:
#  0:i,  1:o,  2:start,  3:phiCut,  4:minIn,  5:maxIn,  6:minOut,  7:maxOut,
#  8:maxDR,  9:minDZ,  10:maxDZ,  11:ptCuts,  12:stubSigma
CUT_TO_LAYER_PAIR_COLUMN = {
    "phiCut":       3,
    "minInner":     4,
    "maxInner":     5,
    "minOuter":     6,
    "maxOuter":     7,
    "maxDR":        8,
    "minDZ":        9,
    "maxDZ":       10,
    "stubSigmaCut": 12,
}

# Mapping from triplet cut_name to layers table column index
# layers columns:  0:index,  1:isBarrel,  2:caDCA,  3:caTheta
CUT_TO_LAYER_COLUMN = {
    "caDCA":    2,
    "caTheta":  3,
}

LAYER_PAIR_COL_NAMES = [
    "i", "o", "start", "phiCut", "minIn", "maxIn", "minOut", "maxOut",
    "maxDR", "minDZ", "maxDZ", "ptCuts", "stubSigma"
]

LAYER_COL_NAMES = ["index", "isBarrel", "caDCA", "caTheta"]

# Default safety margin for suggested values (10%)
DEFAULT_SAFETY_MARGIN = 0.10


def _apply_safety_margin(current_val, suggested_val, bound_type, margin=DEFAULT_SAFETY_MARGIN):
    """Apply a safety margin to a suggested cut value.

    For upper-bound cuts (actual > threshold), increase by margin.
    For lower-bound cuts (actual < threshold), decrease by margin.
    Returns the value with margin applied.
    """
    if suggested_val is None:
        return None
    delta = abs(suggested_val - current_val) if current_val is not None else abs(suggested_val)
    # Use the larger of: margin * |current| or margin * |suggested|
    margin_val = margin * max(abs(suggested_val), abs(current_val) if current_val else 0)
    # For very small values, use absolute margin based on the delta
    if margin_val < 1e-6:
        margin_val = margin * max(abs(delta), 1.0)
    if bound_type == "upper":
        return suggested_val + margin_val
    elif bound_type == "lower":
        return suggested_val - margin_val
    return suggested_val


def _round_config_value(val, reference=None):
    """Round a config value to a reasonable precision for the config file.

    Uses the reference (current config value) to determine appropriate precision.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    # For integers (phiCut, layer indices), round to nearest 50 or 100
    if reference is not None and isinstance(reference, int):
        if abs(val) > 100:
            return int(round(val / 50.0) * 50)
        elif abs(val) > 10:
            return int(round(val / 5.0) * 5)
        else:
            return int(round(val))
    # For float values
    if isinstance(val, float):
        if abs(val) > 100:
            return round(val, 0)
        elif abs(val) > 10:
            return round(val, 1)
        elif abs(val) > 1:
            return round(val, 1)
        elif abs(val) > 0.01:
            return round(val, 3)
        else:
            return round(val, 4)
    return val


def parse_config_tables(config_path):
    """Parse the layerPairs and layers tables from the Python config file.

    This reads the config file as Python, extracting the layerPairs and layers
    lists. Returns (layer_pairs_table, layers_table) where each is a list of
    lists matching the config file format.

    Returns (None, None) if the file cannot be parsed.
    """
    try:
        config_vars = {}
        with open(config_path, 'r') as f:
            source = f.read()
        # Execute just enough to extract the tables.
        # We need to provide a minimal cms mock since the config imports it.
        exec_globals = {'__builtins__': __builtins__}
        # Build a minimal mock of FWCore.ParameterSet.Config
        import types
        mock_cms = types.ModuleType('cms')
        # Create mock classes that accept any arguments
        class _MockObj:
            def __init__(self, *a, **kw):
                pass
        for attr in ['EDProducer', 'InputTag', 'double', 'bool', 'uint32',
                     'string', 'int32', 'PSet', 'vdouble', 'vint32',
                     'vuint32', 'vstring', 'Source', 'Process']:
            setattr(mock_cms, attr, _MockObj)
        class _MockUntracked:
            vstring = _MockObj
            PSet = type('MockPSet', (), {'__init__': lambda s, **kw: None})
            string = _MockObj
        mock_cms.untracked = _MockUntracked()
        mock_cms.optional = type('MockOptional', (), {'untracked': _MockUntracked()})()
        mock_fwcore = types.ModuleType('FWCore')
        mock_ps = types.ModuleType('ParameterSet')
        mock_config = types.ModuleType('Config')
        mock_config.cms = mock_cms
        # We just need to make the import work and capture the tables
        import importlib
        import sys as _sys
        # Simpler approach: just exec the raw lines that define the tables
        layer_pairs_table = None
        layers_table = None
        # Extract layerPairs and layers using regex
        # Find the layerPairs assignment
        lp_match = re.search(
            r'^layerPairs\s*=\s*\[(.*?)\n\]',
            source, re.MULTILINE | re.DOTALL)
        layers_match = re.search(
            r'^layers\s*=\s*\[(.*?)\n\]',
            source, re.MULTILINE | re.DOTALL)
        if lp_match:
            lp_body = lp_match.group(1)
            # Strip fully-commented lines (lines starting with # after whitespace)
            # to avoid picking up commented-out array entries
            lp_body = '\n'.join(
                line for line in lp_body.split('\n')
                if not line.lstrip().startswith('#'))
            # Parse each row: find lines starting with [
            rows = re.findall(r'\[\s*(.*?)\]', lp_body)
            layer_pairs_table = []
            for row in rows:
                # Skip comment-only rows
                row_clean = row.split('#')[0].strip().rstrip(',')
                if not row_clean:
                    continue
                vals = []
                for v in row_clean.split(','):
                    v = v.strip()
                    if not v:
                        continue
                    if v in ('True', 'False'):
                        vals.append(v == 'True')
                    elif '.' in v:
                        vals.append(float(v))
                    else:
                        vals.append(int(v))
                if len(vals) >= 12:
                    layer_pairs_table.append(vals)
        if layers_match:
            l_body = layers_match.group(1)
            # Strip fully-commented lines
            l_body = '\n'.join(
                line for line in l_body.split('\n')
                if not line.lstrip().startswith('#'))
            rows = re.findall(r'\[\s*(.*?)\]', l_body)
            layers_table = []
            for row in rows:
                row_clean = row.split('#')[0].strip().rstrip(',')
                if not row_clean:
                    continue
                vals = []
                for v in row_clean.split(','):
                    v = v.strip()
                    if not v:
                        continue
                    if v in ('True', 'False'):
                        vals.append(v == 'True')
                    elif '.' in v:
                        vals.append(float(v))
                    else:
                        vals.append(int(v))
                if len(vals) >= 4:
                    layers_table.append(vals)
        return layer_pairs_table, layers_table
    except Exception as e:
        print(f"  Warning: Could not parse config file {config_path}: {e}",
              file=sys.stderr)
        return None, None


def _find_config_path(script_path=None):
    """Find the default config file path relative to the script location.

    The script is in RecoTracker/PixelSeeding/test/ and the config is in
    RecoTracker/PixelSeeding/python/caHitNtupletAlpakaPhase2OTStubs_cfi.py
    """
    import os
    if script_path is None:
        script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    # Go from test/ to python/
    config_path = os.path.join(
        os.path.dirname(script_dir), "python",
        "caHitNtupletAlpakaPhase2OTStubs_cfi.py")
    if os.path.exists(config_path):
        return config_path
    return None


def _build_pair_id_to_config_index(layer_pairs_table, events):
    """Build mapping from runtime pair_id to config table row index.

    The runtime pair_id is the sequential index into the layerPairsStubs array
    (layerPairs after excluding any layers). With no exclusions, pair_id == row index.

    However, since the config may have layersToExclude, we need to filter.
    We use the (inner, outer) layer mapping from the events to validate.

    Returns a dict: {pair_id: config_row_index}
    """
    if layer_pairs_table is None:
        return {}

    # Build the expected pair_layer_map from events
    event_pair_layers = {}
    for ev in events:
        for pid, (il, ol) in ev.pair_layer_map.items():
            if pid not in event_pair_layers:
                event_pair_layers[pid] = (il, ol)

    # Normally, pair_id == config row index (no exclusions)
    # Verify this by checking that layer pairs match
    mapping = {}
    for pid, (il, ol) in event_pair_layers.items():
        if pid < len(layer_pairs_table):
            row = layer_pairs_table[pid]
            if row[0] == il and row[1] == ol:
                mapping[pid] = pid
            else:
                # Mismatch: try to find the correct row
                # This handles the case where some layers were excluded
                # We'd need to replicate the filtering logic
                pass
    return mapping


class CutSuggestion:
    """A single cut relaxation suggestion for a specific layer pair."""

    def __init__(self, pair_id, inner_layer, outer_layer, cut_name,
                 current_threshold, suggested_threshold, n_recoverable,
                 worst_actual):
        self.pair_id = pair_id
        self.inner_layer = inner_layer
        self.outer_layer = outer_layer
        self.cut_name = cut_name
        self.current_threshold = current_threshold
        self.suggested_threshold = suggested_threshold
        self.n_recoverable = n_recoverable
        self.worst_actual = worst_actual

    def margin_pct(self):
        """How much the threshold needs to change, as a percentage."""
        if self.current_threshold is None or self.current_threshold == 0:
            return None
        return abs(self.suggested_threshold - self.current_threshold) / abs(self.current_threshold) * 100


class TripletCutSuggestion:
    """A cut relaxation suggestion for triplet formation (theta/DCA)."""

    def __init__(self, layer_triple, cut_name, current_threshold,
                 suggested_threshold, n_recoverable, worst_actual):
        self.layer_triple = layer_triple  # (layer1, layer2, layer3)
        self.cut_name = cut_name
        self.current_threshold = current_threshold
        self.suggested_threshold = suggested_threshold
        self.n_recoverable = n_recoverable
        self.worst_actual = worst_actual


def analyze_cut_suggestions(ev, layer_map):
    """Analyze killed doublets and failed triplets to suggest cut relaxations.

    Returns:
        doublet_suggestions: list of CutSuggestion objects
        triplet_suggestions: list of TripletCutSuggestion objects
    """
    doublet_suggestions = _analyze_doublet_cuts(ev, layer_map)
    triplet_suggestions = _analyze_triplet_cuts(ev, layer_map)
    return doublet_suggestions, triplet_suggestions


def _analyze_doublet_cuts(ev, layer_map):
    """Analyze kill_cut_values to suggest doublet cut relaxations."""
    suggestions = []

    # Group kill_cut_values by (pair_id, cut_name)
    # For each group, find the extreme actual values and the threshold
    from collections import defaultdict
    grouped = defaultdict(list)
    for code, entries in ev.kill_cut_values.items():
        for (pid, il, ol, cut_name, actual, thresh_low, thresh_high) in entries:
            if actual is None:
                continue
            grouped[(pid, cut_name)].append(
                (il, ol, actual, thresh_low, thresh_high))

    for (pid, cut_name), entries in sorted(grouped.items()):
        bound_type = CUT_BOUND_TYPE.get(cut_name)
        if bound_type is None:
            continue

        n_kills = len(entries)
        il = entries[0][0]
        ol = entries[0][1]

        if bound_type == "upper":
            # Actual > threshold, so we need to increase threshold
            # The threshold is in thresh_high (or thresh_low for some)
            actuals = [e[2] for e in entries]
            thresholds = [e[4] for e in entries if e[4] is not None]
            if not thresholds:
                thresholds = [e[3] for e in entries if e[3] is not None]
            if not thresholds:
                continue
            worst_actual = max(actuals)
            current_thresh = thresholds[0]  # threshold is constant per pair
            # Suggest the worst actual value (with a small margin)
            suggested = worst_actual
            suggestions.append(CutSuggestion(
                pid, il, ol, cut_name,
                current_thresh, suggested, n_kills, worst_actual))
        elif bound_type == "lower":
            # Actual < threshold, so we need to decrease threshold
            actuals = [e[2] for e in entries]
            thresholds = [e[3] for e in entries if e[3] is not None]
            if not thresholds:
                thresholds = [e[4] for e in entries if e[4] is not None]
            if not thresholds:
                continue
            worst_actual = min(actuals)
            current_thresh = thresholds[0]
            suggested = worst_actual
            suggestions.append(CutSuggestion(
                pid, il, ol, cut_name,
                current_thresh, suggested, n_kills, worst_actual))

    return suggestions


def _analyze_triplet_cuts(ev, layer_map):
    """Analyze failed triplet checks to suggest theta/DCA cut relaxations."""
    suggestions = []

    if not ev.triplet_checks:
        return suggestions

    # Group failed checks by layer triple
    from collections import defaultdict
    failed_by_layers = defaultdict(list)
    for c in ev.triplet_checks:
        if c['overall_passed']:
            continue
        layers = (c['layer1'], c['layer2'], c['layer3'])
        failed_by_layers[layers].append(c)

    for layers, checks in sorted(failed_by_layers.items()):
        # Theta failures
        theta_fails = [c for c in checks if not c['theta_passed']]
        if theta_fails:
            worst_theta = max(c['theta_value'] for c in theta_fails)
            # All checks for the same layer triple have the same theta_cut
            current_theta = theta_fails[0]['theta_cut']
            # The actual check is: theta_value > theta_threshold
            # where theta_threshold = theta_cut * (... geometry factors)
            # We report the raw theta_value vs theta_threshold
            worst_threshold = max(c['theta_threshold'] for c in theta_fails)
            suggestions.append(TripletCutSuggestion(
                layers, "caTheta", worst_threshold, worst_theta,
                len(theta_fails), worst_theta))

        # DCA failures
        dca_fails = [c for c in checks if not c['dca_passed']]
        if dca_fails:
            worst_dca = max(c['dca_value'] for c in dca_fails)
            worst_threshold = max(c['dca_threshold'] for c in dca_fails)
            current_dca = dca_fails[0]['dca_cut']
            suggestions.append(TripletCutSuggestion(
                layers, "caDCA", worst_threshold, worst_dca,
                len(dca_fails), worst_dca))

    return suggestions


def _format_config_row(row, highlight_cols=None):
    """Format a layerPairs config row as a Python list literal.

    highlight_cols is an optional set of column indices that changed.
    """
    parts = []
    for i, val in enumerate(row):
        if isinstance(val, bool):
            parts.append(f"{str(val):>5}")
        elif isinstance(val, int):
            parts.append(f"{val:>6}")
        elif isinstance(val, float):
            if abs(val) >= 100:
                parts.append(f"{val:>7.0f}")
            elif abs(val) >= 1:
                parts.append(f"{val:>7.1f}")
            else:
                parts.append(f"{val:>7.2f}")
        else:
            parts.append(f"{val!r:>6}")
    return "[" + ", ".join(parts) + "]"


def _format_layer_row(row):
    """Format a layers config row as a Python list literal."""
    parts = []
    for i, val in enumerate(row):
        if isinstance(val, bool):
            parts.append(f"{str(val):>5}")
        elif isinstance(val, int):
            parts.append(f"{val:>5}")
        elif isinstance(val, float):
            parts.append(f"{val:>7.3f}")
        else:
            parts.append(f"{val!r:>5}")
    return "[" + ", ".join(parts) + "]"


def generate_config_snippets(doublet_suggestions, triplet_suggestions,
                             layer_pairs_table, layers_table,
                             pair_id_to_config, layer_map,
                             global_pair_layers,
                             margin=DEFAULT_SAFETY_MARGIN):
    """Generate ready-to-use configuration snippets from aggregated suggestions.

    Args:
        doublet_suggestions: dict of (pair_id, cut_name) -> list of CutSuggestion
        triplet_suggestions: dict of (layer_triple, cut_name) -> list of TripletCutSuggestion
        layer_pairs_table: parsed layerPairs config table (list of lists)
        layers_table: parsed layers config table (list of lists)
        pair_id_to_config: mapping from pair_id to config table row index
        layer_map: CA layer map
        global_pair_layers: mapping from pair_id to (inner_layer, outer_layer)
        margin: safety margin fraction (default 10%)

    Returns a list of strings (lines to print).
    """
    lines = []

    # --- layerPairs snippets ---
    # Group suggestions by pair_id to produce one row per affected pair
    pair_suggestions = defaultdict(dict)  # {pair_id: {cut_name: (suggested_val, current_val, bound_type, n_recoverable)}}
    for (pid, cut_name), sugg_list in doublet_suggestions.items():
        bound_type = CUT_BOUND_TYPE.get(cut_name, "?")
        if bound_type == "upper":
            worst = max(sugg_list, key=lambda s: s.suggested_threshold)
        elif bound_type == "lower":
            worst = min(sugg_list, key=lambda s: s.suggested_threshold)
        else:
            continue
        total_recoverable = sum(s.n_recoverable for s in sugg_list)
        pair_suggestions[pid][cut_name] = (
            worst.suggested_threshold, worst.current_threshold,
            bound_type, total_recoverable)

    if pair_suggestions and layer_pairs_table is not None:
        lines.append("")
        lines.append(f"  SUGGESTED CONFIGURATION VALUES (with {int(margin*100)}% safety margin):")
        lines.append(f"  {'.' * 72}")
        lines.append(f"  layerPairs columns:")
        lines.append(f"  #  i,  o, start, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts, stubSigma")
        lines.append("")

        for pid in sorted(pair_suggestions.keys()):
            cuts = pair_suggestions[pid]
            config_idx = pair_id_to_config.get(pid)
            if config_idx is None or config_idx >= len(layer_pairs_table):
                continue

            current_row = list(layer_pairs_table[config_idx])
            suggested_row = list(current_row)  # copy
            changes = []

            for cut_name, (suggested_val, current_val, bound_type, n_rec) in cuts.items():
                col = CUT_TO_LAYER_PAIR_COLUMN.get(cut_name)
                if col is None or col >= len(current_row):
                    continue
                # Apply safety margin
                margined_val = _apply_safety_margin(
                    current_val, suggested_val, bound_type, margin)
                # Round to reasonable precision
                margined_val = _round_config_value(margined_val, current_row[col])
                suggested_row[col] = margined_val
                col_name = LAYER_PAIR_COL_NAMES[col] if col < len(LAYER_PAIR_COL_NAMES) else f"col{col}"
                changes.append(
                    f"{col_name} {_fmt_config_val(current_row[col])}->{_fmt_config_val(margined_val)}"
                    f" ({n_rec} doublets)")

            if not changes:
                continue

            # Format output
            if pid in global_pair_layers:
                il, ol = global_pair_layers[pid]
                in_n = layer_name(il, layer_map)
                out_n = layer_name(ol, layer_map)
                pair_label = f"Pair {pid} ({in_n} -> {out_n})"
            else:
                pair_label = f"Pair {pid}"

            lines.append(f"  {pair_label}:")
            lines.append(f"    Current:   {_format_config_row(current_row)}")
            lines.append(f"    Suggested: {_format_config_row(suggested_row)}")
            lines.append(f"    Changes: {', '.join(changes)}")
            lines.append("")

    # --- layers (caTheta/caDCA) snippets ---
    # Group triplet suggestions: for caTheta, the affected layer is the middle layer
    # (where the cut is applied); for caDCA, it's also the middle layer.
    # But the caTheta/caDCA config is per-layer, and the actual cut depends on geometry.
    # We need to suggest which layer's cut to increase.
    layer_cut_suggestions = defaultdict(dict)
    # {layer_idx: {cut_name: (worst_actual, worst_threshold, current_cut, n_recoverable)}}
    for (layer_triple, cut_name), sugg_list in triplet_suggestions.items():
        # The cut is applied based on the middle layer's config
        # For caTheta: theta_cut comes from layers[middle_layer][3]
        # For caDCA: dca_cut comes from layers[middle_layer][2]
        middle_layer = layer_triple[1]
        worst = max(sugg_list, key=lambda s: s.suggested_threshold)
        total_recoverable = sum(s.n_recoverable for s in sugg_list)
        # Current cut from config
        if layers_table is not None and middle_layer < len(layers_table):
            col = CUT_TO_LAYER_COLUMN.get(cut_name)
            if col is not None:
                current_config = layers_table[middle_layer][col]
                # The suggested_threshold is the worst raw value that failed;
                # the current_threshold is the computed threshold (after geometry factors).
                # We need to scale the config value proportionally.
                # Ratio: new_config = current_config * (worst_actual / worst_threshold)
                # (since threshold = config * geometry_factor)
                if worst.current_threshold > 0:
                    scale = worst.suggested_threshold / worst.current_threshold
                    suggested_config = current_config * scale
                else:
                    suggested_config = current_config
                layer_cut_suggestions[middle_layer][cut_name] = (
                    worst.suggested_threshold, worst.current_threshold,
                    current_config, suggested_config, total_recoverable)

    if layer_cut_suggestions and layers_table is not None:
        if pair_suggestions and layer_pairs_table is not None:
            lines.append(f"  {'.' * 72}")
        else:
            lines.append("")
            lines.append(f"  SUGGESTED CONFIGURATION VALUES (with {int(margin*100)}% safety margin):")
            lines.append(f"  {'.' * 72}")
        lines.append(f"  layers (caLayers) columns:")
        lines.append(f"  # index, isBarrel, caDCA, caTheta")
        lines.append("")

        for layer_idx in sorted(layer_cut_suggestions.keys()):
            cuts = layer_cut_suggestions[layer_idx]
            if layer_idx >= len(layers_table):
                continue

            current_row = list(layers_table[layer_idx])
            suggested_row = list(current_row)
            changes = []

            for cut_name, (worst_actual, worst_thresh, current_config, suggested_config, n_rec) in cuts.items():
                col = CUT_TO_LAYER_COLUMN.get(cut_name)
                if col is None:
                    continue
                margined_config = _apply_safety_margin(
                    current_config, suggested_config, "upper", margin)
                margined_config = _round_config_value(margined_config, current_row[col])
                suggested_row[col] = margined_config
                col_name = LAYER_COL_NAMES[col] if col < len(LAYER_COL_NAMES) else f"col{col}"
                changes.append(
                    f"{col_name} {_fmt_config_val(current_row[col])}->{_fmt_config_val(margined_config)}"
                    f" ({n_rec} triplets)")

            if not changes:
                continue

            lname = layer_name(layer_idx, layer_map)
            lines.append(f"  Layer {layer_idx} ({lname}):")
            lines.append(f"    Current:   {_format_layer_row(current_row)}")
            lines.append(f"    Suggested: {_format_layer_row(suggested_row)}")
            lines.append(f"    Changes: {', '.join(changes)}")
            lines.append("")

    return lines


def _identify_missing_connections(ev, layer_map):
    """Identify layer pairs where all doublets were killed, which could
    prevent triplets or longer tracks from forming.

    Returns a list of (pair_id, inner_layer, outer_layer, n_kills, dominant_code)
    for pairs that have kills but zero surviving doublets.
    """
    missing = []
    for pair_id in sorted(ev.kills_by_pair.keys()):
        if pair_id < 0:
            continue
        # Check if this pair has any surviving doublets
        n_doublets = len(ev.doublets_by_pair.get(pair_id, []))
        if n_doublets > 0:
            continue
        # This pair had all its candidates killed
        codes = ev.kills_by_pair[pair_id]
        total_kills = sum(codes.values())
        if total_kills == 0:
            continue
        dominant_code = max(codes, key=codes.get)
        if pair_id in ev.pair_layer_map:
            il, ol = ev.pair_layer_map[pair_id]
        else:
            il, ol = -1, -1
        missing.append((pair_id, il, ol, total_kills, dominant_code))
    return missing


def _find_affected_triplets(missing_pairs, ev, layer_map):
    """Given layer pairs with no surviving doublets, identify which layer
    triplets cannot form because they need one of these pairs.

    A triplet on layers (A, B, C) requires doublets on pairs (A->B) and (B->C).
    If either pair is missing, the triplet cannot form.

    Returns a list of (layer_triple, missing_pair_id, missing_pair_layers) tuples.
    """
    # Build set of missing pair layer tuples
    missing_pair_layers = set()
    pair_id_for_layers = {}
    for pid, il, ol, _, _ in missing_pairs:
        if il >= 0 and ol >= 0:
            missing_pair_layers.add((il, ol))
            pair_id_for_layers[(il, ol)] = pid

    if not missing_pair_layers:
        return []

    # Build set of all layer pairs that DO have doublets
    active_pairs = set()
    for pair_id, doublets in ev.doublets_by_pair.items():
        if doublets and pair_id in ev.pair_layer_map:
            il, ol = ev.pair_layer_map[pair_id]
            active_pairs.add((il, ol))

    # For each missing pair (A->B), check if there exists an active pair
    # (B->C) or (X->A) that would have formed a triplet
    affected = []
    for (mil, mol) in missing_pair_layers:
        pid = pair_id_for_layers[(mil, mol)]
        # Missing pair is inner doublet of a triplet: need (mol -> C)
        for (ail, aol) in active_pairs:
            if ail == mol:
                affected.append(((mil, mol, aol), pid, (mil, mol)))
        # Missing pair is outer doublet of a triplet: need (X -> mil)
        for (ail, aol) in active_pairs:
            if aol == mil:
                affected.append(((ail, mil, mol), pid, (mil, mol)))

    return affected


# ---------------------------------------------------------------------------
# Printing routines
# ---------------------------------------------------------------------------

KILL_CODE_DESCRIPTIONS = {
    1: "invalid detector index",
    2: "z/inner value out of range",
    4: "module index invalid / cluster size cut",
    5: "outer value / dz / z0 cut",
    6: "phi cut (idphi > iphicut)",
    7: "z-size cut",
    8: "pt cut",
    9: "stub direction inconsistent",
    10: "stub sigma cut",
}


def print_header(header_info, layer_map):
    """Print global configuration header."""
    print("=" * 80)
    print("  CA Tracking Log Summary")
    print("=" * 80)
    if header_info:
        if "n_layers" in header_info:
            print(f"  Layers: {header_info['n_layers']}")
        if "n_pairs" in header_info:
            print(f"  Layer pairs: {header_info['n_pairs']}")
    if layer_map:
        print(f"\n  Layer map ({len(layer_map)} CA layers):")
        for ca_layer in sorted(layer_map.keys()):
            subsys, subdet_layer, mod_start, is_barrel = layer_map[ca_layer]
            if is_barrel:
                btype = "Barrel"
            else:
                sign = _endcap_sign(ca_layer, subsys, layer_map)
                btype = f"Endcap {sign}z"
            print(f"    CA {ca_layer:>2} -> {subsys} {btype} L{subdet_layer} "
                  f"(module start: {mod_start})")
    print()


def print_event_summary(ev, layer_map, verbose=False, ot_ev=None, full=False,
                        suggest_cuts=False, layer_pairs_table=None,
                        layers_table=None, pair_id_to_config=None,
                        safety_margin=DEFAULT_SAFETY_MARGIN):
    """Print a summary for a single event."""
    print(f"\n{'=' * 80}")
    print(f"=== Event {ev.event} (Run {ev.run}, Record {ev.record_number}, "
          f"LumiSection {ev.lumi}) ===")
    print(f"{'=' * 80}")

    # --- OT Stub Formation data (from optional second log) ---
    if ot_ev is not None:
        ot_layer_starts = _build_ot_layer_starts(layer_map) if layer_map else []
        print(f"\n  OT Stub Formation:")
        print(f"    OT RecHits:          {ot_ev.n_ot_rechits}")
        print(f"    Bend-based stubs:    {ot_ev.n_bend_stubs}")
        print(f"    VH-style stubs:      {ot_ev.n_vh_style_stubs}")
        print(f"    VectorHits (CPU):    {ot_ev.n_vh_accepted} accepted, "
              f"{ot_ev.n_vh_rejected} rejected "
              f"({ot_ev.n_vh_output} output)")

        # Per-OT-layer breakdown for VH-style stubs (primary)
        if ot_ev.vh_style_stubs and ot_layer_starts:
            by_layer = _stubs_by_ot_layer(
                ot_ev.vh_style_stubs, ot_layer_starts, layer_map)
            print(f"\n    VH-style stubs by OT layer:")
            for ca_layer, stubs in by_layer.items():
                lname = layer_name(ca_layer, layer_map)
                avg_pt = sum(s['ptEst'] for s in stubs) / len(stubs)
                print(f"      {lname:>9}: {len(stubs):>3} stubs  "
                      f"(avg ptEst={avg_pt:.1f} GeV)")

        # Per-OT-layer breakdown for bend-based stubs (only if different)
        if ot_ev.bend_stubs and ot_layer_starts:
            if ot_ev.n_bend_stubs != ot_ev.n_vh_style_stubs:
                by_layer = _stubs_by_ot_layer(
                    ot_ev.bend_stubs, ot_layer_starts, layer_map)
                print(f"\n    Bend-based stubs by OT layer:")
                for ca_layer, stubs in by_layer.items():
                    lname = layer_name(ca_layer, layer_map)
                    avg_pt = sum(s['ptEst'] for s in stubs) / len(stubs)
                    print(f"      {lname:>9}: {len(stubs):>3} stubs  "
                          f"(avg ptEst={avg_pt:.1f} GeV)")

        # Per-module stub counts (verbose)
        # CountStubs uses OT-relative module indices (0-13199);
        # add OT_BARREL_START to convert to CA module index.
        if verbose and ot_ev.module_stub_counts:
            print(f"\n    Per-module stub counts ({len(ot_ev.module_stub_counts)} modules):")
            for mod_idx in sorted(ot_ev.module_stub_counts.keys()):
                n_stubs, max_bend = ot_ev.module_stub_counts[mod_idx]
                ca_idx = mod_idx + OT_BARREL_START
                ca_layer = _detidx_to_ot_layer(ca_idx, ot_layer_starts)
                lname = layer_name(ca_layer, layer_map) if ca_layer else f"OT?{mod_idx}"
                print(f"      Module {mod_idx:>5} ({lname:>9}): "
                      f"{n_stubs} stubs (maxBend={max_bend:.6f})")

        # Individual stub details (verbose)
        if verbose and ot_ev.vh_style_stubs:
            limit = None if full else 20
            stubs_to_show = ot_ev.vh_style_stubs if full else ot_ev.vh_style_stubs[:20]
            print(f"\n    VH-style stub details{'' if full else ' (first 20)'}:")
            for s in stubs_to_show:
                ca_layer = _detidx_to_ot_layer(s['detIdx'], ot_layer_starts)
                lname = layer_name(ca_layer, layer_map) if ca_layer else "?"
                print(f"      Stub {s['idx']:>3} [{lname:>9}]: "
                      f"r={s['r']:>8.2f} z={s['z']:>8.2f} "
                      f"bend={s['bend']:>10.6f} ptEst={s['ptEst']:>7.1f} "
                      f"hits=({s['innerHit']},{s['outerHit']})")
            if not full and len(ot_ev.vh_style_stubs) > 20:
                print(f"      ... and {len(ot_ev.vh_style_stubs) - 20} more")

    # --- Doublet building overview ---
    total_doublets = sum(len(d) for d in ev.doublets_by_pair.values())
    print(f"\nDoublet building: {ev.n_hits} hits, "
          f"{total_doublets} doublets created")
    if ev.pt_min > 0:
        print(f"  Cuts: z0Cut={ev.cell_z0_cut}, ptMin={ev.pt_min}")

    # --- Doublets by layer pair ---
    if ev.doublets_by_pair:
        print(f"\n  Doublets by layer pair:")
        for pair_id in sorted(ev.doublets_by_pair.keys()):
            doublets = ev.doublets_by_pair[pair_id]
            if pair_id in ev.pair_layer_map:
                inner_l, outer_l = ev.pair_layer_map[pair_id]
                inner_name = layer_name(inner_l, layer_map)
                outer_name = layer_name(outer_l, layer_map)
                print(f"    Pair {pair_id:>3} ({inner_name:>9} -> {outer_name:<9}): "
                      f"{len(doublets):>4} doublets")
            else:
                print(f"    Pair {pair_id:>3} (unknown layers): "
                      f"{len(doublets):>4} doublets")
            if verbose:
                for d in doublets:
                    print(f"      doublet {d[0]}: hit {d[3]} -> hit {d[4]}")

    # --- Also show layer pairs that were explored but produced no doublets ---
    explored_no_doublets = set(ev.pair_layer_map.keys()) - set(ev.doublets_by_pair.keys())
    if explored_no_doublets and verbose:
        print(f"\n  Layer pairs explored with 0 doublets:")
        for pair_id in sorted(explored_no_doublets):
            inner_l, outer_l = ev.pair_layer_map[pair_id]
            inner_name = layer_name(inner_l, layer_map)
            outer_name = layer_name(outer_l, layer_map)
            print(f"    Pair {pair_id:>3} ({inner_name:>9} -> {outer_name:<9})")

    # --- Kill statistics during doublet formation ---
    if ev.kills:
        total_kills = sum(ev.kills.values())
        print(f"\n  Doublet kills: {total_kills} total")
        for code in sorted(ev.kills.keys()):
            desc = KILL_CODE_DESCRIPTIONS.get(code, f"unknown code {code}")
            print(f"    Code {code} ({desc}): {ev.kills[code]}")

        # Show kills broken down by layer pair
        if ev.kills_by_pair:
            print(f"\n  Kills by layer pair:")
            for pair_id in sorted(ev.kills_by_pair.keys()):
                codes = ev.kills_by_pair[pair_id]
                total_pair_kills = sum(codes.values())
                if pair_id in ev.pair_layer_map:
                    inner_l, outer_l = ev.pair_layer_map[pair_id]
                    inner_n = layer_name(inner_l, layer_map)
                    outer_n = layer_name(outer_l, layer_map)
                    pair_label = f"Pair {pair_id:>3} ({inner_n:>9} -> {outer_n:<9})"
                elif pair_id >= 0:
                    pair_label = f"Pair {pair_id:>3} (unknown layers          )"
                else:
                    pair_label = f"Pair  ?? (no context                )"
                codes_str = ", ".join(
                    f"code {c}:{n}" for c, n in sorted(codes.items())
                )
                print(f"    {pair_label}: {total_pair_kills:>3} kills ({codes_str})")

        # Verbose: individual kill details
        if verbose:
            for code in sorted(ev.kill_details.keys()):
                details = ev.kill_details[code]
                if not details:
                    continue
                desc = KILL_CODE_DESCRIPTIONS.get(code, f"unknown code {code}")
                print(f"\n  Kill details - Code {code} ({desc}):")
                details_to_show = details if full else details[:10]
                for pid, il, ol, detail_str in details_to_show:
                    if pid >= 0 and pid in ev.pair_layer_map:
                        il_actual, ol_actual = ev.pair_layer_map[pid]
                        in_n = layer_name(il_actual, layer_map)
                        out_n = layer_name(ol_actual, layer_map)
                        pair_str = f"pair {pid:>3} ({in_n}->{out_n})"
                    elif pid >= 0:
                        pair_str = f"pair {pid:>3} (L{il}->L{ol})"
                    else:
                        pair_str = "pair ??"
                    if detail_str:
                        print(f"      [{pair_str}] {detail_str}")
                    else:
                        print(f"      [{pair_str}]")
                if not full and len(details) > 10:
                    print(f"      ... and {len(details) - 10} more")

    # --- Triplets / cells ---
    if ev.triplets:
        print(f"\n  Triplets: {len(ev.triplets)}")
        if verbose:
            triplets_to_show = ev.triplets if full else ev.triplets[:20]
            for t in triplets_to_show:
                triplet_no, cut1, cut2, pair_a, pair_b, inner_d, outer_d, linfo = t
                print(f"    Triplet {triplet_no:>3}: pairs ({pair_a},{pair_b}) "
                      f"doublets {inner_d}->{outer_d} "
                      f"layers ({linfo[0]},{linfo[1]},{linfo[2]},{linfo[3]})")
            if not full and len(ev.triplets) > 20:
                print(f"    ... and {len(ev.triplets) - 20} more")

    if ev.cells:
        print(f"  Cells filled: {len(ev.cells)}")

    # --- Triplet Formation Debug (CA_DEBUG) ---
    if ev.triplet_checks:
        total_checks = len(ev.triplet_checks)
        passed = sum(1 for c in ev.triplet_checks if c['overall_passed'])
        failed_theta = sum(1 for c in ev.triplet_checks if not c['theta_passed'] and c['dca_passed'])
        failed_dca = sum(1 for c in ev.triplet_checks if c['theta_passed'] and not c['dca_passed'])
        failed_both = sum(1 for c in ev.triplet_checks if not c['theta_passed'] and not c['dca_passed'])

        print(f"\n  Triplet Formation Checks: {total_checks} total")
        print(f"    Passed:           {passed:>6} ({100*passed/total_checks:.1f}%)")
        print(f"    Failed theta:     {failed_theta:>6} ({100*failed_theta/total_checks:.1f}%)")
        print(f"    Failed DCA:       {failed_dca:>6} ({100*failed_dca/total_checks:.1f}%)")
        print(f"    Failed both:      {failed_both:>6} ({100*failed_both/total_checks:.1f}%)")

        # Count by stub type combinations
        stub_combos = defaultdict(lambda: {'passed': 0, 'failed': 0})
        for c in ev.triplet_checks:
            combo = f"{c['stub_type1']}-{c['stub_type2']}-{c['stub_type3']}"
            if c['overall_passed']:
                stub_combos[combo]['passed'] += 1
            else:
                stub_combos[combo]['failed'] += 1

        print(f"\n    By stub type combination:")
        for combo in sorted(stub_combos.keys()):
            stats = stub_combos[combo]
            total = stats['passed'] + stats['failed']
            pct = 100 * stats['passed'] / total if total > 0 else 0
            print(f"      {combo:>18}: {stats['passed']:>5} passed / {stats['failed']:>5} failed ({pct:.1f}% pass rate)")

        # Summary by layer triplet
        if ev.triplet_check_summary:
            print(f"\n    By layer triplet:")
            for layers in sorted(ev.triplet_check_summary.keys()):
                stats = ev.triplet_check_summary[layers]
                total = stats['passed'] + stats['failed_theta'] + stats['failed_dca'] + stats['failed_both']
                pct = 100 * stats['passed'] / total if total > 0 else 0
                l1_name = layer_name(layers[0], layer_map)
                l2_name = layer_name(layers[1], layer_map)
                l3_name = layer_name(layers[2], layer_map)
                print(f"      {l1_name:>9} -> {l2_name:>9} -> {l3_name:<9}: "
                      f"{stats['passed']:>4} pass, {stats['failed_theta']:>4} theta, "
                      f"{stats['failed_dca']:>4} DCA, {stats['failed_both']:>4} both ({pct:.1f}% pass)")

        # Verbose: individual check details
        if verbose:
            # Show failed checks with stubs (most interesting for debugging)
            failed_with_stubs = [c for c in ev.triplet_checks
                                 if not c['overall_passed'] and
                                 (c['stub_type1'] != 'pixel' or c['stub_type2'] != 'pixel' or c['stub_type3'] != 'pixel')]
            if failed_with_stubs:
                print(f"\n    Failed triplet checks involving stubs:")
                checks_to_show = failed_with_stubs if full else failed_with_stubs[:20]
                for c in checks_to_show:
                    l1_name = layer_name(c['layer1'], layer_map)
                    l2_name = layer_name(c['layer2'], layer_map)
                    l3_name = layer_name(c['layer3'], layer_map)
                    reason = []
                    if not c['theta_passed']:
                        reason.append(f"theta ({c['theta_value']:.4f} > {c['theta_threshold']:.4f})")
                    if not c['dca_passed']:
                        reason.append(f"DCA ({c['dca_value']:.4f} > {c['dca_threshold']:.4f})")
                    reason_str = ", ".join(reason)
                    stub_str = f"{c['stub_type1']}-{c['stub_type2']}-{c['stub_type3']}"
                    print(f"      cells ({c['inner_cell']},{c['outer_cell']}) "
                          f"layers ({l1_name}->{l2_name}->{l3_name}) "
                          f"stubs [{stub_str}]: {reason_str}")
                if not full and len(failed_with_stubs) > 20:
                    print(f"      ... and {len(failed_with_stubs) - 20} more")

    # --- Fishbone cleaning ---
    if ev.fishbone_kills:
        n_same = sum(1 for f in ev.fishbone_kills if f[5] == 'same')
        n_diff = sum(1 for f in ev.fishbone_kills if f[5] == 'diff')
        print(f"\n  Fishbone cleaning: {len(ev.fishbone_kills)} kills "
              f"({n_same} same-layer, {n_diff} cross-layer)")
        if verbose:
            kills_to_show = ev.fishbone_kills if full else ev.fishbone_kills[:10]
            for fk in kills_to_show:
                killer, killed, cos_val, n1, n2, sd = fk
                print(f"    cell {killer:>3} kills cell {killed:>3} "
                      f"(cos={cos_val:.7f}, n1={n1:.1f}, n2={n2:.1f}, {sd})")
            if not full and len(ev.fishbone_kills) > 10:
                print(f"    ... and {len(ev.fishbone_kills) - 10} more")

    # --- n-tuplet building ---
    if ev.n_cells_for_ntuplets > 0 or ev.n_triplets_for_ntuplets > 0:
        print(f"\n  Ntuplet input: {ev.n_cells_for_ntuplets} cells, "
              f"{ev.n_triplets_for_ntuplets} triplets")

    # --- Tracks ---
    if ev.tracks:
        doublet_layers = build_doublet_layers(ev)
        total_pixel_hits = 0
        total_ot_stubs = 0
        print(f"\n  Tracks found (raw): {len(ev.tracks)}")
        for t in ev.tracks:
            track_idx, nhits, cell_ids = t
            layers = get_track_layers(cell_ids, doublet_layers)
            cell_str = " -> ".join(str(c) for c in cell_ids)
            if layers is not None:
                n_pix, n_ot, expanded = count_hits(layers, layer_map)
                total_pixel_hits += n_pix
                total_ot_stubs += n_ot
                layer_str = " -> ".join(layer_name(l, layer_map) for l in layers)
                print(f"    Track {track_idx:>3}: {len(layers)} layers, "
                      f"{expanded} hits ({n_pix} pixel + {n_ot} stub x2), "
                      f"cells [{cell_str}]")
                print(f"             {layer_str}")
            else:
                print(f"    Track {track_idx:>3}: {nhits} hits, "
                      f"cells [{cell_str}]")
        n_tracks = len(ev.tracks)
        total_expanded = total_pixel_hits + total_ot_stubs * 2
        print(f"  Track hit summary: {total_pixel_hits} pixel hits + "
              f"{total_ot_stubs} stubs (x2) = {total_expanded} total hits "
              f"across {n_tracks} tracks")

    # --- Track quality (nLayers) ---
    if ev.track_nlayers and verbose:
        print(f"\n  Track quality (nLayers):")
        for tnl in ev.track_nlayers:
            track_idx, total, nlayers = tnl
            print(f"    Track {track_idx:>3}: {nlayers} unique layers "
                  f"(of {total} total)")

    # --- nSizes ---
    if ev.n_sizes:
        labels = ["nHits", "nDoublets", "nCells", "nTriplets",
                   "nTracks(raw)", "nTracks(final)", "maxDoublets"]
        print(f"\n  Pipeline sizes:")
        for i, val in enumerate(ev.n_sizes):
            label = labels[i] if i < len(labels) else f"field_{i}"
            print(f"    {label:>20}: {val}")

    # --- Final summary ---
    print(f"\n  Total tracks (final): {ev.total_tracks}")

    # --- Barrel event summary ---
    if ev.eta_pos_tracks > 0 or ev.eta_neg_tracks > 0:
        print(f"\n  Barrel summary (|eta| < 1.0):")
        print(f"    eta > 0: {ev.eta_pos_tracks} tracks, "
              f"{ev.eta_pos_issues} with issues")
        print(f"    eta < 0: {ev.eta_neg_tracks} tracks, "
              f"{ev.eta_neg_issues} with issues")

    # --- Cut suggestions ---
    if suggest_cuts:
        _print_cut_suggestions(ev, layer_map, full=full,
                               layer_pairs_table=layer_pairs_table,
                               layers_table=layers_table,
                               pair_id_to_config=pair_id_to_config,
                               safety_margin=safety_margin)


def _print_cut_suggestions(ev, layer_map, full=False,
                           layer_pairs_table=None, layers_table=None,
                           pair_id_to_config=None,
                           safety_margin=DEFAULT_SAFETY_MARGIN):
    """Print cut relaxation suggestions for a single event.

    If layer_pairs_table and layers_table are provided (from parsing the config
    file), also shows the config column index and a suggested value with safety
    margin.
    """
    doublet_sugg, triplet_sugg = analyze_cut_suggestions(ev, layer_map)
    missing_pairs = _identify_missing_connections(ev, layer_map)
    affected_triplets = _find_affected_triplets(missing_pairs, ev, layer_map)

    has_any = (doublet_sugg or triplet_sugg or missing_pairs)
    if not has_any:
        return

    print(f"\n  {'~' * 72}")
    print(f"  CUT RELAXATION SUGGESTIONS")
    print(f"  {'~' * 72}")

    # --- Missing connections (pairs with zero surviving doublets) ---
    if missing_pairs:
        print(f"\n  Layer pairs with ALL candidates killed ({len(missing_pairs)} pairs):")
        for pid, il, ol, n_kills, dom_code in missing_pairs:
            if il >= 0:
                in_n = layer_name(il, layer_map)
                out_n = layer_name(ol, layer_map)
                pair_str = f"Pair {pid:>3} ({in_n:>9} -> {out_n:<9})"
            else:
                pair_str = f"Pair {pid:>3}"
            dom_desc = KILL_CODE_DESCRIPTIONS.get(dom_code, f"code {dom_code}")
            print(f"    {pair_str}: {n_kills} kills "
                  f"(dominant: {dom_desc})")

        if affected_triplets:
            print(f"\n  Triplets blocked by missing doublet pairs:")
            shown = affected_triplets if full else affected_triplets[:20]
            for (l1, l2, l3), pid, (mil, mol) in shown:
                l1_n = layer_name(l1, layer_map)
                l2_n = layer_name(l2, layer_map)
                l3_n = layer_name(l3, layer_map)
                mil_n = layer_name(mil, layer_map)
                mol_n = layer_name(mol, layer_map)
                print(f"    Triplet [{l1_n} -> {l2_n} -> {l3_n}] "
                      f"blocked: pair {pid} ({mil_n} -> {mol_n}) has no doublets")
            if not full and len(affected_triplets) > 20:
                print(f"    ... and {len(affected_triplets) - 20} more")

    # --- Doublet cut suggestions ---
    if doublet_sugg:
        print(f"\n  Doublet cut relaxation suggestions ({len(doublet_sugg)} cuts):")
        for s in doublet_sugg:
            if s.inner_layer >= 0 and s.pair_id in ev.pair_layer_map:
                il_actual, ol_actual = ev.pair_layer_map[s.pair_id]
                in_n = layer_name(il_actual, layer_map)
                out_n = layer_name(ol_actual, layer_map)
                pair_str = f"pair {s.pair_id:>3} ({in_n:>9} -> {out_n:<9})"
            elif s.pair_id >= 0:
                pair_str = f"pair {s.pair_id:>3}"
            else:
                pair_str = "pair ??"
            config_name = CUT_CONFIG_NAMES.get(s.cut_name, s.cut_name)
            bound_type = CUT_BOUND_TYPE.get(s.cut_name, "?")
            margin = s.margin_pct()
            margin_str = f" ({margin:.1f}% change)" if margin is not None else ""

            # Config column info
            col = CUT_TO_LAYER_PAIR_COLUMN.get(s.cut_name)
            col_str = ""
            margined_str = ""
            if col is not None:
                col_name = LAYER_PAIR_COL_NAMES[col] if col < len(LAYER_PAIR_COL_NAMES) else f"col{col}"
                col_str = f" [config col {col}: {col_name}]"
                # Compute margined value if config table is available
                if (layer_pairs_table is not None and pair_id_to_config is not None
                        and s.pair_id in pair_id_to_config):
                    cfg_idx = pair_id_to_config[s.pair_id]
                    if cfg_idx < len(layer_pairs_table):
                        ref_val = layer_pairs_table[cfg_idx][col]
                        margined = _apply_safety_margin(
                            s.current_threshold, s.suggested_threshold,
                            bound_type, safety_margin)
                        margined = _round_config_value(margined, ref_val)
                        margined_str = f" -> suggest config: {_fmt_number(margined)}"

            if bound_type == "upper":
                direction = "increase"
                fmt_current = _fmt_number(s.current_threshold)
                fmt_suggest = _fmt_number(s.suggested_threshold)
                print(f"    [{pair_str}] {s.cut_name}: "
                      f"{direction} from {fmt_current} to >= {fmt_suggest}"
                      f"{margin_str}{margined_str}  "
                      f"({s.n_recoverable} doublets recoverable)"
                      f"{col_str}")
            elif bound_type == "lower":
                direction = "decrease"
                fmt_current = _fmt_number(s.current_threshold)
                fmt_suggest = _fmt_number(s.suggested_threshold)
                print(f"    [{pair_str}] {s.cut_name}: "
                      f"{direction} from {fmt_current} to <= {fmt_suggest}"
                      f"{margin_str}{margined_str}  "
                      f"({s.n_recoverable} doublets recoverable)"
                      f"{col_str}")

    # --- Triplet cut suggestions ---
    if triplet_sugg:
        print(f"\n  Triplet cut relaxation suggestions ({len(triplet_sugg)} cuts):")
        for s in triplet_sugg:
            l1_n = layer_name(s.layer_triple[0], layer_map)
            l2_n = layer_name(s.layer_triple[1], layer_map)
            l3_n = layer_name(s.layer_triple[2], layer_map)
            triple_str = f"{l1_n} -> {l2_n} -> {l3_n}"
            fmt_current = _fmt_number(s.current_threshold)
            fmt_suggest = _fmt_number(s.suggested_threshold)

            # Config column info for triplet cuts
            col = CUT_TO_LAYER_COLUMN.get(s.cut_name)
            col_str = ""
            if col is not None:
                col_name = LAYER_COL_NAMES[col] if col < len(LAYER_COL_NAMES) else f"col{col}"
                middle_layer = s.layer_triple[1]
                ml_name = layer_name(middle_layer, layer_map)
                col_str = f" [layers[{middle_layer}] ({ml_name}) col {col}: {col_name}]"

            print(f"    [{triple_str}] {s.cut_name}: "
                  f"increase from {fmt_current} to >= {fmt_suggest}  "
                  f"({s.n_recoverable} triplets recoverable)"
                  f"{col_str}")


def _fmt_number(val):
    """Format a number for display: use int format for integers, else float."""
    if val is None:
        return "?"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        if abs(val) < 0.001 or abs(val) > 1e6:
            return f"{val:.4e}"
        return f"{val:.4f}"
    return str(val)


def _fmt_config_val(val):
    """Format a config value concisely for the 'Changes' summary line.

    More concise than _fmt_number: strips trailing zeros from floats.
    """
    if val is None:
        return "?"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        if abs(val) < 0.001 or abs(val) > 1e6:
            return f"{val:.4e}"
        # Use enough precision but strip trailing zeros
        formatted = f"{val:.4f}".rstrip('0').rstrip('.')
        return formatted
    return str(val)


def print_global_summary(events, layer_map, ot_events=None, suggest_cuts=False,
                         layer_pairs_table=None, layers_table=None,
                         pair_id_to_config=None,
                         safety_margin=DEFAULT_SAFETY_MARGIN):
    """Print a global summary across all events."""
    print(f"\n{'=' * 80}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Total events processed: {len(events)}")

    total_tracks_all = sum(ev.total_tracks for ev in events)
    total_doublets_all = sum(
        sum(len(d) for d in ev.doublets_by_pair.values()) for ev in events
    )
    total_triplets_all = sum(len(ev.triplets) for ev in events)
    total_cells_all = sum(len(ev.cells) for ev in events)
    total_fishbone_kills = sum(len(ev.fishbone_kills) for ev in events)
    total_doublet_kills = sum(sum(ev.kills.values()) for ev in events)

    print(f"  Total tracks across all events: {total_tracks_all}")
    print(f"  Total doublets created: {total_doublets_all}")
    print(f"  Total triplets: {total_triplets_all}")
    print(f"  Total cells: {total_cells_all}")
    print(f"  Total fishbone kills: {total_fishbone_kills}")
    print(f"  Total doublet kills: {total_doublet_kills}")

    # Kill breakdown
    all_kills = defaultdict(int)
    all_kills_by_pair = defaultdict(lambda: defaultdict(int))
    global_pair_layers = {}
    for ev in events:
        for code, count in ev.kills.items():
            all_kills[code] += count
        for pair_id, codes in ev.kills_by_pair.items():
            for code, count in codes.items():
                all_kills_by_pair[pair_id][code] += count
            if pair_id in ev.pair_layer_map and pair_id not in global_pair_layers:
                global_pair_layers[pair_id] = ev.pair_layer_map[pair_id]
    if all_kills:
        print(f"\n  Doublet kill breakdown:")
        for code in sorted(all_kills.keys()):
            desc = KILL_CODE_DESCRIPTIONS.get(code, f"unknown code {code}")
            print(f"    Code {code} ({desc}): {all_kills[code]}")

        print(f"\n  Kills by layer pair (all events):")
        for pair_id in sorted(all_kills_by_pair.keys()):
            codes = all_kills_by_pair[pair_id]
            total_pair_kills = sum(codes.values())
            if pair_id in global_pair_layers:
                inner_l, outer_l = global_pair_layers[pair_id]
                inner_n = layer_name(inner_l, layer_map)
                outer_n = layer_name(outer_l, layer_map)
                pair_label = f"Pair {pair_id:>3} ({inner_n:>9} -> {outer_n:<9})"
            elif pair_id >= 0:
                pair_label = f"Pair {pair_id:>3}"
            else:
                pair_label = f"Pair  ??"
            codes_str = ", ".join(
                f"code {c}:{n}" for c, n in sorted(codes.items())
            )
            print(f"    {pair_label}: {total_pair_kills:>4} kills ({codes_str})")

    # OT stub formation global summary
    if ot_events:
        ot_map = {oe.event: oe for oe in ot_events}
        total_rechits = sum(oe.n_ot_rechits for oe in ot_events)
        total_bend = sum(oe.n_bend_stubs for oe in ot_events)
        total_vh_style = sum(oe.n_vh_style_stubs for oe in ot_events)
        total_vh_acc = sum(oe.n_vh_accepted for oe in ot_events)
        total_vh_rej = sum(oe.n_vh_rejected for oe in ot_events)
        print(f"\n  OT Stub Formation (all events):")
        print(f"    Total OT RecHits:          {total_rechits}")
        print(f"    Total bend-based stubs:    {total_bend}")
        print(f"    Total VH-style stubs:      {total_vh_style}")
        print(f"    Total VectorHits (CPU):    {total_vh_acc} accepted, "
              f"{total_vh_rej} rejected")

    # Per-event summary table
    has_ot = bool(ot_events)
    ot_map = {oe.event: oe for oe in ot_events} if ot_events else {}
    if has_ot:
        print(f"\n  {'Event':>6} {'Hits':>6} {'Doublets':>9} {'Triplets':>9} "
              f"{'Cells':>7} {'FB Kills':>9} {'Tracks':>7} "
              f"{'OTHits':>7} {'BStubs':>7} {'VHStubs':>8}")
        print(f"  {'-' * 6} {'-' * 6} {'-' * 9} {'-' * 9} "
              f"{'-' * 7} {'-' * 9} {'-' * 7} "
              f"{'-' * 7} {'-' * 7} {'-' * 8}")
    else:
        print(f"\n  {'Event':>6} {'Hits':>6} {'Doublets':>9} {'Triplets':>9} "
              f"{'Cells':>7} {'FB Kills':>9} {'Tracks':>7}")
        print(f"  {'-' * 6} {'-' * 6} {'-' * 9} {'-' * 9} "
              f"{'-' * 7} {'-' * 9} {'-' * 7}")
    for ev in events:
        n_doublets = sum(len(d) for d in ev.doublets_by_pair.values())
        n_fb_kills = len(ev.fishbone_kills)
        row = (f"  {ev.event:>6} {ev.n_hits:>6} {n_doublets:>9} "
               f"{len(ev.triplets):>9} {len(ev.cells):>7} "
               f"{n_fb_kills:>9} {ev.total_tracks:>7}")
        if has_ot:
            oe = ot_map.get(ev.event)
            if oe:
                row += (f" {oe.n_ot_rechits:>7} {oe.n_bend_stubs:>7} "
                        f"{oe.n_vh_style_stubs:>8}")
            else:
                row += f" {'?':>7} {'?':>7} {'?':>8}"
        print(row)

    # Layer pair usage across events
    pair_usage = defaultdict(int)
    pair_layers = {}
    for ev in events:
        for pair_id, doublets in ev.doublets_by_pair.items():
            pair_usage[pair_id] += len(doublets)
            if pair_id in ev.pair_layer_map and pair_id not in pair_layers:
                pair_layers[pair_id] = ev.pair_layer_map[pair_id]

    if pair_usage:
        print(f"\n  Layer pair usage (total doublets across all events):")
        for pair_id in sorted(pair_usage.keys()):
            count = pair_usage[pair_id]
            if pair_id in pair_layers:
                inner_l, outer_l = pair_layers[pair_id]
                inner_n = layer_name(inner_l, layer_map)
                outer_n = layer_name(outer_l, layer_map)
                print(f"    Pair {pair_id:>3} ({inner_n:>9} -> {outer_n:<9}): "
                      f"{count:>5} doublets")
            else:
                print(f"    Pair {pair_id:>3}: {count:>5} doublets")

    # Global cut suggestions (aggregated across all events)
    if suggest_cuts:
        _print_global_cut_suggestions(events, layer_map,
                                      layer_pairs_table=layer_pairs_table,
                                      layers_table=layers_table,
                                      pair_id_to_config=pair_id_to_config,
                                      safety_margin=safety_margin)

    print()


def _print_global_cut_suggestions(events, layer_map,
                                  layer_pairs_table=None, layers_table=None,
                                  pair_id_to_config=None,
                                  safety_margin=DEFAULT_SAFETY_MARGIN):
    """Print aggregated cut relaxation suggestions across all events.

    If layer_pairs_table and layers_table are provided (from parsing the config
    file), also produces ready-to-paste configuration snippets showing current
    vs suggested values with safety margin.
    """
    # Aggregate all kill_cut_values across events
    all_doublet_sugg = defaultdict(list)  # (pair_id, cut_name) -> list of CutSuggestion
    all_triplet_sugg = defaultdict(list)  # (layers, cut_name) -> list of TripletCutSuggestion
    total_missing_pairs = defaultdict(lambda: [0, defaultdict(int)])  # pair_id -> [total_kills, {code: count}]
    global_pair_layers = {}

    for ev in events:
        d_sugg, t_sugg = analyze_cut_suggestions(ev, layer_map)
        for s in d_sugg:
            all_doublet_sugg[(s.pair_id, s.cut_name)].append(s)
        for s in t_sugg:
            all_triplet_sugg[(s.layer_triple, s.cut_name)].append(s)

        missing = _identify_missing_connections(ev, layer_map)
        for pid, il, ol, n_kills, dom_code in missing:
            total_missing_pairs[pid][0] += n_kills
            total_missing_pairs[pid][1][dom_code] += n_kills

        for pid, (il, ol) in ev.pair_layer_map.items():
            if pid not in global_pair_layers:
                global_pair_layers[pid] = (il, ol)

    has_any = (all_doublet_sugg or all_triplet_sugg or total_missing_pairs)
    if not has_any:
        return

    print(f"\n  {'~' * 72}")
    print(f"  GLOBAL CUT RELAXATION SUGGESTIONS (across {len(events)} events)")
    print(f"  {'~' * 72}")

    # Aggregate doublet suggestions: for each (pair, cut), find worst case
    if all_doublet_sugg:
        print(f"\n  Doublet cut suggestions (worst-case across all events):")
        for (pid, cut_name) in sorted(all_doublet_sugg.keys()):
            suggestions = all_doublet_sugg[(pid, cut_name)]
            total_recoverable = sum(s.n_recoverable for s in suggestions)
            n_events_affected = len(suggestions)

            bound_type = CUT_BOUND_TYPE.get(cut_name, "?")
            if bound_type == "upper":
                worst_sugg = max(suggestions, key=lambda s: s.suggested_threshold)
            elif bound_type == "lower":
                worst_sugg = min(suggestions, key=lambda s: s.suggested_threshold)
            else:
                continue

            if pid in global_pair_layers:
                il, ol = global_pair_layers[pid]
                in_n = layer_name(il, layer_map)
                out_n = layer_name(ol, layer_map)
                pair_str = f"pair {pid:>3} ({in_n:>9} -> {out_n:<9})"
            elif pid >= 0:
                pair_str = f"pair {pid:>3}"
            else:
                pair_str = "pair ??"

            fmt_current = _fmt_number(worst_sugg.current_threshold)
            fmt_suggest = _fmt_number(worst_sugg.suggested_threshold)
            margin = worst_sugg.margin_pct()
            margin_str = f" ({margin:.1f}% change)" if margin is not None else ""

            # Config column info
            col = CUT_TO_LAYER_PAIR_COLUMN.get(cut_name)
            col_str = ""
            if col is not None:
                col_name = LAYER_PAIR_COL_NAMES[col] if col < len(LAYER_PAIR_COL_NAMES) else f"col{col}"
                col_str = f" [config col {col}: {col_name}]"

            if bound_type == "upper":
                print(f"    [{pair_str}] {cut_name}: "
                      f"increase from {fmt_current} to >= {fmt_suggest}"
                      f"{margin_str}  "
                      f"({total_recoverable} doublets in {n_events_affected} events)"
                      f"{col_str}")
            else:
                print(f"    [{pair_str}] {cut_name}: "
                      f"decrease from {fmt_current} to <= {fmt_suggest}"
                      f"{margin_str}  "
                      f"({total_recoverable} doublets in {n_events_affected} events)"
                      f"{col_str}")

    # Aggregate triplet suggestions
    if all_triplet_sugg:
        print(f"\n  Triplet cut suggestions (worst-case across all events):")
        for (layers, cut_name) in sorted(all_triplet_sugg.keys()):
            suggestions = all_triplet_sugg[(layers, cut_name)]
            total_recoverable = sum(s.n_recoverable for s in suggestions)
            n_events_affected = len(suggestions)
            worst_sugg = max(suggestions, key=lambda s: s.suggested_threshold)

            l1_n = layer_name(layers[0], layer_map)
            l2_n = layer_name(layers[1], layer_map)
            l3_n = layer_name(layers[2], layer_map)
            triple_str = f"{l1_n} -> {l2_n} -> {l3_n}"
            fmt_current = _fmt_number(worst_sugg.current_threshold)
            fmt_suggest = _fmt_number(worst_sugg.suggested_threshold)

            # Config column info for triplet cuts
            col = CUT_TO_LAYER_COLUMN.get(cut_name)
            col_str = ""
            if col is not None:
                col_name = LAYER_COL_NAMES[col] if col < len(LAYER_COL_NAMES) else f"col{col}"
                middle_layer = layers[1]
                ml_name = layer_name(middle_layer, layer_map)
                col_str = f" [layers[{middle_layer}] ({ml_name}) col {col}: {col_name}]"

            print(f"    [{triple_str}] {cut_name}: "
                  f"increase from {fmt_current} to >= {fmt_suggest}  "
                  f"({total_recoverable} triplets in {n_events_affected} events)"
                  f"{col_str}")

    # --- Ready-to-paste configuration snippets ---
    if (layer_pairs_table is not None or layers_table is not None) and pair_id_to_config is not None:
        snippet_lines = generate_config_snippets(
            all_doublet_sugg, all_triplet_sugg,
            layer_pairs_table, layers_table,
            pair_id_to_config, layer_map,
            global_pair_layers, margin=safety_margin)
        if snippet_lines:
            print(f"\n  {'~' * 72}")
            for line in snippet_lines:
                print(line)


def print_compact_summary(events):
    """Print a very compact summary (one line per event)."""
    print(f"\n{'Event':>6} {'Run':>5} {'Hits':>6} {'Doublets':>9} "
          f"{'Cells':>7} {'FB Kills':>9} {'D.Kills':>8} {'Tracks':>7}")
    print(f"{'-' * 6} {'-' * 5} {'-' * 6} {'-' * 9} "
          f"{'-' * 7} {'-' * 9} {'-' * 8} {'-' * 7}")
    for ev in events:
        n_doublets = sum(len(d) for d in ev.doublets_by_pair.values())
        n_fb = len(ev.fishbone_kills)
        n_dk = sum(ev.kills.values())
        print(f"{ev.event:>6} {ev.run:>5} {ev.n_hits:>6} {n_doublets:>9} "
              f"{len(ev.cells):>7} {n_fb:>9} {n_dk:>8} {ev.total_tracks:>7}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse CA tracking log file and produce a clean summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "logfile",
        help="Path to the CA tracking log file"
    )
    parser.add_argument(
        "--event", "-e",
        type=int, default=None,
        help="Show only this event number (CMS event ID, not record number)"
    )
    parser.add_argument(
        "--record", "-r",
        type=int, default=None,
        help="Show only this record number (1-based sequential order)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full detail (individual doublets, kill details, etc.)"
    )
    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Show only the compact per-event summary table"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip printing the global header / layer map"
    )
    parser.add_argument(
        "--no-global-summary",
        action="store_true",
        help="Skip printing the global summary across all events"
    )
    parser.add_argument(
        "--ot-log",
        default=None,
        help="Optional OT stub formation log file (same events, same order)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Print all items without truncation (no '... and N more')"
    )
    parser.add_argument(
        "--suggest-cuts",
        action="store_true",
        help="Analyze killed doublets and failed triplets to suggest cut "
             "relaxations that would recover them"
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to the Python config file (caHitNtupletAlpakaPhase2OTStubs_cfi.py) "
             "for reading current parameter values. Auto-detected if not specified. "
             "Only used with --suggest-cuts."
    )
    parser.add_argument(
        "--safety-margin",
        type=float, default=DEFAULT_SAFETY_MARGIN,
        help=f"Safety margin fraction for suggested config values "
             f"(default: {DEFAULT_SAFETY_MARGIN}, i.e. {int(DEFAULT_SAFETY_MARGIN*100)}%%)"
    )

    args = parser.parse_args()

    # Parse the CA tracking log
    try:
        header_info, layer_map, events = parse_log(args.logfile)
    except FileNotFoundError:
        print(f"Error: File not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print("No events found in the log file.", file=sys.stderr)
        sys.exit(1)

    # Parse the optional OT stub formation log
    ot_events = None
    ot_map = {}
    if args.ot_log:
        try:
            ot_events = parse_ot_stub_log(args.ot_log)
            ot_map = {oe.event: oe for oe in ot_events}
            print(f"  OT stub log: {len(ot_events)} events parsed "
                  f"from {args.ot_log}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Warning: OT log file not found: {args.ot_log}",
                  file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error parsing OT log file: {e}",
                  file=sys.stderr)

    # Parse config file for cut suggestions (if requested)
    layer_pairs_table = None
    layers_table = None
    pair_id_to_config = None
    if args.suggest_cuts:
        config_path = args.config_file
        if config_path is None:
            config_path = _find_config_path()
        if config_path is not None:
            layer_pairs_table, layers_table = parse_config_tables(config_path)
            if layer_pairs_table is not None:
                pair_id_to_config = _build_pair_id_to_config_index(
                    layer_pairs_table, events)
                n_mapped = len(pair_id_to_config)
                n_total = len(layer_pairs_table)
                print(f"  Config file: {config_path}", file=sys.stderr)
                print(f"    {n_total} layerPairs rows, {n_mapped} mapped to runtime pair IDs",
                      file=sys.stderr)
                if layers_table:
                    print(f"    {len(layers_table)} layers rows", file=sys.stderr)
            else:
                print(f"  Warning: Could not parse layerPairs from {config_path}",
                      file=sys.stderr)
        else:
            print(f"  Warning: Config file not found (use --config-file to specify)",
                  file=sys.stderr)

    # Filter events if requested
    selected_events = events
    if args.event is not None:
        selected_events = [ev for ev in events if ev.event == args.event]
        if not selected_events:
            available = sorted(set(ev.event for ev in events))
            print(f"Event {args.event} not found. Available events: {available}",
                  file=sys.stderr)
            sys.exit(1)
    elif args.record is not None:
        selected_events = [ev for ev in events if ev.record_number == args.record]
        if not selected_events:
            available = sorted(set(ev.record_number for ev in events))
            print(f"Record {args.record} not found. Available records: {available}",
                  file=sys.stderr)
            sys.exit(1)

    # Output
    if args.summary_only:
        print_compact_summary(selected_events)
        return

    if not args.no_header:
        print_header(header_info, layer_map)

    for ev in selected_events:
        ot_ev = ot_map.get(ev.event)
        print_event_summary(ev, layer_map, verbose=args.verbose, ot_ev=ot_ev,
                            full=args.full,
                            suggest_cuts=args.suggest_cuts,
                            layer_pairs_table=layer_pairs_table,
                            layers_table=layers_table,
                            pair_id_to_config=pair_id_to_config,
                            safety_margin=args.safety_margin)

    if not args.no_global_summary and len(selected_events) > 1:
        selected_ot = [ot_map[ev.event] for ev in selected_events
                       if ev.event in ot_map] if ot_map else None
        print_global_summary(selected_events, layer_map, ot_events=selected_ot,
                             suggest_cuts=args.suggest_cuts,
                             layer_pairs_table=layer_pairs_table,
                             layers_table=layers_table,
                             pair_id_to_config=pair_id_to_config,
                             safety_margin=args.safety_margin)


if __name__ == "__main__":
    main()
