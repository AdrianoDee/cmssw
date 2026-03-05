#!/usr/bin/env python3
"""
compareTrackCollections.py - Compare track collections from two separate ROOT files

Compares CA Extension (P-hits only from TOB PS barrel) vs Stubs (all OT stubs) approaches.

Usage:
    python compareTrackCollections.py caext_file.root stubs_file.root [options]

Options:
    --caext-tracks LABEL    Track collection label for CAExt (default: hltPhase2PixelTracks)
    --stubs-tracks LABEL    Track collection label for Stubs (default: hltPhase2PixelTracks)
    --caext-process PROC    Process name for CAExt (default: HLT)
    --stubs-process PROC    Process name for Stubs (default: HLT)
    --max-events N          Maximum events to process (default: all)
    --output FILE           Output text file for detailed track dump
    --match-dpt-rel VAL     Relative pT matching tolerance (default: 0.1)
    --match-deta VAL        Delta eta matching tolerance (default: 0.02)
    --match-dphi VAL        Delta phi matching tolerance (default: 0.02)
    --verbose               Print details for each unmatched track
    --diff-threshold, -d    Relative difference threshold for highlighting (default: 0.05 = 5%)

Note:
  - CAExt uses P-hits from TOB PS barrel layers 1-2 only -> fewer OT hits
  - Stubs uses inner+outer hits from all OT stubs -> more OT hits per track
  - Matching is done by shared pixel hits or kinematics

Output file format (--output):
  All lines have a consistent column structure for easy parsing with grep/awk.

  Categories (grep-friendly):
    MATCHED     - Matched track pair, values shown as caext_value(stubs_value)
    ONLY_CAEXT  - Track found only in CAExt collection (missed by stubs)
    ONLY_STUBS  - Track found only in Stubs collection (extra in stubs)

  MATCHED line format (single line per matched pair):
    EventID MATCHED pt_caext(pt_stubs) eta_caext(eta_stubs) phi_caext(phi_stubs) nHits_caext(nHits_stubs) nPixel_caext(nPixel_stubs) nOT_caext(nOT_stubs) chi2_caext(chi2_stubs) ndof_caext(ndof_stubs) match_type shared_pixels

  ONLY_CAEXT and ONLY_STUBS line format (single values, no parentheses):
    EventID Category pt eta phi nHits nPixel nOT chi2 ndof match_type shared_pixels

  Example MATCHED line:
    12345 MATCHED 5.234(5.241) 1.2345(1.2348) 0.5678(0.5681) 7(9) 4(4) 3(5) 1.234(1.567) 11.0(15.0) pixel 4

  This format makes it easy to see differences: 3(5) means CAExt has 3 OT hits, Stubs has 5.

  Example parsing:
    grep MATCHED output.txt       # All matched track pairs
    grep ONLY_CAEXT output.txt    # Tracks missed by stubs
    grep ONLY_STUBS output.txt    # Extra tracks in stubs
    less -R output.txt            # View with ANSI color highlighting

  Difference highlighting (--diff-threshold):
    When --diff-threshold is set (default 0.05 = 5%), individual values (pt, eta,
    phi, chi2) that differ by more than the threshold are highlighted in RED:
    - Only the specific value pair is colored, not the entire line
    - Console output: RED color (ANSI escape codes)
    - File output: RED color (ANSI escape codes) - use 'less -R' or 'cat' to view
    The relative difference is calculated as: |v1 - v2| / max(|v1|, |v2|, epsilon)

  Example with only chi2 exceeding threshold:
    12345 MATCHED pt=5.234(5.241) eta=1.2345(1.2348) phi=0.5678(0.5681) ... \033[91mchi2=1.234(5.567)\033[0m ...
"""

from DataFormats.FWLite import Events, Handle
import sys
import math
import argparse
from collections import defaultdict

# ANSI color codes for terminal output
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'

# Subdetector IDs (from DataFormats/SiPixelDetId/interface/PixelSubdetector.h
# and DataFormats/SiStripDetId/interface/SiStripEnums.h)
PIXEL_BARREL = 1  # PixelSubdetector::PixelBarrel
PIXEL_ENDCAP = 2  # PixelSubdetector::PixelEndcap

# Phase-2 Outer Tracker subdetector IDs (from Phase2Tracker namespace in SiStripEnums.h)
# Note: These are different from Phase-1 (Run-2) values!
# Phase-1 had: TIB=3, TID=4, TOB=5, TEC=6
# Phase-2 has: Endcap=4, Barrel=5
PHASE2_OT_ENDCAP = 4  # Phase2Tracker::Endcap (OT endcap disks)
PHASE2_OT_BARREL = 5  # Phase2Tracker::Barrel (OT barrel: TBPS + TB2S)


def delta_phi(phi1, phi2):
    """Compute delta phi in [-pi, pi]"""
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return dphi


def relative_difference(val1, val2, epsilon=1e-10):
    """
    Compute relative difference between two values.
    Uses max of absolute values in denominator to handle negative values (like eta).
    Returns |val1 - val2| / max(|val1|, |val2|, epsilon)
    """
    abs_diff = abs(val1 - val2)
    max_abs = max(abs(val1), abs(val2), epsilon)
    return abs_diff / max_abs


def check_significant_difference(trk_caext, trk_stubs, threshold, epsilon=1e-10):
    """
    Check if any of pt, eta, phi, or chi2 has relative difference > threshold.
    Returns (has_diff, dict of individual relative differences)
    """
    rel_diffs = {
        'pt': relative_difference(trk_caext['pt'], trk_stubs['pt'], epsilon),
        'eta': relative_difference(trk_caext['eta'], trk_stubs['eta'], epsilon),
        'phi': relative_difference(trk_caext['phi'], trk_stubs['phi'], epsilon),
        'chi2': relative_difference(trk_caext['chi2'], trk_stubs['chi2'], epsilon),
    }
    has_diff = any(rd > threshold for rd in rel_diffs.values())
    return has_diff, rel_diffs


def format_value_pair(name, val1, val2, threshold, format_spec='.2f'):
    """
    Format a single value pair with optional RED coloring if difference exceeds threshold.

    Args:
        name: The name of the value (e.g., 'pt', 'eta')
        val1: First value (CAExt)
        val2: Second value (Stubs)
        threshold: Relative difference threshold for highlighting
        format_spec: Format specification for the values (e.g., '.2f', '.3f', '.4f')

    Returns:
        Formatted string, optionally wrapped in RED ANSI codes
    """
    rel_diff = relative_difference(val1, val2)
    formatted = f"{name}={val1:{format_spec}}({val2:{format_spec}})"
    if rel_diff > threshold:
        return f"{COLOR_RED}{formatted}{COLOR_RESET}"
    return formatted


def get_eta_bin(eta):
    """Categorize eta into barrel/transition/endcap"""
    abs_eta = abs(eta)
    if abs_eta < 0.8:
        return "barrel"
    elif abs_eta < 1.5:
        return "transition"
    else:
        return "endcap"


def get_track_info(track):
    """Extract track information into a dictionary"""
    info = {
        'pt': track.pt(),
        'eta': track.eta(),
        'phi': track.phi(),
        'nHits': track.numberOfValidHits(),
        'nPixelHits': track.hitPattern().numberOfValidPixelHits(),
        'chi2': track.normalizedChi2(),
        'ndof': track.ndof(),
        'dxy': track.dxy(),
        'dz': track.dz(),
    }

    # Get hit DetIds
    hits = []
    pixel_detids = set()
    ot_detids = set()

    for i in range(track.recHitsSize()):
        hit = track.recHit(i)
        if hit.isValid():
            detid = hit.geographicalId().rawId()
            subdet = hit.geographicalId().subdetId()
            # Note: globalPosition() requires geometry service which may not be
            # available in FWLite. We store basic info without global position.
            hit_info = {
                'detid': detid,
                'subdet': subdet,
            }
            hits.append(hit_info)

            if subdet in [PIXEL_BARREL, PIXEL_ENDCAP]:
                pixel_detids.add(detid)
            elif subdet in [PHASE2_OT_BARREL, PHASE2_OT_ENDCAP]:
                ot_detids.add(detid)

    info['hits'] = hits
    info['pixel_detids'] = pixel_detids
    info['ot_detids'] = ot_detids
    info['nOTHits'] = len(ot_detids)

    return info


def tracks_match(t1, t2, dpt_rel=0.1, deta=0.02, dphi=0.02, min_shared_pixel=3):
    """
    Check if two tracks match.
    First try shared pixel hits, then fall back to kinematics.
    """
    # Count shared pixel hits
    shared_pixels = len(t1['pixel_detids'] & t2['pixel_detids'])

    # If enough shared pixel hits, consider it a match
    if shared_pixels >= min_shared_pixel:
        return True, shared_pixels, 'pixel'

    # Fall back to kinematic matching
    pt_rel = abs(t1['pt'] - t2['pt']) / t1['pt'] if t1['pt'] > 0 else 999
    d_eta = abs(t1['eta'] - t2['eta'])
    d_phi = abs(delta_phi(t1['phi'], t2['phi']))

    if pt_rel < dpt_rel and d_eta < deta and d_phi < dphi:
        return True, shared_pixels, 'kinematic'

    return False, shared_pixels, None


def process_event(tracks_caext, tracks_stubs, args):
    """
    Process a single event: match tracks between collections.
    Returns statistics and lists of matched/unmatched tracks.
    """
    caext_matched = [False] * len(tracks_caext)
    stubs_matched = [False] * len(tracks_stubs)
    matches = []  # (i_caext, j_stubs, shared_pixels, match_type)

    # Match tracks
    for i, t1 in enumerate(tracks_caext):
        best_match = -1
        best_shared = 0
        best_type = None

        for j, t2 in enumerate(tracks_stubs):
            if stubs_matched[j]:
                continue

            is_match, shared, match_type = tracks_match(
                t1, t2,
                dpt_rel=args.match_dpt_rel,
                deta=args.match_deta,
                dphi=args.match_dphi
            )

            if is_match:
                # Prefer pixel-based matches with more shared hits
                if match_type == 'pixel' and shared > best_shared:
                    best_match = j
                    best_shared = shared
                    best_type = match_type
                elif best_match < 0:
                    best_match = j
                    best_shared = shared
                    best_type = match_type

        if best_match >= 0:
            caext_matched[i] = True
            stubs_matched[best_match] = True
            matches.append((i, best_match, best_shared, best_type))

    # Collect unmatched tracks
    only_caext = [tracks_caext[i] for i in range(len(tracks_caext)) if not caext_matched[i]]
    only_stubs = [tracks_stubs[j] for j in range(len(tracks_stubs)) if not stubs_matched[j]]
    matched_caext = [tracks_caext[i] for i, _, _, _ in matches]

    return {
        'n_caext': len(tracks_caext),
        'n_stubs': len(tracks_stubs),
        'n_matched': len(matches),
        'n_only_caext': len(only_caext),
        'n_only_stubs': len(only_stubs),
        'only_caext': only_caext,
        'only_stubs': only_stubs,
        'matched': matched_caext,
        'matches': matches,
    }


def format_track(trk, verbose=False):
    """Format track info for printing"""
    s = f"pt={trk['pt']:.2f} eta={trk['eta']:.3f} phi={trk['phi']:.3f} " \
        f"nHits={trk['nHits']} nPixel={trk['nPixelHits']} nOT={trk['nOTHits']} chi2={trk['chi2']:.2f} ndof={trk['ndof']:.1f}"

    if verbose and trk['hits']:
        s += "\n      Hits: "
        for h in trk['hits']:
            s += f"[subdet={h['subdet']} detid={h['detid']}] "

    return s


def get_event_key(event):
    """Extract (run, lumi, event) tuple from a FWLite event"""
    event_id = event.eventAuxiliary().id()
    return (event_id.run(), event_id.luminosityBlock(), event_id.event())


def main():
    parser = argparse.ArgumentParser(description='Compare track collections from two ROOT files')
    parser.add_argument('caext_file', help='ROOT file with CAExtension tracks')
    parser.add_argument('stubs_file', help='ROOT file with Stubs tracks')
    parser.add_argument('--caext-tracks', default='hltPhase2PixelTracks',
                        help='Track collection label for CAExt')
    parser.add_argument('--stubs-tracks', default='hltPhase2PixelTracks',
                        help='Track collection label for Stubs')
    parser.add_argument('--caext-process', default='HLT',
                        help='Process name for CAExt tracks')
    parser.add_argument('--stubs-process', default='HLT',
                        help='Process name for Stubs tracks')
    parser.add_argument('--max-events', type=int, default=-1,
                        help='Maximum events to process (-1 for all)')
    parser.add_argument('--output', default='',
                        help='Output file for detailed track dump')
    parser.add_argument('--match-dpt-rel', type=float, default=0.1,
                        help='Relative pT matching tolerance')
    parser.add_argument('--match-deta', type=float, default=0.02,
                        help='Delta eta matching tolerance')
    parser.add_argument('--match-dphi', type=float, default=0.02,
                        help='Delta phi matching tolerance')
    parser.add_argument('--verbose', action='store_true',
                        help='Print details for unmatched tracks')
    parser.add_argument('--diff-threshold', '-d', type=float, default=0.05,
                        help='Relative difference threshold for highlighting matched tracks (default: 0.05 = 5%%)')
    args = parser.parse_args()

    # Open ROOT files using FWLite
    print(f"Opening CAExt file: {args.caext_file}")
    try:
        events_caext = Events(args.caext_file)
    except Exception as e:
        print(f"ERROR: Cannot open {args.caext_file}: {e}")
        return 1

    print(f"Opening Stubs file: {args.stubs_file}")
    try:
        events_stubs = Events(args.stubs_file)
    except Exception as e:
        print(f"ERROR: Cannot open {args.stubs_file}: {e}")
        return 1

    # Create handles for track collections
    tracks_handle_caext = Handle("std::vector<reco::Track>")
    tracks_handle_stubs = Handle("std::vector<reco::Track>")

    # Create labels (module_label, product_instance, process_name)
    caext_label = (args.caext_tracks, "", args.caext_process)
    stubs_label = (args.stubs_tracks, "", args.stubs_process)

    print(f"Using CAExt label: {caext_label}")
    print(f"Using Stubs label: {stubs_label}")

    # Build event index for stubs file (run, lumi, event) -> event index
    print("Building event index for Stubs file...")
    stubs_event_index = {}
    for i_entry, event in enumerate(events_stubs):
        key = get_event_key(event)
        stubs_event_index[key] = i_entry

    print(f"Found {len(stubs_event_index)} events in Stubs file")

    # Open output file
    outfile = None
    if args.output:
        outfile = open(args.output, 'w')
        outfile.write("# Track Comparison: CAExtension vs Stubs\n")
        outfile.write("#\n")
        outfile.write("# Categories:\n")
        outfile.write("#   MATCHED     - Matched track pair, values shown as caext_value(stubs_value)\n")
        outfile.write("#   ONLY_CAEXT  - Track found only in CAExt collection (missed by stubs)\n")
        outfile.write("#   ONLY_STUBS  - Track found only in Stubs collection (extra in stubs)\n")
        outfile.write("#\n")
        outfile.write("# MATCHED line format (single line per matched pair):\n")
        outfile.write("#   EventID MATCHED pt_caext(pt_stubs) eta_caext(eta_stubs) phi_caext(phi_stubs) nHits_caext(nHits_stubs) nPixel_caext(nPixel_stubs) nOT_caext(nOT_stubs) chi2_caext(chi2_stubs) ndof_caext(ndof_stubs) match_type shared_pixels\n")
        outfile.write("#\n")
        outfile.write("# ONLY_CAEXT and ONLY_STUBS line format (single values, no parentheses):\n")
        outfile.write("#   EventID Category pt eta phi nHits nPixel nOT chi2 ndof match_type shared_pixels\n")
        outfile.write("#\n")
        outfile.write("# Example MATCHED line: 12345 MATCHED 5.234(5.241) 1.2345(1.2348) 0.5678(0.5681) 7(9) 4(4) 3(5) 1.234(1.567) 11.0(15.0) pixel 4\n")
        outfile.write("#\n")
        outfile.write(f"# Individual values (pt, eta, phi, chi2) highlighted in RED (ANSI color) indicate values that\n")
        outfile.write(f"# differ by more than {args.diff_threshold*100:.1f}% (relative difference threshold). Use 'less -R' to view colors.\n")
        outfile.write(f"# Only the specific values exceeding the threshold are colored, not the entire line.\n")
        outfile.write("#\n")

    # Statistics
    total_stats = defaultdict(int)
    eta_stats = {
        'barrel': defaultdict(int),
        'transition': defaultdict(int),
        'endcap': defaultdict(int),
    }

    # Collect all unmatched tracks for summary
    all_only_caext = []
    all_only_stubs = []

    # Counter for matched tracks with significant differences
    n_matched_with_diff = 0

    # Determine number of events to process
    n_events_total = events_caext.size()
    n_events = n_events_total
    if args.max_events > 0:
        n_events = min(n_events, args.max_events)

    n_matched_events = 0

    print(f"\nProcessing {n_events} events...")
    print("-" * 70)

    # Process events from caext file
    for i_entry, event_caext in enumerate(events_caext):
        if args.max_events > 0 and i_entry >= args.max_events:
            break

        key = get_event_key(event_caext)
        run, lumi, event_id = key

        # Find matching event in stubs file
        if key not in stubs_event_index:
            print(f"WARNING: Event {run}:{lumi}:{event_id} not found in Stubs file")
            continue

        # Get the matching stubs event
        stubs_idx = stubs_event_index[key]
        events_stubs.to(stubs_idx)

        n_matched_events += 1

        # Get track collections using FWLite
        try:
            event_caext.getByLabel(caext_label, tracks_handle_caext)
            tracks_caext_raw = tracks_handle_caext.product()
        except Exception as e:
            print(f"ERROR: Cannot get tracks from CAExt file for event {event_id}: {e}")
            continue

        try:
            events_stubs.getByLabel(stubs_label, tracks_handle_stubs)
            tracks_stubs_raw = tracks_handle_stubs.product()
        except Exception as e:
            print(f"ERROR: Cannot get tracks from Stubs file for event {event_id}: {e}")
            continue

        # Extract track info
        tracks_caext = [get_track_info(trk) for trk in tracks_caext_raw]
        tracks_stubs = [get_track_info(trk) for trk in tracks_stubs_raw]

        # Process event
        result = process_event(tracks_caext, tracks_stubs, args)

        # Update statistics
        total_stats['n_events'] += 1
        total_stats['n_caext'] += result['n_caext']
        total_stats['n_stubs'] += result['n_stubs']
        total_stats['n_matched'] += result['n_matched']
        total_stats['n_only_caext'] += result['n_only_caext']
        total_stats['n_only_stubs'] += result['n_only_stubs']

        # Bin by eta
        for trk in result['only_caext']:
            eta_bin = get_eta_bin(trk['eta'])
            eta_stats[eta_bin]['only_caext'] += 1
            all_only_caext.append((event_id, trk))

        for trk in result['only_stubs']:
            eta_bin = get_eta_bin(trk['eta'])
            eta_stats[eta_bin]['only_stubs'] += 1
            all_only_stubs.append((event_id, trk))

        for trk in result['matched']:
            eta_bin = get_eta_bin(trk['eta'])
            eta_stats[eta_bin]['matched'] += 1

        # Check for significant differences in matched tracks and count them
        # Also store the diff status for later use in output
        match_diff_status = []  # List of (has_diff, rel_diffs) for each match
        for i_caext, j_stubs, shared_pixels, match_type in result['matches']:
            trk_caext = tracks_caext[i_caext]
            trk_stubs = tracks_stubs[j_stubs]
            has_diff, rel_diffs = check_significant_difference(trk_caext, trk_stubs, args.diff_threshold)
            match_diff_status.append((has_diff, rel_diffs))
            if has_diff:
                n_matched_with_diff += 1

        # Write to output file
        if outfile:
            # For matched tracks, write single line with both CAExt and Stubs values
            for idx, (i_caext, j_stubs, shared_pixels, match_type) in enumerate(result['matches']):
                trk_caext = tracks_caext[i_caext]
                trk_stubs = tracks_stubs[j_stubs]
                # Format each value pair with individual highlighting for those exceeding threshold
                pt_str = format_value_pair('pt', trk_caext['pt'], trk_stubs['pt'], args.diff_threshold, '.3f')
                eta_str = format_value_pair('eta', trk_caext['eta'], trk_stubs['eta'], args.diff_threshold, '.4f')
                phi_str = format_value_pair('phi', trk_caext['phi'], trk_stubs['phi'], args.diff_threshold, '.4f')
                chi2_str = format_value_pair('chi2', trk_caext['chi2'], trk_stubs['chi2'], args.diff_threshold, '.3f')
                # Write combined line: caext_value(stubs_value) for all parameters
                # Only pt, eta, phi, chi2 are highlighted individually; others remain plain
                outfile.write(f"{event_id} MATCHED "
                             f"{pt_str} "
                             f"{eta_str} "
                             f"{phi_str} "
                             f"nHits={trk_caext['nHits']}({trk_stubs['nHits']}) "
                             f"nPixel={trk_caext['nPixelHits']}({trk_stubs['nPixelHits']}) "
                             f"nOT={trk_caext['nOTHits']}({trk_stubs['nOTHits']}) "
                             f"{chi2_str} "
                             f"ndof={trk_caext['ndof']:.1f}({trk_stubs['ndof']:.1f}) "
                             f"{match_type} {shared_pixels}\n")
            for trk in result['only_caext']:
                outfile.write(f"{event_id} ONLY_CAEXT {trk['pt']:.3f} {trk['eta']:.4f} {trk['phi']:.4f} "
                             f"{trk['nHits']} {trk['nPixelHits']} {trk['nOTHits']} {trk['chi2']:.3f} "
                             f"{trk['ndof']:.1f} none 0\n")
            for trk in result['only_stubs']:
                outfile.write(f"{event_id} ONLY_STUBS {trk['pt']:.3f} {trk['eta']:.4f} {trk['phi']:.4f} "
                             f"{trk['nHits']} {trk['nPixelHits']} {trk['nOTHits']} {trk['chi2']:.3f} "
                             f"{trk['ndof']:.1f} none 0\n")

        # Print per-event summary (every 10 events or if there are issues)
        if i_entry % 10 == 0 or result['n_only_caext'] > 0:
            print(f"Event {event_id}: CAExt={result['n_caext']} Stubs={result['n_stubs']} "
                  f"Matched={result['n_matched']} OnlyCAExt={result['n_only_caext']} OnlyStubs={result['n_only_stubs']}")

            if args.verbose:
                # Print matched tracks with both CAExt and Stubs hit info
                for idx, (i_caext, j_stubs, shared_pixels, match_type) in enumerate(result['matches']):
                    trk_caext = tracks_caext[i_caext]
                    trk_stubs = tracks_stubs[j_stubs]
                    # Format each value pair with individual highlighting for those exceeding threshold
                    pt_str = format_value_pair('pt', trk_caext['pt'], trk_stubs['pt'], args.diff_threshold, '.2f')
                    eta_str = format_value_pair('eta', trk_caext['eta'], trk_stubs['eta'], args.diff_threshold, '.3f')
                    phi_str = format_value_pair('phi', trk_caext['phi'], trk_stubs['phi'], args.diff_threshold, '.3f')
                    chi2_str = format_value_pair('chi2', trk_caext['chi2'], trk_stubs['chi2'], args.diff_threshold, '.2f')
                    print(f"    MATCHED ({match_type}, {shared_pixels} shared): "
                          f"{pt_str} {eta_str} {phi_str} "
                          f"nHits={trk_caext['nHits']}({trk_stubs['nHits']}) "
                          f"nPixel={trk_caext['nPixelHits']}({trk_stubs['nPixelHits']}) "
                          f"nOT={trk_caext['nOTHits']}({trk_stubs['nOTHits']}) "
                          f"{chi2_str} "
                          f"ndof={trk_caext['ndof']:.1f}({trk_stubs['ndof']:.1f})")

                # Print missed tracks
                for trk in result['only_caext']:
                    print(f"    MISSED: {format_track(trk, verbose=True)}")

    # Close output file
    if outfile:
        outfile.close()

    # Print summary
    print("\n")
    print("=" * 70)
    print("Track Comparison Summary: CAExtension (P-hits) vs Stubs")
    print("=" * 70)
    print(f"Events analyzed: {n_matched_events}")
    print()

    if n_matched_events == 0:
        print("No events were matched between the two files.")
        print("=" * 70)
        return 0

    print("Total tracks:")
    print(f"  CAExtension: {total_stats['n_caext']} ({total_stats['n_caext']/n_matched_events:.1f} per event)")
    print(f"  Stubs:       {total_stats['n_stubs']} ({total_stats['n_stubs']/n_matched_events:.1f} per event)")
    print()
    print("Matching results:")
    print(f"  Matched:     {total_stats['n_matched']}")
    print(f"  OnlyCAExt:   {total_stats['n_only_caext']} (MISSED by stubs)")
    print(f"  OnlyStubs:   {total_stats['n_only_stubs']} (EXTRA in stubs)")
    print()
    if total_stats['n_matched'] > 0:
        diff_pct = 100.0 * n_matched_with_diff / total_stats['n_matched']
        print(f"Matched tracks with significant differences (threshold={args.diff_threshold*100:.1f}%):")
        print(f"  {n_matched_with_diff} of {total_stats['n_matched']} matched tracks ({diff_pct:.1f}%)")
        print()

    # Efficiency by eta region
    print("Efficiency by eta region (stubs vs CAExt as reference):")
    print("-" * 70)
    print(f"{'Region':<12} {'Matched':>10} {'Missed':>10} {'Total':>10} {'Efficiency':>12} {'Miss Rate':>12}")
    print("-" * 70)
    for region in ['barrel', 'transition', 'endcap']:
        matched = eta_stats[region]['matched']
        missed = eta_stats[region]['only_caext']
        total = matched + missed
        if total > 0:
            eff = 100.0 * matched / total
            miss_rate = 100.0 * missed / total
            print(f"{region:<12} {matched:>10} {missed:>10} {total:>10} {eff:>11.1f}% {miss_rate:>11.1f}%")
        else:
            print(f"{region:<12} {0:>10} {0:>10} {0:>10} {'N/A':>12} {'N/A':>12}")
    print("-" * 70)

    # Print sample of missed tracks
    if all_only_caext:
        print()
        print("Sample of tracks MISSED by stubs (first 20):")
        print("-" * 70)
        for i, (evt, trk) in enumerate(all_only_caext[:20]):
            print(f"  Event {evt}: {format_track(trk)}")

    # Print sample of extra tracks
    if all_only_stubs and args.verbose:
        print()
        print("Sample of EXTRA tracks in stubs (first 10):")
        print("-" * 70)
        for i, (evt, trk) in enumerate(all_only_stubs[:10]):
            print(f"  Event {evt}: {format_track(trk)}")

    print()
    print("Note: CAExt uses P-hits from TOB PS barrel layers 1-2 only")
    print("      Stubs uses inner+outer hits from all OT stubs")
    print("=" * 70)

    if args.output:
        print(f"\nDetailed output written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
