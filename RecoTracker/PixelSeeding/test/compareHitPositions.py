#!/usr/bin/env python3
"""
Check consistency between OT_HIT and PS_STUB positions within a single log file.

This script parses debug output and verifies that PS_STUB positions match their
corresponding OT_HIT positions, using lowerHitIdx to match stubs to hits.

Usage:
    python compareHitPositions.py stubs_log.txt [--tolerance 1e-4] [--verbose]
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class HitPosition:
    """Represents a hit position with its coordinates and errors."""
    det_id: int
    idx: int
    x: float
    y: float
    z: float
    r: float
    iphi: int
    xerr_local: Optional[float] = None  # Local x error (variance)
    yerr_local: Optional[float] = None  # Local y error (variance)
    xglobal_err: Optional[float] = None  # Global x error (variance)
    yglobal_err: Optional[float] = None  # Global y error (variance)
    zglobal_err: Optional[float] = None  # Global z error (variance)
    extra_info: Dict[str, str] = field(default_factory=dict)  # Store additional fields

    def __str__(self):
        err_str = ""
        if self.xerr_local is not None:
            err_str += f" xerrLocal={self.xerr_local:.4e}"
        if self.yerr_local is not None:
            err_str += f" yerrLocal={self.yerr_local:.4e}"
        return f"detId={self.det_id} idx={self.idx} x={self.x:.6f} y={self.y:.6f} z={self.z:.6f} r={self.r:.4f} iphi={self.iphi}{err_str}"


def parse_ot_hit(line: str) -> Optional[HitPosition]:
    """
    Parse an OT_HIT line from CA+Stubs debug output.

    Format (with error fields):
    OT_HIT sensorDetId=437292900 hitOffset=0 x=-12.345678 y=23.456789 z=45.678901 r=26.5123 iphi=12345 xerrLocal=1.2345e-04 yerrLocal=2.3456e-04 xGlobalErr=1.0000e-04 yGlobalErr=2.0000e-04 zGlobalErr=3.0000e-04 stackDetId=437292899 detIdx=1850
    """
    # Try parsing with error fields first
    pattern_with_errors = r'OT_HIT\s+sensorDetId=(\d+)\s+hitOffset=(\d+)\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi=([-\d]+)\s+xerrLocal=([-\d.eE+]+)\s+yerrLocal=([-\d.eE+]+)\s+xGlobalErr=([-\d.eE+]+)\s+yGlobalErr=([-\d.eE+]+)\s+zGlobalErr=([-\d.eE+]+)'
    match = re.search(pattern_with_errors, line)
    if match:
        extra = {}
        # Extract stackDetId and detIdx if present
        stack_match = re.search(r'stackDetId=(\d+)', line)
        if stack_match:
            extra['stackDetId'] = stack_match.group(1)
        det_match = re.search(r'detIdx=(\d+)', line)
        if det_match:
            extra['detIdx'] = det_match.group(1)

        return HitPosition(
            det_id=int(match.group(1)),
            idx=int(match.group(2)),
            x=float(match.group(3)),
            y=float(match.group(4)),
            z=float(match.group(5)),
            r=float(match.group(6)),
            iphi=int(match.group(7)),
            xerr_local=float(match.group(8)),
            yerr_local=float(match.group(9)),
            xglobal_err=float(match.group(10)),
            yglobal_err=float(match.group(11)),
            zglobal_err=float(match.group(12)),
            extra_info=extra
        )

    # Fallback to pattern without error fields (legacy format)
    pattern = r'OT_HIT\s+sensorDetId=(\d+)\s+hitOffset=(\d+)\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi=([-\d]+)'
    match = re.search(pattern, line)
    if match:
        extra = {}
        # Extract stackDetId and detIdx if present
        stack_match = re.search(r'stackDetId=(\d+)', line)
        if stack_match:
            extra['stackDetId'] = stack_match.group(1)
        det_match = re.search(r'detIdx=(\d+)', line)
        if det_match:
            extra['detIdx'] = det_match.group(1)

        return HitPosition(
            det_id=int(match.group(1)),
            idx=int(match.group(2)),
            x=float(match.group(3)),
            y=float(match.group(4)),
            z=float(match.group(5)),
            r=float(match.group(6)),
            iphi=int(match.group(7)),
            extra_info=extra
        )
    return None


def parse_stub(line: str) -> Optional[HitPosition]:
    """
    Parse a PS_STUB or PS_STUB_VH line from stub debug output.

    Format (with error fields):
    PS_STUB module=42 stubIdx=0 lowerHitIdx=5: x=-12.345678 y=23.456789 z=45.678901 r=26.5123 iphi_lower=12345 xerrLocal=1.2345e-04 yerrLocal=2.3456e-04 sensorDetId=437292900
    """
    # Try parsing with error fields first
    pattern_with_errors = r'PS_STUB(?:_VH)?\s+module=(\d+)\s+stubIdx=(\d+)\s+lowerHitIdx=(\d+):\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi_lower=([-\d]+)\s+xerrLocal=([-\d.eE+]+)\s+yerrLocal=([-\d.eE+]+)\s+sensorDetId=(\d+)'
    match = re.search(pattern_with_errors, line)
    if match:
        extra = {
            'module': match.group(1),
            'lowerHitIdx': match.group(3)
        }

        return HitPosition(
            det_id=int(match.group(11)),  # sensorDetId
            idx=int(match.group(2)),      # stubIdx
            x=float(match.group(4)),
            y=float(match.group(5)),
            z=float(match.group(6)),
            r=float(match.group(7)),
            iphi=int(match.group(8)),
            xerr_local=float(match.group(9)),
            yerr_local=float(match.group(10)),
            extra_info=extra
        )

    # Fallback to pattern without error fields (legacy format)
    pattern = r'PS_STUB(?:_VH)?\s+module=(\d+)\s+stubIdx=(\d+)\s+lowerHitIdx=(\d+):\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi_lower=([-\d]+)\s+sensorDetId=(\d+)'
    match = re.search(pattern, line)
    if match:
        extra = {
            'module': match.group(1),
            'lowerHitIdx': match.group(3)
        }

        return HitPosition(
            det_id=int(match.group(9)),  # sensorDetId
            idx=int(match.group(2)),      # stubIdx
            x=float(match.group(4)),
            y=float(match.group(5)),
            z=float(match.group(6)),
            r=float(match.group(7)),
            iphi=int(match.group(8)),
            extra_info=extra
        )
    return None


def parse_p_hit(line: str) -> Optional[HitPosition]:
    """
    Parse a P_HIT line from CA Extension debug output.

    Format (with error fields):
    P_HIT detId=437292900 idx=5 x=-12.345678 y=23.456789 z=45.678901 r=26.5123 iphi=12345 xerrLocal=1.2345e-04 yerrLocal=2.3456e-04 moduleOffset=123
    """
    # Try parsing with error fields first
    pattern_with_errors = r'P_HIT\s+detId=(\d+)\s+idx=(\d+)\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi=([-\d]+)\s+xerrLocal=([-\d.eE+]+)\s+yerrLocal=([-\d.eE+]+)\s+moduleOffset=(\d+)'
    match = re.search(pattern_with_errors, line)
    if match:
        extra = {
            'moduleOffset': match.group(10)
        }

        return HitPosition(
            det_id=int(match.group(1)),
            idx=int(match.group(2)),
            x=float(match.group(3)),
            y=float(match.group(4)),
            z=float(match.group(5)),
            r=float(match.group(6)),
            iphi=int(match.group(7)),
            xerr_local=float(match.group(8)),
            yerr_local=float(match.group(9)),
            extra_info=extra
        )

    # Fallback to pattern without error fields (legacy format)
    pattern = r'P_HIT\s+detId=(\d+)\s+idx=(\d+)\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)\s+r=([-\d.eE+]+)\s+iphi=([-\d]+)\s+moduleOffset=(\d+)'
    match = re.search(pattern, line)
    if match:
        extra = {
            'moduleOffset': match.group(8)
        }

        return HitPosition(
            det_id=int(match.group(1)),
            idx=int(match.group(2)),
            x=float(match.group(3)),
            y=float(match.group(4)),
            z=float(match.group(5)),
            r=float(match.group(6)),
            iphi=int(match.group(7)),
            extra_info=extra
        )
    return None


@dataclass
class EventData:
    """Represents all hit data from a single event."""
    run: int
    event: int
    lumi: int
    ot_hits: List[HitPosition]
    stubs: List[HitPosition]
    p_hits: List[HitPosition] = field(default_factory=list)  # P_HIT entries (CA Extension)

    def __str__(self):
        return f"Run {self.run}, Event {self.event}, LumiSection {self.lumi}"


def parse_event_boundary(line: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse an event boundary line.

    Format:
    Begin processing the 1st record. Run 1, Event 2, LumiSection 1 on stream 0 at ...

    Returns:
        Tuple of (run, event, lumi) or None if not an event boundary
    """
    pattern = r'Begin processing the \d+\w* record\. Run (\d+), Event (\d+), LumiSection (\d+)'
    match = re.search(pattern, line)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def parse_log_file_by_event(filepath: str) -> List[EventData]:
    """
    Parse a log file and extract hit data grouped by event.

    This is essential for consistency checks where hitOffset values
    restart at 0 for each event, so matching must be done within individual events.

    Returns:
        List of EventData objects, one per event
    """
    events = []
    current_event = None

    # Temporary storage for current event's hits
    current_ot_hits = []
    current_stubs = []
    current_p_hits = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # Check for event boundary
                boundary = parse_event_boundary(line)
                if boundary:
                    # Save previous event if it exists
                    if current_event is not None:
                        events.append(EventData(
                            run=current_event[0],
                            event=current_event[1],
                            lumi=current_event[2],
                            ot_hits=current_ot_hits,
                            stubs=current_stubs,
                            p_hits=current_p_hits
                        ))
                    # Start new event
                    current_event = boundary
                    current_ot_hits = []
                    current_stubs = []
                    current_p_hits = []
                    continue

                # Only process hit lines if we're inside an event
                if current_event is None:
                    continue

                if 'OT_HIT' in line:
                    hit = parse_ot_hit(line)
                    if hit:
                        current_ot_hits.append(hit)
                elif 'PS_STUB' in line:
                    stub = parse_stub(line)
                    if stub:
                        current_stubs.append(stub)
                elif 'P_HIT' in line:
                    hit = parse_p_hit(line)
                    if hit:
                        current_p_hits.append(hit)

        # Don't forget the last event
        if current_event is not None:
            events.append(EventData(
                run=current_event[0],
                event=current_event[1],
                lumi=current_event[2],
                ot_hits=current_ot_hits,
                stubs=current_stubs,
                p_hits=current_p_hits
            ))

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    return events


def positions_match(hit1: HitPosition, hit2: HitPosition, tolerance: float, error_tolerance: float = 1e-6) -> Tuple[bool, Dict[str, float]]:
    """
    Compare two hit positions within tolerance.

    Returns:
        Tuple of (match_bool, differences_dict)
    """
    diff_x = abs(hit1.x - hit2.x)
    diff_y = abs(hit1.y - hit2.y)
    diff_z = abs(hit1.z - hit2.z)
    diff_r = abs(hit1.r - hit2.r)

    diffs = {
        'x': diff_x,
        'y': diff_y,
        'z': diff_z,
        'r': diff_r
    }

    # Add error comparisons if both hits have error fields
    if hit1.xerr_local is not None and hit2.xerr_local is not None:
        diffs['xerr_local'] = abs(hit1.xerr_local - hit2.xerr_local)
    if hit1.yerr_local is not None and hit2.yerr_local is not None:
        diffs['yerr_local'] = abs(hit1.yerr_local - hit2.yerr_local)

    # Position matching
    pos_matches = all(diffs[k] <= tolerance for k in ['x', 'y', 'z', 'r'])

    # Error matching (use relative tolerance for errors)
    err_matches = True
    if 'xerr_local' in diffs:
        ref_xerr = max(abs(hit1.xerr_local), abs(hit2.xerr_local), 1e-10)
        if diffs['xerr_local'] / ref_xerr > error_tolerance:
            err_matches = False
    if 'yerr_local' in diffs:
        ref_yerr = max(abs(hit1.yerr_local), abs(hit2.yerr_local), 1e-10)
        if diffs['yerr_local'] / ref_yerr > error_tolerance:
            err_matches = False

    return pos_matches and err_matches, diffs


def build_hit_index_map(ot_hits: List[HitPosition]) -> Dict[int, HitPosition]:
    """
    Build a map from hitOffset to OT_HIT for fast lookup.

    This is used to find the OT_HIT that corresponds to a stub's lowerHitIdx.
    """
    hit_map = {}
    for hit in ot_hits:
        # hit.idx is the hitOffset for OT_HITs
        hit_map[hit.idx] = hit
    return hit_map


def build_detid_map(hits: List[HitPosition]) -> Dict[int, List[HitPosition]]:
    """
    Build a map from detector ID to list of hits for matching by detId.

    This is used for comparing P_HITs (CA Extension) with OT_HITs/PS_STUBs (Stubs).
    """
    detid_map = {}
    for hit in hits:
        if hit.det_id not in detid_map:
            detid_map[hit.det_id] = []
        detid_map[hit.det_id].append(hit)
    return detid_map


def single_file_consistency_check(
    ot_hits: List[HitPosition],
    stubs: List[HitPosition],
    tolerance: float,
    error_tolerance: float = 1e-6
) -> Dict:
    """
    Check consistency between OT_HIT and PS_STUB entries within a single event.

    The PS_STUB's lowerHitIdx should match an OT_HIT's hitOffset, and their
    positions should be identical (or very close) since the stub uses the
    lower hit's position.

    Returns:
        Dictionary with comparison statistics
    """
    # Build lookup from hitOffset to OT_HIT
    hit_map = build_hit_index_map(ot_hits)

    stats = {
        'total_ot_hits': len(ot_hits),
        'total_stubs': len(stubs),
        'matched_stubs': 0,
        'mismatched_stubs': [],
        'error_mismatched_stubs': [],  # Stubs with matching positions but mismatched errors
        'stubs_with_missing_hit': [],
        'unused_ot_hits': set(hit_map.keys())  # Track which hits are used
    }

    for stub in stubs:
        lower_hit_idx = int(stub.extra_info.get('lowerHitIdx', -1))

        if lower_hit_idx < 0:
            # Should not happen if parsing is correct
            stats['stubs_with_missing_hit'].append({
                'stub': stub,
                'reason': 'No lowerHitIdx in stub data'
            })
            continue

        if lower_hit_idx not in hit_map:
            stats['stubs_with_missing_hit'].append({
                'stub': stub,
                'lowerHitIdx': lower_hit_idx,
                'reason': f'No OT_HIT with hitOffset={lower_hit_idx}'
            })
            continue

        ot_hit = hit_map[lower_hit_idx]

        # Mark this hit as used
        stats['unused_ot_hits'].discard(lower_hit_idx)

        # Compare positions
        matches, diffs = positions_match(stub, ot_hit, tolerance, error_tolerance)

        # Check if position matches but errors don't
        pos_only_matches = all(diffs[k] <= tolerance for k in ['x', 'y', 'z', 'r'])
        error_mismatch = pos_only_matches and not matches

        if matches:
            stats['matched_stubs'] += 1
        elif error_mismatch:
            stats['error_mismatched_stubs'].append({
                'stub': stub,
                'ot_hit': ot_hit,
                'lowerHitIdx': lower_hit_idx,
                'diffs': diffs
            })
        else:
            stats['mismatched_stubs'].append({
                'stub': stub,
                'ot_hit': ot_hit,
                'lowerHitIdx': lower_hit_idx,
                'diffs': diffs
            })

    # Convert set to list for JSON serialization
    stats['unused_ot_hits'] = sorted(list(stats['unused_ot_hits']))

    return stats


def single_file_consistency_check_by_event(
    events: List[EventData],
    tolerance: float,
    error_tolerance: float = 1e-6
) -> Dict:
    """
    Check consistency between OT_HIT and PS_STUB entries, processing each event separately.

    This is critical because hitOffset values restart at 0 for each event,
    so stubs must be matched to hits within the same event.

    Returns:
        Dictionary with per-event and aggregate statistics
    """
    per_event_stats = []
    aggregate_stats = {
        'total_events': len(events),
        'events_with_all_matches': 0,
        'events_with_mismatches': 0,
        'events_with_error_mismatches': 0,
        'events_with_missing_hits': 0,
        'total_ot_hits': 0,
        'total_stubs': 0,
        'total_p_hits': 0,
        'total_matched_stubs': 0,
        'total_mismatched_stubs': 0,
        'total_error_mismatched_stubs': 0,
        'total_stubs_with_missing_hit': 0,
        'events_with_issues': []  # List of (event_info, stats) for events with problems
    }

    for event_data in events:
        event_stats = single_file_consistency_check(
            event_data.ot_hits,
            event_data.stubs,
            tolerance,
            error_tolerance
        )
        event_stats['event_info'] = str(event_data)
        event_stats['run'] = event_data.run
        event_stats['event'] = event_data.event
        event_stats['lumi'] = event_data.lumi
        event_stats['total_p_hits'] = len(event_data.p_hits)

        per_event_stats.append(event_stats)

        # Update aggregate stats
        aggregate_stats['total_ot_hits'] += event_stats['total_ot_hits']
        aggregate_stats['total_stubs'] += event_stats['total_stubs']
        aggregate_stats['total_p_hits'] += event_stats['total_p_hits']
        aggregate_stats['total_matched_stubs'] += event_stats['matched_stubs']
        aggregate_stats['total_mismatched_stubs'] += len(event_stats['mismatched_stubs'])
        aggregate_stats['total_error_mismatched_stubs'] += len(event_stats['error_mismatched_stubs'])
        aggregate_stats['total_stubs_with_missing_hit'] += len(event_stats['stubs_with_missing_hit'])

        has_mismatches = len(event_stats['mismatched_stubs']) > 0
        has_error_mismatches = len(event_stats['error_mismatched_stubs']) > 0
        has_missing = len(event_stats['stubs_with_missing_hit']) > 0

        if has_mismatches:
            aggregate_stats['events_with_mismatches'] += 1
        if has_error_mismatches:
            aggregate_stats['events_with_error_mismatches'] += 1
        if has_missing:
            aggregate_stats['events_with_missing_hits'] += 1
        if not has_mismatches and not has_error_mismatches and not has_missing:
            aggregate_stats['events_with_all_matches'] += 1
        else:
            aggregate_stats['events_with_issues'].append(event_stats)

    return {
        'per_event': per_event_stats,
        'aggregate': aggregate_stats
    }


def print_report(stats: Dict, tolerance: float, error_tolerance: float, filepath: str):
    """Print a formatted consistency check report with per-event breakdown."""
    agg = stats['aggregate']
    per_event = stats['per_event']

    print("=" * 80)
    print("OT_HIT vs PS_STUB CONSISTENCY CHECK REPORT (PER-EVENT)")
    print("=" * 80)
    print()
    print(f"File: {filepath}")
    print(f"Position tolerance: {tolerance}")
    print(f"Error tolerance (relative): {error_tolerance}")
    print()

    print("-" * 40)
    print("AGGREGATE SUMMARY")
    print("-" * 40)
    print(f"Total events processed:     {agg['total_events']}")
    print(f"Total OT_HIT entries:       {agg['total_ot_hits']}")
    print(f"Total PS_STUB entries:      {agg['total_stubs']}")
    if agg['total_p_hits'] > 0:
        print(f"Total P_HIT entries:        {agg['total_p_hits']}")
    print()

    print("-" * 40)
    print("AGGREGATE CONSISTENCY RESULTS")
    print("-" * 40)
    print(f"Stubs with matching positions & errors: {agg['total_matched_stubs']}")
    print(f"Stubs with position mismatches:         {agg['total_mismatched_stubs']}")
    print(f"Stubs with error mismatches only:       {agg['total_error_mismatched_stubs']}")
    print(f"Stubs with missing OT_HIT:              {agg['total_stubs_with_missing_hit']}")
    print()
    print(f"Events with all stubs matched:          {agg['events_with_all_matches']}")
    print(f"Events with position mismatches:        {agg['events_with_mismatches']}")
    print(f"Events with error mismatches:           {agg['events_with_error_mismatches']}")
    print(f"Events with missing OT_HITs:            {agg['events_with_missing_hits']}")
    print()

    # Print per-event summary
    print("-" * 40)
    print("PER-EVENT SUMMARY")
    print("-" * 40)
    for event_stats in per_event:
        event_info = event_stats['event_info']
        n_ot = event_stats['total_ot_hits']
        n_stubs = event_stats['total_stubs']
        n_matched = event_stats['matched_stubs']
        n_mismatch = len(event_stats['mismatched_stubs'])
        n_err_mismatch = len(event_stats['error_mismatched_stubs'])
        n_missing = len(event_stats['stubs_with_missing_hit'])

        status = "OK" if n_mismatch == 0 and n_err_mismatch == 0 and n_missing == 0 else "ISSUES"
        print(f"  {event_info}: {n_ot} OT_HITs, {n_stubs} stubs, "
              f"{n_matched} matched, {n_mismatch} pos mismatches, {n_err_mismatch} err mismatches, {n_missing} missing [{status}]")
    print()

    # Print details for events with issues
    if agg['events_with_issues']:
        print("-" * 40)
        print(f"EVENTS WITH ISSUES ({len(agg['events_with_issues'])})")
        print("-" * 40)

        for event_stats in agg['events_with_issues']:
            event_info = event_stats['event_info']
            print(f"\n  === {event_info} ===")

            # Stubs with missing OT_HIT
            if event_stats['stubs_with_missing_hit']:
                print(f"\n  Stubs with missing OT_HIT ({len(event_stats['stubs_with_missing_hit'])}):")
                for i, item in enumerate(event_stats['stubs_with_missing_hit'][:10]):
                    stub = item['stub']
                    reason = item['reason']
                    lower_idx = item.get('lowerHitIdx', 'N/A')
                    print(f"    #{i+1}: {stub}")
                    print(f"         lowerHitIdx={lower_idx}, Reason: {reason}")
                if len(event_stats['stubs_with_missing_hit']) > 10:
                    print(f"    ... and {len(event_stats['stubs_with_missing_hit']) - 10} more")

            # Position mismatches
            if event_stats['mismatched_stubs']:
                print(f"\n  Position mismatches ({len(event_stats['mismatched_stubs'])}):")
                for i, mismatch in enumerate(event_stats['mismatched_stubs'][:10]):
                    stub = mismatch['stub']
                    ot_hit = mismatch['ot_hit']
                    diffs = mismatch['diffs']
                    lower_idx = mismatch['lowerHitIdx']
                    print(f"    Mismatch #{i+1} (lowerHitIdx={lower_idx}):")
                    print(f"      PS_STUB: {stub}")
                    print(f"      OT_HIT:  {ot_hit}")
                    diff_str = f"dx={diffs['x']:.6e}, dy={diffs['y']:.6e}, dz={diffs['z']:.6e}, dr={diffs['r']:.6e}"
                    if 'xerr_local' in diffs:
                        diff_str += f", dxerr={diffs['xerr_local']:.6e}"
                    if 'yerr_local' in diffs:
                        diff_str += f", dyerr={diffs['yerr_local']:.6e}"
                    print(f"      Differences: {diff_str}")
                if len(event_stats['mismatched_stubs']) > 10:
                    print(f"    ... and {len(event_stats['mismatched_stubs']) - 10} more mismatches")

            # Error-only mismatches
            if event_stats['error_mismatched_stubs']:
                print(f"\n  Error mismatches (positions OK) ({len(event_stats['error_mismatched_stubs'])}):")
                for i, mismatch in enumerate(event_stats['error_mismatched_stubs'][:10]):
                    stub = mismatch['stub']
                    ot_hit = mismatch['ot_hit']
                    diffs = mismatch['diffs']
                    lower_idx = mismatch['lowerHitIdx']
                    print(f"    Error mismatch #{i+1} (lowerHitIdx={lower_idx}):")
                    print(f"      PS_STUB: {stub}")
                    print(f"      OT_HIT:  {ot_hit}")
                    err_diff_str = ""
                    if 'xerr_local' in diffs:
                        err_diff_str += f"dxerr={diffs['xerr_local']:.6e}"
                    if 'yerr_local' in diffs:
                        if err_diff_str:
                            err_diff_str += ", "
                        err_diff_str += f"dyerr={diffs['yerr_local']:.6e}"
                    print(f"      Error differences: {err_diff_str}")
                if len(event_stats['error_mismatched_stubs']) > 10:
                    print(f"    ... and {len(event_stats['error_mismatched_stubs']) - 10} more error mismatches")

        print()

    # Final verdict
    print("=" * 80)
    total_issues = agg['total_mismatched_stubs'] + agg['total_error_mismatched_stubs'] + agg['total_stubs_with_missing_hit']
    if total_issues == 0:
        if agg['total_stubs'] == 0:
            print("RESULT: No stubs found to check.")
        else:
            print(f"RESULT: All {agg['total_matched_stubs']} stubs across {agg['total_events']} events "
                  f"are consistent with their corresponding OT_HITs!")
    else:
        issues = []
        if agg['total_mismatched_stubs'] > 0:
            issues.append(f"{agg['total_mismatched_stubs']} position mismatches in "
                         f"{agg['events_with_mismatches']} events")
        if agg['total_error_mismatched_stubs'] > 0:
            issues.append(f"{agg['total_error_mismatched_stubs']} error mismatches in "
                         f"{agg['events_with_error_mismatches']} events")
        if agg['total_stubs_with_missing_hit'] > 0:
            issues.append(f"{agg['total_stubs_with_missing_hit']} stubs with missing OT_HIT in "
                         f"{agg['events_with_missing_hits']} events")
        print(f"RESULT: Issues found: {', '.join(issues)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Check consistency between OT_HIT and PS_STUB positions within a log file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s stubs_log.txt
    %(prog)s stubs_log.txt --tolerance 1e-6
    %(prog)s stubs_log.txt --error-tolerance 1e-4
    %(prog)s stubs_log.txt --verbose

This script verifies that PS_STUB positions and errors match their corresponding
OT_HIT positions and errors within the same log file. Matching is done per-event
using the lowerHitIdx field from stubs to find the corresponding OT_HIT by hitOffset.

IMPORTANT: Matching is done per-event because hitOffset values restart at 0
for each event. Event boundaries are detected using lines like:
'Begin processing the Nth record. Run X, Event Y, LumiSection Z ...'
        """
    )

    parser.add_argument('logfile', help='Log file to check for OT_HIT vs PS_STUB consistency')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-4,
                        help='Position tolerance for matching (default: 1e-4)')
    parser.add_argument('--error-tolerance', '-e', type=float, default=1e-6,
                        help='Relative tolerance for error matching (default: 1e-6)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    filepath = args.logfile

    print(f"Parsing {filepath} by event...")
    events = parse_log_file_by_event(filepath)

    if not events:
        print(f"\nError: No events found in {filepath}")
        print("Make sure the log file contains event boundaries like:")
        print("  'Begin processing the Nth record. Run X, Event Y, LumiSection Z ...'")
        sys.exit(1)

    # Calculate totals for verbose output
    total_ot_hits = sum(len(e.ot_hits) for e in events)
    total_stubs = sum(len(e.stubs) for e in events)
    total_p_hits = sum(len(e.p_hits) for e in events)

    if args.verbose:
        print(f"\nFile contents ({len(events)} events):")
        print(f"  OT_HITs: {total_ot_hits}")
        print(f"  PS_STUBs: {total_stubs}")
        if total_p_hits > 0:
            print(f"  P_HITs: {total_p_hits}")

    if total_ot_hits == 0:
        print(f"\nError: No OT_HIT entries found in {filepath}")
        print("This script requires OT_HIT entries to match against PS_STUBs.")
        sys.exit(1)

    if total_stubs == 0:
        print(f"\nWarning: No PS_STUB entries found in {filepath}")
        print("Nothing to check - no stubs to verify against hits.")
        sys.exit(0)

    print(f"\nChecking consistency: PS_STUB positions/errors vs OT_HIT positions/errors...")
    print(f"(Matching stubs to hits within each event using lowerHitIdx -> hitOffset)")
    print()

    # Perform consistency check by event
    stats = single_file_consistency_check_by_event(events, args.tolerance, args.error_tolerance)

    # Print report
    print_report(stats, args.tolerance, args.error_tolerance, filepath)

    # Return exit code based on results
    agg = stats['aggregate']
    if agg['total_mismatched_stubs'] > 0 or agg['total_error_mismatched_stubs'] > 0 or agg['total_stubs_with_missing_hit'] > 0:
        sys.exit(1)
    return 0


if __name__ == '__main__':
    main()
