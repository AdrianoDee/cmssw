#!/usr/bin/env python3
# Aggregate gblReplay output into per-layer residual-pull statistics: per detector layer, the clipped
# mean/sigma of the biased (pb) and unbiased (pu) smoothed-residual pulls in the curvilinear U
# (transverse) and V (longitudinal) directions. Layers are classified from the hit's global (r, z) plus
# the module-type discriminator gem.
#
# Per-charge sections report the signed smoothed residual (rU, rV) per (hit-class x layer/disk-bin x
# charge): the charge-antisymmetric part (mean(q=+1) - mean(q=-1)) isolates a Lorentz-angle/CPE-model
# position bias, the charge-symmetric part a geometry/alignment-like bias. The same is reported for the
# unbiased (leave-one-out) pull puU/puV, which can show a coherent position shift the fit otherwise
# absorbs into curvature.
#
# Usage: python3 analyze_layer_pulls.py replay_pt10_mode0.txt [more replay outputs...]
#   Each file is reported separately (name it <sample>_<mode>.txt for readable headers).
import sys
import math
import re
import numpy as np

HIT_RE = re.compile(
    r"REPLAY_HIT tk (\d+) i (\d+) r (\S+) z (\S+) gem (\S+) rU (\S+) rV (\S+) sU (\S+) sV (\S+)"
    r" pbU (\S+) pbV (\S+) puU (\S+) puV (\S+) cond (\S+)")
# q is optional in the regex, so REPLAY_TRK lines without a charge field still match.
TRK_RE = re.compile(
    r"REPLAY_TRK tk (\d+) N (\d+) mode (\d+) pt (\S+) eta (\S+)(?: q (\S+))? chi2 (\S+) chi2dev (\S+) dHp (\S+)")

# (r, z) layer classification, Phase-2 D121. OT barrel radii ~23/35.7/50.8/68.6/86/110.8 cm (TBPS tilted rings
# droop a few cm inward at high |z|); TEDD double-disks at |z| ~ 131/157/189/224/266. Bins are deliberately
# generous with gaps -> unclassified hits are counted and printed so the bins can be tuned on real data.
OT_BARREL_R = [(20.0, 29.0, "OT-L28"), (31.0, 43.0, "OT-L29"), (45.0, 58.5, "OT-L30"),
               (63.0, 76.0, "OT-L31"), (80.0, 94.0, "OT-L32"), (102.0, 118.0, "OT-L33")]
OT_DISK_Z = [(120.0, 145.0, "1"), (145.0, 172.0, "2"), (172.0, 205.0, "3"),
             (205.0, 245.0, "4"), (245.0, 282.0, "5")]
PIX_BARREL_R = [(2.0, 4.9, "pixB-L0"), (5.5, 8.9, "pixB-L1"), (9.5, 13.5, "pixB-L2"), (14.0, 19.5, "pixB-L3")]


def modtype(gem):
    if gem < 3e-4:
        return "pix"
    if gem < 0.1:
        return "PS"
    return "2S"


# TFPX/TEPX |z| boundary (Phase-2 D121 geometry, Tracker_DD4hep_compatible_2021_02/pixfwd.xml, "Phase2PixelEndcap"
# polycone): rMax switches from TBPXAndTFPXOuterRadius(21.1cm) to TEPXOuterRadius(30.7cm) exactly at the
# z=120.9cm -> z=129.9cm ZSection gap -- the physical envelope boundary between the two forward-pixel disk
# systems (TFPX = intermediate disks, ZPixelForward=29.1cm out to 120.9cm; TEPX = far disks near the beampipe,
# 129.9cm to 247.8cm). PIXD_SPLIT_Z = 125.0 (gap midpoint) below.
PIXD_SPLIT_Z = 125.0


def classify(r, z, gem):
    mt = modtype(gem)
    if mt == "pix":
        if abs(z) < 26.0:
            for lo, hi, name in PIX_BARREL_R:
                if lo <= r <= hi:
                    return name
            return "pixB-?"
        side = "f" if z > 0 else "b"
        return f"TFPX-{side}" if abs(z) < PIXD_SPLIT_Z else f"TEPX-{side}"
    # OT stub (PS or 2S)
    if abs(z) < 118.0:
        for lo, hi, name in OT_BARREL_R:
            if lo <= r <= hi:
                return f"{name}({mt})"
        return f"OT-B?({mt},r={r:.0f})"
    for lo, hi, name in OT_DISK_Z:
        if lo <= abs(z) <= hi:
            side = "f" if z > 0 else "b"
            return f"OT-D{name}{side}({mt})"
    return f"OT-D?({mt},z={z:.0f})"


def clipped_stats(a, nsig=3.0, it=4):
    """Iteratively sigma-clipped mean/std -- robust Gaussian-core estimate without a fitter."""
    a = a[np.isfinite(a)]
    if len(a) < 30:
        return (float("nan"), float("nan"), len(a))
    m, s = np.mean(a), np.std(a)
    for _ in range(it):
        sel = np.abs(a - m) < nsig * s
        if sel.sum() < 10:
            break
        m, s = np.mean(a[sel]), np.std(a[sel])
    return (m, s, len(a))


def signed_stats(a, nsig=4.0, it=4):
    """Iteratively sigma-clipped mean/sem/median for a RAW (signed, un-normalized) residual sample.
    N returned is the raw finite count (consistent with the 'n' column of the pull-width table above);
    mean/sem/median are computed on the sigma-clipped core (robust against stray mis-associated hits)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 5:
        return (float("nan"), float("nan"), float("nan"), n)
    m, s = np.mean(a), np.std(a)
    sel = np.ones(n, dtype=bool)
    for _ in range(it):
        if s <= 0.:
            break
        sel = np.abs(a - m) < nsig * s
        if sel.sum() < 5:
            break
        m, s = np.mean(a[sel]), np.std(a[sel])
    core = a[sel] if sel.sum() else a
    med = np.median(core)
    nsel = len(core)
    sem = s / math.sqrt(nsel) if nsel > 0 and s > 0 else float("nan")
    return (m, sem, med, n)


LAYER_ORDER_KEY = {"pixB": 0, "TFPX": 1, "TEPX": 2, "OT-L": 3, "OT-B": 4, "OT-D": 5}


def layer_sort(name):
    pop, lay = name.split("] ", 1) if "] " in name else ("", name)
    for pre, k in LAYER_ORDER_KEY.items():
        if lay.startswith(pre):
            return (pop, k, name)
    return (pop, 9, name)


# Coarse hit-class taxonomy: collapse the fine per-layer bin above into one of {pixel-barrel, TFPX, TEPX,
# PS-barrel, 2S-barrel, PS-disk, 2S-disk, other}.
def hit_class(lay_fine):
    if lay_fine.startswith("pixB"):
        return "pixel-barrel"
    if lay_fine.startswith("TFPX"):
        return "TFPX"
    if lay_fine.startswith("TEPX"):
        return "TEPX"
    if lay_fine.startswith("OT-L") or lay_fine.startswith("OT-B?"):
        if "(PS)" in lay_fine:
            return "PS-barrel"
        if "(2S)" in lay_fine:
            return "2S-barrel"
        return "OT-barrel-other"
    if lay_fine.startswith("OT-D"):
        if "(PS)" in lay_fine:
            return "PS-disk"
        if "(2S)" in lay_fine:
            return "2S-disk"
        return "OT-disk-other"
    return "other"


def report(fname):
    # two track populations fitted by the same kernel: pixel-only (prompt quads) and OT-only (stub tracks);
    # split the tables by the first hit's radius (records are contiguous: a track starts at i==0)
    hits = {}   # (population, layer) -> dict of lists
    cq = {}     # (population, layer) -> {"hitclass": str, "q": {charge -> {"rU": [...], "rV": [...]}}}
    trk_q = {}  # tk -> charge (int), from REPLAY_TRK; TRK line always precedes its own REPLAY_HIT lines
    trks = []
    unclass = 0
    pop = "?"
    with open(fname) as f:
        for line in f:
            m = HIT_RE.match(line)
            if m:
                g = m.groups()
                tkh = int(g[0])
                r, z, gem = float(g[2]), float(g[3]), float(g[4])
                if int(g[1]) == 0:
                    pop = "PIX" if r < 19.0 else "OT"
                lay_fine = classify(r, z, gem)
                lay = f"[{pop}] " + lay_fine
                if "?" in lay:
                    unclass += 1
                d = hits.setdefault(lay, {k: [] for k in ("pbU", "pbV", "puU", "puV", "rU", "rV", "cond")})
                rU, rV = float(g[5]), float(g[6])
                d["rU"].append(rU)
                d["rV"].append(rV)
                d["pbU"].append(float(g[9]))
                d["pbV"].append(float(g[10]))
                d["puU"].append(float(g[11]))
                d["puV"].append(float(g[12]))
                d["cond"].append(float(g[13]))
                q = trk_q.get(tkh)
                if q is not None:
                    cqd = cq.setdefault(lay, {"hitclass": hit_class(lay_fine), "q": {}})
                    qd = cqd["q"].setdefault(q, {"rU": [], "rV": [], "puU": [], "puV": []})
                    qd["rU"].append(rU)
                    qd["rV"].append(rV)
                    qd["puU"].append(float(g[11]))
                    qd["puV"].append(float(g[12]))
                continue
            m = TRK_RE.match(line)
            if m:
                g = m.groups()
                tk = int(g[0])
                qraw = g[5]
                q = int(qraw) if qraw is not None else None
                trk_q[tk] = q
                trks.append([float(g[1]), float(g[2]), float(g[3]), float(g[4]),
                             float(q) if q is not None else float("nan"),
                             float(g[6]), float(g[7]), float(g[8])])

    print(f"\n===== {fname} =====")
    if trks:
        t = np.array(trks)
        # columns: N(0) mode(1) pt(2) eta(3) q(4) chi2(5) chi2dev(6) dHp(7)
        mode = int(t[0, 1])
        dhp = t[np.isfinite(t[:, 7]), 7]
        chi2 = t[np.isfinite(t[:, 5]), 5]
        nq = int(np.isfinite(t[:, 4]).sum())
        print(f"tracks {len(t)}  mode {mode}  <chi2/ndof-ish> {np.median(chi2):.3f}  "
              f"replay-vs-device max|dHp|: median {np.median(dhp):.2e}  p99 {np.percentile(dhp, 99):.2e}  "
              f"(q parsed for {nq}/{len(t)} tracks)"
              if len(dhp) else f"tracks {len(t)} mode {mode} (no device-fit cross-check)")
    hdr = (f"{'layer':>16} {'n':>6} | {'pbU mean':>9} {'sig':>6} | {'pbV mean':>9} {'sig':>6} "
           f"| {'puU mean':>9} {'sig':>6} | {'puV mean':>9} {'sig':>6} | {'med cond':>9}")
    print(hdr)
    print("-" * len(hdr))
    for lay in sorted(hits, key=layer_sort):
        d = hits[lay]
        cols = []
        for k in ("pbU", "pbV", "puU", "puV"):
            m, s, n = clipped_stats(np.array(d[k]))
            cols.append((m, s))
        n = len(d["pbU"])
        cond = np.median(np.array(d["cond"]))
        print(f"{lay:>16} {n:>6} | " + " | ".join(
            (f"{m:>+9.3f} {s:>6.3f}" if m == m else f"{'--':>9} {'--':>6}") for m, s in cols) +
            f" | {cond:>9.3g}")
    if unclass:
        print(f"[{unclass} hits in unclassified '?' bins -- tune the (r,z) bins above]")

    # Per-charge signed-residual section (CPE forward-pixel bias instrumentation).
    if cq:
        print()
        print("-- per-charge SIGNED residual (rU/rV), hit-class x layer/disk-bin x charge --")
        print("   antisym = mean(q=+1) - mean(q=-1)  [Lorentz-angle/CPE-model bias: flips sign with charge]")
        print("   sym     = mean over both charges combined  [geometry/alignment-like: charge-independent]")
        hdr2 = (f"{'hit-class':>14} {'layer':>18} {'q':>3} {'N':>7} | {'rU mean':>10} {'sem':>9} {'median':>10}"
                f" | {'rV mean':>10} {'sem':>9} {'median':>10}")
        print(hdr2)
        print("-" * len(hdr2))
        for lay in sorted(cq, key=layer_sort):
            cqd = cq[lay]
            hc = cqd["hitclass"]
            for qv in sorted(cqd["q"]):
                qd = cqd["q"][qv]
                mU, seU, medU, nU = signed_stats(np.array(qd["rU"]))
                mV, seV, medV, nV = signed_stats(np.array(qd["rV"]))
                print(f"{hc:>14} {lay:>18} {qv:>+3d} {nU:>7} | {mU:>+10.5f} {seU:>9.5f} {medU:>+10.5f}"
                      f" | {mV:>+10.5f} {seV:>9.5f} {medV:>+10.5f}")
            pos, neg = cqd["q"].get(1), cqd["q"].get(-1)
            if pos and neg:
                mUp, _, _, _ = signed_stats(np.array(pos["rU"]))
                mUn, _, _, _ = signed_stats(np.array(neg["rU"]))
                mVp, _, _, _ = signed_stats(np.array(pos["rV"]))
                mVn, _, _, _ = signed_stats(np.array(neg["rV"]))
                antiU, symU = mUp - mUn, (mUp + mUn) / 2.
                antiV, symV = mVp - mVn, (mVp + mVn) / 2.
                allU = np.concatenate([np.array(pos["rU"]), np.array(neg["rU"])])
                allV = np.concatenate([np.array(pos["rV"]), np.array(neg["rV"])])
                symU_pooled, _, _, nAll = signed_stats(allU)
                symV_pooled, _, _, _ = signed_stats(allV)
                print(f"{'':>14} {lay:>18} {'+/-':>3} {nAll:>7} | antisymU {antiU:>+10.5f}  "
                      f"symU(avg) {symU:>+10.5f}  symU(pooled) {symU_pooled:>+10.5f}"
                      f" | antisymV {antiV:>+10.5f}  symV(avg) {symV:>+10.5f}  symV(pooled) {symV_pooled:>+10.5f}")
        print("-" * len(hdr2))

        # Per-charge unbiased-pull section: the biased residual measures the hit against a fit that
        # already used it, so a coherent shift partially cancels, while the leave-one-out pull shows it
        # as a nonzero mean. Only the dimensionless pull is available: REPLAY_HIT emits the biased
        # residual's sigma (sU/sV), not the leave-one-out one, so the residual in cm cannot be rebuilt.
        print()
        print("-- per-charge UNBIASED PULL (puU/puV), hit-class x layer/disk-bin x charge --")
        print("   (dimensionless pulls, NOT residuals in cm -- see comment above on why: Su is not emitted)")
        hdr3 = (f"{'hit-class':>14} {'layer':>18} {'q':>3} {'N':>7} | {'puU mean':>10} {'sem':>9} {'median':>10}"
                f" | {'puV mean':>10} {'sem':>9} {'median':>10}")
        print(hdr3)
        print("-" * len(hdr3))
        for lay in sorted(cq, key=layer_sort):
            cqd = cq[lay]
            hc = cqd["hitclass"]
            for qv in sorted(cqd["q"]):
                qd = cqd["q"][qv]
                mU, seU, medU, nU = signed_stats(np.array(qd["puU"]))
                mV, seV, medV, nV = signed_stats(np.array(qd["puV"]))
                print(f"{hc:>14} {lay:>18} {qv:>+3d} {nU:>7} | {mU:>+10.5f} {seU:>9.5f} {medU:>+10.5f}"
                      f" | {mV:>+10.5f} {seV:>9.5f} {medV:>+10.5f}")
            pos, neg = cqd["q"].get(1), cqd["q"].get(-1)
            if pos and neg:
                mUp, _, _, _ = signed_stats(np.array(pos["puU"]))
                mUn, _, _, _ = signed_stats(np.array(neg["puU"]))
                mVp, _, _, _ = signed_stats(np.array(pos["puV"]))
                mVn, _, _, _ = signed_stats(np.array(neg["puV"]))
                antiU, symU = mUp - mUn, (mUp + mUn) / 2.
                antiV, symV = mVp - mVn, (mVp + mVn) / 2.
                allU = np.concatenate([np.array(pos["puU"]), np.array(neg["puU"])])
                allV = np.concatenate([np.array(pos["puV"]), np.array(neg["puV"])])
                symU_pooled, _, _, nAll = signed_stats(allU)
                symV_pooled, _, _, _ = signed_stats(allV)
                print(f"{'':>14} {lay:>18} {'+/-':>3} {nAll:>7} | antisymU {antiU:>+10.5f}  "
                      f"symU(avg) {symU:>+10.5f}  symU(pooled) {symU_pooled:>+10.5f}"
                      f" | antisymV {antiV:>+10.5f}  symV(avg) {symV:>+10.5f}  symV(pooled) {symV_pooled:>+10.5f}")
        print("-" * len(hdr3))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_layer_pulls.py <gblReplay output> [...]")
    for fn in sys.argv[1:]:
        report(fn)
