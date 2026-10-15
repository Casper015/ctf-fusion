"""Generate the T0037 Colab stage-1 report from the verified backup; no games run.

    python bots/p_only/report/decision_collection/colab_20261012/generate_report.py --backup <backup-directory>

The backup directory holds t0037/ (the Colab payload), VERIFIED.json and connection_checkpoint.json. Every payload file
is checked against the backup manifest before anything is read. Every number and claim in RESULTS.md is computed here
from those receipts (rows, matches, run.json, registry.json, FROZEN*.json, the schedules, environment.json,
probe_*.json, the branch checkpoint); a claim the data do not support stops the script instead of being written.
"""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

BRANCH_STATUS = ("a branch-continuation run started on Colab after PR #36 merged; its last verified checkpoint was "
                 "{roots} roots ({branches} continuations) before the runtime disconnected, its completion was never "
                 "verified, and no branch rows are in this backup")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, required=True)
    args = ap.parse_args()
    source, out = args.backup / "t0037", Path(__file__).resolve().parent
    manifest = load(source / "BACKUP_MANIFEST_20261012.json")
    for name, info in manifest["files"].items():
        data = (source / name).read_bytes()
        assert len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], name

    rows = [load(p) for p in sorted((source / "data/rows").glob("*.json"))]
    matches = list((source / "data/matches").rglob("*.json.gz"))
    report = load(source / "report/run.json")
    errors = list((source / "data/errors").glob("*.json"))
    assert len(rows) == len(matches) == report["games"] == report["fixtures_scheduled"]
    assert len({r["fixture_id"] for r in rows}) == len(rows)
    per_count = collections.Counter(r["teams"] for r in rows)
    maps = set(per_count.values())
    assert len(maps) == 1, per_count
    maps_per_count, counts = maps.pop(), sorted(per_count)

    seats = [(r, seat, f) for r in rows for seat, f in enumerate(r["faults"])]
    faults = sum(f["faults"] for _, _, f in seats)
    late = sum(f["late"] for _, _, f in seats)
    dead = sum(bool(f["dead"]) for _, _, f in seats)
    bad = [{"fixture_id": r["fixture_id"], "seat": seat, "name": f["name"], "faults": f["faults"], "late": f["late"],
            "dead": f["dead"]} for r, seat, f in seats if f["faults"] or f["late"] or f["dead"]]
    for n, d in report["per_count"].items():
        rr = [r for r in rows if r["teams"] == int(n)]
        assert sum(f["faults"] for r in rr for f in r["faults"]) == d["seat_faults"]
        assert sum(d["selected_root_strata"].values()) == d["roots"]

    for p in sorted((source / "report").glob("*.json")):
        shutil.copyfile(p, out / p.name)
    shutil.copyfile(source / "BACKUP_MANIFEST_20261012.json", out / "BACKUP_MANIFEST.json")
    shutil.copyfile(source / "probe.py", out / "probe.py")
    checkpoint = load(args.backup / "connection_checkpoint.json")
    branch = BRANCH_STATUS.format(roots=checkpoint["last_verified_roots"], branches=checkpoint["last_verified_branches"])
    assert not (source / "data/branches").exists(), "branch rows are present: the report's branch status would be wrong"
    receipt = load(args.backup / "VERIFIED.json")
    assert receipt["all_file_hashes_pass"] and receipt["matches"] == receipt["rows"] == len(rows)
    receipt.update(fault_seats=bad, stage=f"whole-match collection only; {branch}")
    (out / "BACKUP_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")

    env = load(out / "environment.json")
    frozen, full = load(out / "FROZEN.json"), load(out / "FROZEN_full_schedule.json")
    registry, probes, aa = load(out / "registry.json"), load(out / "probe_feasibility.json"), load(out / "probe_aa.json")
    schedule, schedule_all = load(source / "SCHEDULE.json"), load(source / "SCHEDULE_all.json")
    fx = {f["fixture_id"]: f for f in schedule["fixtures"]}
    fx_all = {f["fixture_id"]: f for f in schedule_all["fixtures"]}
    common = set(fx) & set(fx_all)
    identical = sum(fx[k] == fx_all[k] for k in common)
    all_counts = collections.Counter(f["teams"] for f in schedule_all["fixtures"])
    des = frozen["designated"]
    reg = registry["counts"]
    enabled = {k.split(":")[0]: v for k, v in reg.items() if k.endswith(":enabled")}
    gate = {n: all(r["faults"] == r["late"] == 0 and not r["dead"] for r in probes if r["teams"] == n)
            for n in sorted({r["teams"] for r in probes})}
    failed_gate = [n for n, ok in gate.items() if not ok]
    assert not set(failed_gate) & set(counts), "a collected count failed the Colab gate"
    share = lambda x: "all" if x == 1 else f"{100 * x:.1f}%"  # noqa: E731
    fault_names = collections.Counter(b["name"] for b in bad if b["faults"])
    fault_fixtures = sorted({b["fixture_id"] for b in bad})
    compiler = env["compiler"].split()[0] + " " + env["compiler"].split()[-1]

    lines = ["# T0037 Colab stage-1 collection report", "",
             "Generated by `generate_report.py` from the verified backup: every number below is computed from the "
             "receipts. Original Colab receipts are retained byte-for-byte.", "",
             "## Scope and status", "",
             f"**Whole-match collection is complete; no decision-branch labels exist for this corpus.** This report covers "
             f"preflight, the admitted opponent pool, the sampler, {len(rows)} complete matches and root availability. "
             f"For the branch stage, {branch}. T0037 was completed on the Windows host instead (PR #44, seed 20261701); "
             "this Colab corpus stays a separately identified historical and exploratory source, never pooled with host "
             "data. No training, export, production change or playing-strength claim.", "",
             "All timing and fault observations below belong to a Google Colab CPU runtime, not the target Windows host. "
             f"Counts {', '.join(map(str, failed_gate))} failed the declared Colab feasibility gate and were not collected. "
             f"Passing two recorded probes is not a guarantee: the collection later retained {faults} seat fault(s).", "",
             "## Frozen experiment", "",
             f"- Source: `{env['ctf_fusion_commit']}`; engine: `{frozen['engine']['commit']}`.",
             f"- Runtime: {env['cpu']}, {env['vcpus']} vCPUs, {env['memory_gb']} GB RAM, Python {env['python']}, "
             f"{compiler}, no GPU.",
             f"- Official first/turn limits: {frozen['limits_ms'][0]}/{frozen['limits_ms'][1]} ms; "
             f"{env['collect_workers']} collection worker(s), frozen before collection.",
             f"- Carrier: `{des['variant']}`, {des['policy'].split(' (')[0]}, V {des['value']}; binary SHA-256 "
             f"`{des['binary_sha256']}`.",
             f"- Pool identity: `{frozen['pool_identity'][:16]}`; full-schedule frozen identity: `{full['identity'][:16]}`; "
             f"pilot frozen identity: `{frozen['identity'][:16]}`. Full identities are in the receipts.",
             f"- Training-only seed **{schedule['seed']} is consumed**. Initially {len(fx_all)} fixtures "
             f"({max(all_counts.values())} each, {min(all_counts)}–{max(all_counts)}); redeclared before games as "
             f"{len(fx)} fixtures ({maps_per_count} each, {counts[0]}–{counts[-1]}). {identical} of the {len(common)} "
             "overlapping fixtures are identical.",
             f"- Pilot schedule identity: `{frozen['schedule']['identity'][:16]}`. The raw schedule file SHA-256 is "
             f"`{frozen['schedule']['sha256']}`.", "",
             "## Availability and preflight", "",
             f"The registry contains {sum(enabled.values())} enabled opponents ({enabled.get('reference', 0)} reference, "
             f"{enabled.get('addition', 0)} addition, {enabled.get('neural', 0)} neural), "
             f"{reg.get('neural:disabled', 0)} missing-weight entries with zero sampling, and "
             f"{reg.get('neural:not_registered', 0)} excluded roles. The decision-log carriers are excluded from opponents. "
             "Availability, resolutions, model/source inputs and binary identities are retained in `registry.json`, "
             "`availability.json` and `FROZEN.json`. Sampler simulation and realized component, neural-group and lineage "
             "masses are in `sampler.json`; realized proportions differ by count and should not be described as exact "
             "target weights.", "",
             "The gate required zero faults and zero late replies across every seat in both sampled recorded-input "
             "compositions. All seats ran concurrently; replies were not applied. The recorded states and two compositions "
             "per count provide a bounded load probe, not exhaustive admission across all compositions.", "",
             "| Teams | Probe faults (two compositions) | Worst observed max reply ms | Colab gate |", "|---|---|---|---|"]
    for n in sorted(gate):
        rr = [r for r in probes if r["teams"] == n]
        lines.append(f"| {n} | {', '.join(str(r['faults']) for r in rr)} | {max(r['worst_max_ms'] for r in rr):.3f} | "
                     f"{'pass' if gate[n] else 'fail'} |")
    cand, v12, c08, cross = aa["@candidate"], aa["v12_ponly_xxl"], aa["c08_ponly_xxl"], aa["cross"]
    lines += ["", f"A/A ({aa['probe']}): the hooked carrier reproduced {round(cand['identical_share'] * cand['turns'])}/"
              f"{cand['turns']} of its own orders and production v12 {round(v12['identical_share'] * v12['turns'])}/"
              f"{v12['turns']}; carrier versus production v12 matched {share(cross['designated_vs_v12_ponly_xxl'])}. C08 A/A "
              f"matched {share(c08['identical_share'])}; v12 versus C08 matched {share(cross['v12_vs_c08'])}. These are "
              "limited response/stability checks, not proof of equivalence under every load or group-wide diversity. "
              "Details: `probe_aa.json`.", "",
              "## Whole-match coverage", "",
              "| Teams | Maps | Usable roots | Faults | Cut turns / all carrier turns |", "|---|---|---|---|---|"]
    for n, d in sorted(report["per_count"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {n} | {d['games']} | {d['roots']} | {d['seat_faults']} | {d['stages'].get('cut', 0)} / "
                     f"{sum(d['stages'].values())} |")
    strata = collections.Counter()
    for d in report["per_count"].values():
        strata.update(d["selected_root_strata"])
    if bad:
        fault_text = (f"{'Fixture' if len(fault_fixtures) == 1 else 'Fixtures'} "
                      f"{', '.join(f'`{x}`' for x in fault_fixtures)} retained "
                      + ", ".join(f"{v} fault(s) of {'the designated carrier' if k == '@candidate' else k}"
                                  for k, v in sorted(fault_names.items())) + ".")
    else:
        fault_text = "No seat fault was retained."
    lines += ["", f"Selected-root strata: {dict(strata)}. `run.json` also retains observable-stratum opportunities, logged "
              "per-turn K, root rejection reasons and stage counts. Root selection follows the frozen collector's "
              "deterministic hash over eligible logged turns; it does not optimize final outcome or enforce stratum quotas. "
              "A usable root is not a branch label, and none were completed for this corpus.", "",
              f"Collection error records: {len(errors)} (run.json: {report['errors']}). Late replies over all seats: {late}; "
              f"dead bots: {dead}. {fault_text} No failed seat or fixture was silently removed. Match utility means in the "
              "original receipts are descriptive collection outputs, not paired strength evidence.", "",
              "## Backup", "",
              f"Archive `T0037_colab_seed{schedule['seed']}_20261012.tar.gz`: {receipt['bytes']} bytes; SHA-256 "
              f"`{receipt['archive_sha256']}`. All {receipt['verified_files']} archived payload file hashes were verified "
              f"locally, with {receipt['matches']} match files and {receipt['rows']} row files. The archive and its extracted "
              "copy are in the Player5 workspace backup directory (`ctf-arena/Player5/workspace/backups/`, synchronized by "
              "OneDrive). Raw data and rebuildable binaries are not committed. `BACKUP_MANIFEST.json` records payload "
              "hashes; `BACKUP_VERIFIED.json` records the verification result and this branch status. The archive excludes "
              "rebuildable binaries and Python bytecode but retains build manifests.", "",
              "Reproduce: `python bots/p_only/report/decision_collection/colab_20261012/generate_report.py --backup "
              "<backup-directory>` (the directory contains `t0037/`, `VERIFIED.json` and `connection_checkpoint.json`). "
              "It verifies source hashes before copying original receipts and generating this report. No new seed is "
              "consumed.", "",
              "## Status", "",
              f"This corpus is closed as a historical, exploratory whole-match source. Seed {schedule['seed']} stays "
              "consumed. Any later use must keep it separate from the Windows host pilot and label it Colab; branch labels "
              "could only come from a new, separately frozen collection.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified backup and generated stage-1 report:", len(rows), "matches")


if __name__ == "__main__":
    main()
