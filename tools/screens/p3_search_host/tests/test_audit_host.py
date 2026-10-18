"""Regression tests for tools/screens/p3_search_host/audit_host.py (task T0084).

    python -m unittest discover -s tools/screens/p3_search_host/tests

A small synthetic screen (two team counts, an arm, its A/A copy on the same binary, a direct arm) written in the
host layout: receipt files, results.jsonl and gzipped match records named <fixture_id>-<sha256(arm)[:10]>.
No compiler, arena or host is needed.
"""
from __future__ import annotations

import copy
import gzip
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import audit_host  # noqa: E402

COMMITTED = HERE.parents[3] / "reports" / "host" / "agent_v2_p3_search"
DIRECT, SEARCH = "x_direct", "x_search"
AA = SEARCH + "~aa"
ARMS = (DIRECT, SEARCH, AA)
BINARY = {DIRECT: "d" * 64, SEARCH: "s" * 64, AA: "s" * 64}
UTILITY = {  # (teams, map) -> utilities per arm
    (2, 0): (0.50, 0.60, 0.60), (2, 1): (0.40, 0.45, 0.45), (2, 2): (0.70, 0.65, 0.70),
    (3, 0): (0.30, 0.10, 0.10), (3, 1): (0.60, 0.55, 0.40), (3, 2): (0.20, 0.25, 0.25)}


def fixture_id(teams, m):
    return f"s1-n{teams:02d}-m{m:05d}"


def counter_line(used, fallback, overrides=None):
    line = f"AGENTV2 turns {used + fallback} refused 0 inconsistent 0 search_used {used} search_fallback {fallback} " \
           f"leaves 0 max_search_us {1000 + used} critic fp32"
    return line + ("" if overrides is None else f" overrides {overrides} gate_kept {used - overrides}")


class Screen:
    """Writes the synthetic screen under a temporary directory."""

    def __init__(self, root: Path):
        self.root, self.data, self.receipt = root, root / "data", root / "receipt"
        self.fixtures = [{"fixture_id": fixture_id(t, m), "teams": t, "map_seed": 100 * t + m, "flag_seed": 1,
                          "seat": 0, "opponents": ["Bot%d" % k for k in range(1, t)]} for t, m in UTILITY]
        self.rows = []
        for (t, m), utils in UTILITY.items():
            for arm, u in zip(ARMS, utils):
                score = [round(10 * u)] + [5] * (t - 1)
                self.rows.append({"fixture_id": fixture_id(t, m), "arm": arm, "identity": "id-" + arm, "teams": t,
                                  "map_seed": 100 * t + m, "score": score, "utility": u, "faults": 0, "late": 0,
                                  "opponent_faults": 0, "opponent_late": 0, "max_ms": 10.0})
        self.fault = {"fixture_id": fixture_id(3, 1), "arm": AA}        # the A/A copy faults once, late once
        for r in self.rows:
            if r["fixture_id"] == self.fault["fixture_id"] and r["arm"] == AA:
                r.update(faults=1, late=1, max_ms=31.5)
        self.complete = copy.deepcopy(self.rows)                       # the receipt is the complete screen's
        self.matches = {}
        for (t, m) in UTILITY:
            for arm in ARMS:
                used = 0 if arm == DIRECT else (40 if t == 2 else 10) + (m if arm == AA else 0)
                fallback = 0 if arm == DIRECT else 40 - used + 40
                tail = ["AGENTV2 carrier agent_v2_v1 search 1",
                        counter_line(used, fallback, None if arm == DIRECT else used // 2)]
                faults = [[47, 0, "timeout (>50 ms)"]] if (fixture_id(t, m), arm) == tuple(self.fault.values()) else []
                bots = [{"late": 1 if faults else 0, "stderr_tail": tail}] + [{"late": 0, "stderr_tail": []}] * (t - 1)
                self.matches[(fixture_id(t, m), arm)] = {
                    "teams": t, "seats": [{"name": "@candidate", "sha256": BINARY[arm]}]
                    + [{"name": "Bot%d" % k, "sha256": "o" * 64} for k in range(1, t)], "bots": bots, "faults": faults}

    def write(self, receipt_override=None):
        self.receipt.mkdir(parents=True, exist_ok=True)
        (self.data / "matches").mkdir(parents=True, exist_ok=True)
        (self.receipt / "RUN.json").write_text(json.dumps({"config": {"turn_ms": 50, "work_dir": str(self.data)}}))
        (self.receipt / "schedule.json").write_text(json.dumps({"fixtures": self.fixtures}))
        (self.data / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        for (fx, arm), m in self.matches.items():
            p = self.data / "matches" / audit_host.match_name(fx, m["teams"], arm)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(json.dumps(m).encode()))
        screen = self.receipt_doc()
        if receipt_override:
            receipt_override(screen)
        (self.receipt / "screen.json").write_text(json.dumps(screen))
        return self

    def receipt_doc(self):
        """A receipt that agrees with the rows: the screen's own numbers, rounded as screen.json rounds them."""
        rows = {}
        for r in self.complete:
            rows.setdefault(r["fixture_id"], {})[r["arm"]] = r
        teams = {f["fixture_id"]: f["teams"] for f in self.fixtures}

        def rounded(d):
            return {k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v] if isinstance(v, list) else v)
                    for k, v in d.items()}

        def comparison(a, b, z):
            c = audit_host.compare(rows, teams, a, b, ["2", "2-3"], z)
            return {"overall": rounded(c["overall"]), "bands": {k: rounded(v) for k, v in c["bands"].items()}}

        arms = {arm: {"games": 6, "binary_sha256": BINARY[arm], "faults": 0, "late_replies": 0, "opponent_faults": 0,
                      "opponent_late": 0} for arm in ARMS}
        arms[AA].update(faults=1, late_replies=1)
        return {"seed": 1, "schedule": "ab" * 32, "arms": arms, "identities": {a: "id-" + a for a in ARMS},
                "multiplicity": {"method": "bonferroni", "alpha": 0.05, "comparisons": 1, "z": 2.394},
                "comparisons": {f"{SEARCH} - {DIRECT}": comparison(SEARCH, DIRECT, 2.394)},
                "rerun_noise": {f"{AA} - {SEARCH}": comparison(AA, SEARCH, None)}}

    def audit(self, indexes=()):
        return audit_host.audit(self.data, self.receipt, list(indexes))


class HostAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.screen = Screen(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_comparison_is_the_equal_count_mean_and_reproduces_the_receipt(self):
        doc = self.screen.write().audit()
        c = doc["comparisons"][f"{SEARCH} - {DIRECT}"]
        n2, n3 = (0.10 + 0.05 - 0.05) / 3, (-0.20 - 0.05 + 0.05) / 3
        self.assertAlmostEqual(c["overall"]["mean"], (n2 + n3) / 2)
        self.assertAlmostEqual(c["bands"]["2"]["mean"], n2)
        self.assertEqual(c["overall"]["z"], 2.394)
        self.assertTrue(c["receipt"]["reproduces"], c["receipt"])
        self.assertTrue(doc["comparisons"][f"{AA} - {SEARCH}"]["receipt"]["reproduces"])
        self.assertNotIn("ci", doc["comparisons"][f"{AA} - {SEARCH}"]["overall"])      # the A/A row is 95% only

    def test_a_receipt_that_disagrees_is_reported(self):
        def shift(screen):
            screen["comparisons"][f"{SEARCH} - {DIRECT}"]["bands"]["2"]["mean"] += 0.001
            screen["comparisons"][f"{SEARCH} - {DIRECT}"]["overall"]["verdict"] = "helps"
        rc = self.screen.write(shift).audit()["comparisons"][f"{SEARCH} - {DIRECT}"]["receipt"]
        self.assertFalse(rc["reproduces"])
        self.assertEqual(rc["mismatches"], ["band 2 mean", "overall fixtures/verdict"])

    def test_the_aa_copy_is_read_from_its_own_records(self):
        doc = self.screen.write().audit()
        tel = doc["search_telemetry"]["per_arm"]
        self.assertEqual(tel[SEARCH]["2"]["games"], 3)                    # not 6: same binary, different arm key
        self.assertEqual(tel[SEARCH]["2"]["search_used"], 120)
        self.assertEqual(tel[AA]["2"]["search_used"], 123)
        self.assertAlmostEqual(tel[SEARCH]["3"]["completion"], 30 / 240)
        self.assertAlmostEqual(tel[SEARCH]["3"]["override_rate"], 15 / 30)
        self.assertIsNone(tel[DIRECT]["2"]["completion"])                 # no search: no denominator, not 0%
        (fault,) = doc["faults"]["details"]
        self.assertEqual(fault["match_record"], audit_host.match_name(fault["fixture_id"], 3, AA))
        self.assertEqual(fault["events"], [{"turn": 47, "seat": 0, "bot": "@candidate", "reason": "timeout (>50 ms)",
                                            "role": "candidate"}])
        aa = doc["aa_rerun"][f"{AA} - {SEARCH}"]
        self.assertEqual(aa["total"]["identical"] + aa["total"]["differing"], 6)
        self.assertEqual(aa["per_count"]["3"]["differing_counters_differ"], 1)        # m1 differs, admission too
        self.assertEqual(aa["per_count"]["2"]["differing_counters_differ"], 1)        # m2
        self.assertEqual(aa["per_count"]["2"]["identical_counters_differ"], 1)        # m1: same score, more search

    def test_fault_totals_are_checked_against_the_receipt(self):
        doc = self.screen.write(lambda s: s["arms"][SEARCH].update(faults=2)).audit()
        self.assertTrue(doc["faults"]["per_arm"][AA]["matches_receipt"])
        self.assertFalse(doc["faults"]["per_arm"][SEARCH]["matches_receipt"])
        self.assertEqual(doc["faults"]["candidate_fault_games_max_ms_below_deadline"], 1)

    def test_missing_match_records_are_unavailable_not_zero(self):
        self.screen.write()
        for p in (self.screen.data / "matches").rglob("*.json.gz"):
            p.unlink()
        (self.screen.data / "matches" / "n02").rmdir()
        (self.screen.data / "matches" / "n03").rmdir()
        (self.screen.data / "matches").rmdir()
        doc = self.screen.audit()
        self.assertEqual(doc["search_telemetry"], {"status": "unavailable", "reason": "no matches directory"})
        self.assertEqual(doc["faults"]["details"][0]["match_record"], None)
        self.assertEqual(doc["aa_rerun"][f"{AA} - {SEARCH}"]["counters"], "unavailable: no matches directory")
        md = audit_host.render(doc)
        self.assertIn("Unavailable: no matches directory.", md)
        self.assertIn("record missing", md)

    def test_a_count_with_a_missing_record_gives_no_rate(self):
        self.screen.write()
        (self.screen.data / "matches" / audit_host.match_name(fixture_id(3, 2), 3, SEARCH)).unlink()
        doc = self.screen.audit()
        d = doc["search_telemetry"]["per_arm"][SEARCH]["3"]
        self.assertEqual((d["with_counters"], d["games"], d["complete"]), (2, 3, False))
        self.assertIsNone(d["completion"])
        self.assertIsNotNone(doc["search_telemetry"]["per_arm"][SEARCH]["2"]["completion"])
        self.assertEqual(doc["match_records"], {"status": "computed", "expected": 18, "missing": 1})
        self.assertEqual(doc["aa_rerun"][f"{AA} - {SEARCH}"]["per_count"]["3"]["counters_unavailable"], 1)
        self.assertIn("| 3 | n/a | n/a | n/a | 2/3 |", audit_host.render(doc))

    def test_rows_must_cover_the_receipt_exactly(self):
        base = copy.deepcopy(self.screen.rows)
        for name, rows in [("missing", base[1:]), ("duplicate", base + base[:1]),
                           ("identity", [dict(base[0], identity="other")] + base[1:]),
                           ("unknown arm", base + [dict(base[0], arm="y")])]:
            with self.subTest(name):
                self.screen.rows = rows
                with self.assertRaises(SystemExit):
                    self.screen.write().audit()

    def test_main_refuses_without_results_and_writes_nothing(self):
        out = Path(self.tmp.name) / "out"
        err = io.StringIO()
        with redirect_stderr(err):
            code = audit_host.main(["--data", str(Path(self.tmp.name) / "absent"), "--receipt", str(COMMITTED / "screen"),
                                    "--out-dir", str(out)])
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        self.assertIn("results not found", err.getvalue())

    def test_main_writes_json_and_its_rendering(self):
        self.screen.write()
        out = Path(self.tmp.name) / "out"
        self.assertEqual(audit_host.main(["--data", str(self.screen.data), "--receipt", str(self.screen.receipt),
                                          "--out-dir", str(out)]), 0)
        doc = json.loads((out / "EVIDENCE_AUDIT.json").read_text())
        self.assertEqual((out / "EVIDENCE_AUDIT.md").read_text(), audit_host.render(doc))
        self.assertIn("No corpus index was given: overlap not audited.", audit_host.render(doc))

    def test_corpus_overlap_counts_seats_and_shared_maps(self):
        self.screen.write()
        idx = Path(self.tmp.name) / "index.json"
        idx.write_text(json.dumps({"matches": [{"map_seed": 200}, {"map_seed": 999}],
                                   "manifest": {"seats_by_bot": {"Bot1": 7}}}))
        o = self.screen.audit([idx, Path(self.tmp.name) / "absent.json"])["corpus_overlap"]
        self.assertEqual(o["screen_opponent_seats"], {"Bot1": 6, "Bot2": 3})
        computed, absent = o["corpora"]
        self.assertEqual((computed["map_seed_overlap"], computed["seats_by_opponent"]), (1, {"Bot1": 7, "Bot2": 0}))
        self.assertEqual(absent["status"], "unavailable")

    def test_rendering_states_verdicts_from_the_data(self):
        doc = self.screen.write().audit()
        md = audit_host.render(doc)
        row = next(line for line in md.splitlines() if line.startswith(f"| `{SEARCH} - {DIRECT}`"))
        self.assertIn(f"| {doc['comparisons'][f'{SEARCH} - {DIRECT}']['overall']['verdict']} |", row)
        self.assertIn("not evidence of equivalence", md)

    def test_committed_audit_is_current(self):
        path = COMMITTED / "EVIDENCE_AUDIT.json"
        if not path.is_file():
            self.skipTest("no committed host audit yet (host-only: run audit_host.py on the host)")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], audit_host.SCHEMA)
        self.assertEqual((COMMITTED / "EVIDENCE_AUDIT.md").read_text(encoding="utf-8"), audit_host.render(doc))


if __name__ == "__main__":
    unittest.main()
