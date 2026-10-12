"""Own-action decision learning (task T0034): dataset contract, history, leakage, support, noise, rule and identities.

Fixtures are committed recorded matches (roster/report/results/matches) with *synthetic* continuation outcomes:
the roots, prefixes and candidate orders are real, the scores are made up. Nothing here plays a game or measures
strength.
"""
from __future__ import annotations

import copy
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion import matchlog, native
from fusion.labels import pairwise_outcome
from fusion.paths import REPO
from fusion.protocol import MOVES, parse_block
from fusion.schema import layout
from training import decision_dataset as D
from training import decision_noise as N
from training import evaluate_decisions as E
from training import train_decision_ranker as T
from training.dataset import split_of

ROSTER = REPO / "roster" / "report" / "results" / "matches"


def pick_matches(per_split=2, max_teams=4):
    out = {"train": [], "val": [], "test": []}
    for p in sorted(ROSTER.rglob("*.json.gz")):
        m = matchlog.load(p)
        s = split_of(f"map-{m['map_seed']}")
        if m["teams"] <= max_teams and len(out[s]) < per_split:
            out[s].append(p)
    return [p for s in out.values() for p in s]


def live_turns(match, seat, count=2, start=12):
    turns = []
    for t in range(start, len(match["blocks"]) - 5):
        units = parse_block(match["blocks"][t - 1]).units
        if sum(units[3 * seat + k][2] >= 0 for k in range(3)) >= 2 and match["replies"][t - 1][seat]:
            turns.append(t)
            if len(turns) == count:
                break
    return turns


def deviations(rec, line, k_max=3):
    L = layout()
    legal = L.get(rec, "legal").astype(bool)
    origin = L.get(rec, "origin")
    tok = line.split()
    out = []
    for k in range(3):
        if origin[k] < 0:
            continue
        for m in range(5):
            if legal[k, m] and MOVES[m] != tok[1 + 2 * k]:
                alt = list(tok)
                alt[1 + 2 * k] = MOVES[m]
                s = " ".join(alt)
                if s not in out and s != line:
                    out.append(s)
    return out[:k_max]


CARRIER = "c" * 64


def build_fixture(root: Path, reps=2, seed=0, mutate=None, carrier=CARRIER, scores=True):
    """Matches with a designated seat 0 and branch roots with synthetic outcomes; returns (branches, matches).

    With scores, the carrier logged a shortlist score for the incumbent and every deviation but the last one
    (so one candidate stays unscored), listed in reverse candidate order to exercise the order alignment."""
    rng = np.random.default_rng(seed)
    mdir, bdir = root / "matches", root / "branches"
    for p in pick_matches():
        m = matchlog.load(p)
        m = copy.deepcopy(m)
        m["seats"][0] = dict(m["seats"][0], name="@candidate", sha256=carrier or m["seats"][0]["sha256"])
        matchlog.save(mdir / f"n{m['teams']:02d}" / f"{m['match_id']}.json.gz", m)
        for t in live_turns(m, 0):
            rec = D.root_record(m, 0, t)
            line0 = F.normalize_line(m["replies"][t - 1][0])
            cands = [line0] + deviations(rec, line0)
            branches = []
            for c, line in enumerate(cands):
                for r in range(reps):
                    score = rng.integers(0, 6, m["teams"]).tolist()
                    branches.append({"candidate": c, "repeat": r, "on_policy": c == 0, "line": line, "score": score,
                                     "outcome": float(pairwise_outcome(score)[0]), "prefix_mismatch": 0, "prefix_faults": 0,
                                     "faults": [], "root_block": m["blocks"][t - 1], "next_block": m["blocks"][t]})
            row = {"schema": "ctf-fusion-branches-1", "match_id": m["match_id"], "teams": m["teams"], "map_seed": m["map_seed"],
                   "seat": 0, "turn": t, "init": m["init"], "carrier_sha256": m["seats"][0]["sha256"], "candidates": cands,
                   "branches": branches, "candidate_mode": "single",
                   "root_outcome": float(pairwise_outcome(m["result"]["score"])[0])}
            if scores:
                orders = cands[:-1][::-1] if len(cands) > 2 else cands[::-1]
                row["carrier_scores"] = {"version": D.SCORES_VERSION, "k": len(orders), "orders": orders,
                                         "scores": [round(float(x), 3) for x in rng.normal(0, 10, len(orders))]}
            if mutate:
                row = mutate(row) or row
            out = bdir / f"n{m['teams']:02d}" / f"{m['match_id']}-t{t:03d}.json.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                json.dump(row, fh)
    return bdir, mdir


def make_dataset(tmp: Path, extra=(), **kw) -> Path:
    bdir, mdir = build_fixture(tmp, **kw)
    out = tmp / "data"
    D.main(["--branches", str(bdir), "--matches", str(mdir), "--out", str(out), "--successors",
            "--audit", str(REPO / "bots/p_only/report/v_data/DATA_AUDIT.json"), *extra])
    return out


def collection_receipts(tmp: Path, binary=CARRIER) -> tuple[Path, Path]:
    """A collection's build manifest and identities for a fixture carrier (fake model files: family unverified)."""
    inputs = [{"path": "bots\\p_only\\bots\\variants\\fixture.cpp", "sha256": "1" * 64},
              {"path": "bots\\p_only\\bots\\v12_ponly.hpp", "sha256": "2" * 64},
              {"path": "inference\\models_fixture\\registry.hpp", "sha256": "3" * 64},
              {"path": "inference\\models_fixture\\weights_general.hpp", "sha256": "4" * 64}]
    man = {"variants": {"fixture": {"name": "fixture", "source": "bots\\p_only\\bots\\variants\\fixture.cpp",
                                    "value_tier": None, "fixed_samples": 0, "bundle_sha256": "b" * 64,
                                    "binary_sha256": binary, "inputs": inputs}}}
    ids = {"fixture": {"digest": "x", "components": {"variant": "fixture", "bundle_sha256": "b" * 64, "binary_sha256": binary,
                                                    "inputs_sha256": "9" * 64, "limits_ms": [2000, 50]}}}
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "manifest.json").write_text(json.dumps(man))
    (tmp / "identities.json").write_text(json.dumps(ids))
    return tmp / "manifest.json", tmp / "identities.json"


def handmade(roots):
    """A decision dict from [(teams, map, [(key, outcomes, support[, carrier score]), ...]), ...]; candidate 0 is the
    incumbent."""
    d = {k: [] for k in D.ROOT_KEYS + D.CAND_KEYS}
    rmax = max(len(c[1]) for _, _, cs in roots for c in cs) or 1
    for r, (teams, mp, cs) in enumerate(roots):
        base = len(d["cand_root"])
        for k, v in (("root_match", f"m{r}"), ("root_map", mp), ("root_teams", teams), ("root_seat", 0), ("root_turn", 10),
                     ("root_carrier", "x"), ("root_rec", np.zeros(2, np.int16)), ("root_first", base),
                     ("root_count", len(cs)), ("root_incumbent", base), ("root_original_outcome", np.nan),
                     ("root_prefix_mismatch", 0), ("root_prefix_faults", 0), ("root_shortlist_k", -1), ("root_file", "f")):
            d[k].append(v)
        for c, (key, outs, sup, *cs_) in enumerate(cs):
            o = np.full(rmax, np.nan)
            o[:len(outs)] = outs
            n = len(outs)
            for k, v in (("cand_root", r), ("cand_line", key), ("cand_key", key), ("cand_source", "x"), ("cand_on_policy", c == 0),
                         ("cand_move", np.zeros(3)), ("cand_act", np.zeros(3)), ("cand_target", np.zeros(3)), ("cand_slot", np.zeros(3)),
                         ("cand_flags", 0 if sup else F.PARSE_FAIL), ("cand_dup_of", -1), ("cand_support", sup), ("cand_n", n),
                         ("cand_score", np.zeros((rmax, 15))), ("cand_outcome", o), ("cand_faults", np.zeros(rmax)),
                         ("cand_y", float(np.mean(outs)) if n else np.nan), ("cand_var", float(np.var(outs, ddof=1)) if n >= 2 else np.nan),
                         ("cand_carrier_score", cs_[0] if cs_ else np.nan)):
                d[k].append(v)
    return D.to_arrays(d)


class DatasetContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data = make_dataset(cls.tmp / "a")
        cls.index = D.index_of(cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_roots_and_candidates(self):
        d = D.load(self.data, "train")
        self.assertGreater(len(d["root_match"]), 0)
        mdir = self.tmp / "a" / "matches"
        for r in range(len(d["root_match"])):
            m = matchlog.load(D.find_match(mdir, str(d["root_match"][r]), int(d["root_teams"][r])))
            t, seat = int(d["root_turn"][r]), int(d["root_seat"][r])
            full = native.extract(matchlog.seat_stream(m, seat))           # whole game: the prefix row must be causal
            np.testing.assert_array_equal(d["root_rec"][r], full[(t - 1) * m["teams"] + seat])
            first, count = int(d["root_first"][r]), int(d["root_count"][r])
            self.assertEqual(int(d["root_incumbent"][r]), first)
            self.assertEqual(d["cand_line"][first], F.normalize_line(m["replies"][t - 1][seat]))
            self.assertTrue(d["cand_on_policy"][first] and not d["cand_on_policy"][first + 1:first + count].any())
            self.assertEqual(d["cand_source"][first], "recorded")
            for c in range(first, first + count):
                n = int(d["cand_n"][c])
                self.assertEqual(n, 2)
                us = [pairwise_outcome(s[:m["teams"]])[seat] for s in d["cand_score"][c][:n]]
                np.testing.assert_allclose(d["cand_outcome"][c][:n], us)
                self.assertAlmostEqual(d["cand_y"][c], np.mean(us))
                self.assertAlmostEqual(d["cand_var"][c], np.var(us, ddof=1))
        self.assertFalse(self.index["authoritative"])
        self.assertTrue(any("pending" in r for r in self.index["exploratory_reasons"]))

    def test_carrier_scores_preserved(self):
        bdir = self.tmp / "a" / "branches"
        rows = {}
        for f in bdir.rglob("*.json.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                rows[f.name] = json.load(fh)
        checked = 0
        for s in D.SPLITS:
            d = D.load(self.data, s, test_unlocked=True)
            for r in range(len(d["root_match"])):
                row = rows[str(d["root_file"][r])]
                logged = dict(zip(row["carrier_scores"]["orders"], row["carrier_scores"]["scores"]))
                self.assertEqual(int(d["root_shortlist_k"][r]), row["carrier_scores"]["k"])
                first, count = int(d["root_first"][r]), int(d["root_count"][r])
                for c in range(first, first + count):
                    want = logged.get(str(d["cand_line"][c]), np.nan)
                    np.testing.assert_equal(d["cand_carrier_score"][c], want)
                    checked += 1
        self.assertGreater(checked, 0)
        self.assertEqual(self.index["carrier_scores"]["roots_with_scores"], self.index["carrier_scores"]["roots"])

    def test_splits_follow_maps(self):
        for s in D.SPLITS:
            d = D.load(self.data, s, test_unlocked=True)
            self.assertTrue(all(split_of(f"map-{m}") == s for m in d["root_map"].tolist()))
        with self.assertRaises(SystemExit):
            D.load(self.data, "test")

    def test_features_ignore_outcomes_and_successors(self):
        other = make_dataset(self.tmp / "b", seed=99)       # same roots, other scores
        a, b = D.load(self.data, "train"), D.load(other, "train")
        self.assertFalse(np.array_equal(a["cand_y"], b["cand_y"]))
        for x, y in zip(D.scorer_inputs(a), D.scorer_inputs(b)):
            np.testing.assert_array_equal(x, y)

    def test_future_blocks_do_not_reach_the_root(self):
        p = pick_matches()[0]
        m = matchlog.load(p)
        t = live_turns(m, 0)[0]
        before = D.root_record(m, 0, t)
        m2 = copy.deepcopy(m)
        m2["blocks"] = m2["blocks"][:t]          # no future at all
        np.testing.assert_array_equal(before, D.root_record(m2, 0, t))

    def test_missing_history_rejected(self):
        m = matchlog.load(pick_matches()[0])
        t = live_turns(m, 0)[0]
        short = dict(m, blocks=m["blocks"][:t - 1])
        with self.assertRaises(D.LinkError):
            D.root_record(short, 0, t)
        holed = dict(m, blocks=list(m["blocks"]))
        holed["blocks"][3] = ""
        with self.assertRaises(D.LinkError):
            D.root_record(holed, 0, t)

    def test_noise_report(self):
        out = self.tmp / "noise"
        N.main(["--data", str(self.data), "--out", str(out)])
        rep = json.loads(out.with_suffix(".json").read_text())
        tr = rep["splits"]["train"]
        self.assertIsNotNone(tr["noise"]["within_action_variance"])
        self.assertIn("maps", tr["coverage"])
        self.assertIn("mde", tr["power"])
        with self.assertRaises(SystemExit):
            N.main(["--data", str(self.data), "--splits", "test", "--out", str(out)])


class Linkage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, mutate):
        bdir, mdir = build_fixture(self.tmp, reps=1, mutate=mutate)
        return D.build(sorted(bdir.rglob("*.json.gz")), mdir, log=lambda *_: None)

    def test_rejections(self):
        cases = {
            "map": lambda r: r.update(map_seed=r["map_seed"] + 1),
            "carrier": lambda r: r.update(carrier_sha256="0" * 64),
            "seat": lambda r: r.update(seat=1),
            "init": lambda r: r.update(init=r["init"].replace("INIT", "INIT ", 1)),
            "root block": lambda r: [b.update(root_block="TURN 1\n") for b in r["branches"]],
            "on-policy order": lambda r: r.update(candidates=[r["candidates"][1], r["candidates"][0], *r["candidates"][2:]]),
            "on-policy flag": lambda r: [b.update(on_policy=True) for b in r["branches"] if b["candidate"] == 1],
            "recorded source": lambda r: r.update(sources=["move1"] * len(r["candidates"])),
            "duplicate": lambda r: r.update(candidates=[*r["candidates"][:-1], r["candidates"][1]]),
            "outcome": lambda r: r["branches"][0].update(outcome=r["branches"][0]["outcome"] + 0.1),
            "score order": lambda r: r["carrier_scores"]["orders"].__setitem__(0, "999 S - S - S -"),
            "score incumbent": lambda r: r.update(carrier_scores={**r["carrier_scores"], "k": 1,
                                                                   "orders": [r["candidates"][1]], "scores": [1.0]}),
            "score finite": lambda r: r["carrier_scores"]["scores"].__setitem__(0, float("nan")),
            "score k": lambda r: r["carrier_scores"].update(k=r["carrier_scores"]["k"] + 1),
            "score twice": lambda r: r.update(carrier_scores={**r["carrier_scores"], "k": 2, "orders": [r["candidates"][0]] * 2,
                                                               "scores": [1.0, 2.0]}),
            "score version": lambda r: r["carrier_scores"].update(version="other"),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                shutil.rmtree(self.tmp, ignore_errors=True)
                with self.assertRaises(D.LinkError):
                    self.build(lambda r, fn=fn: (fn(r), r)[1])

    def test_one_carrier_per_dataset(self):
        bdir, mdir = build_fixture(self.tmp, reps=1, carrier=None)   # each match keeps its own seat-0 binary
        with self.assertRaises(D.LinkError):
            D.build(sorted(bdir.rglob("*.json.gz")), mdir, log=lambda *_: None)

    def test_split_contamination(self):
        bdir, mdir = build_fixture(self.tmp, reps=1)
        data = self.tmp / "data"
        D.main(["--branches", str(bdir), "--matches", str(mdir), "--out", str(data)])
        tr = dict(np.load(data / "train" / "decision.npz"))
        va = dict(np.load(data / "val" / "decision.npz"))
        va["root_map"] = va["root_map"].copy()
        va["root_map"][0] = tr["root_map"][0]          # a train map leaks into val
        np.savez_compressed(data / "val" / "decision.npz", **va)
        index = json.loads((data / "index.json").read_text())
        index["splits"]["val"]["sha256"] = D.sha256_file(data / "val" / "decision.npz")
        (data / "index.json").write_text(json.dumps(index))
        with self.assertRaises(SystemExit):
            D.check_isolation(data)
        with self.assertRaises(SystemExit):
            D.load(data, "val")


class Authority(unittest.TestCase):
    """Authoritative ingestion binds receipts to the exact source bytes, and the carrier config to its collection."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.bdir, cls.mdir = build_fixture(cls.tmp / "a", reps=1)
        cls.dig = D.source_digests(cls.bdir, cls.mdir)
        man, ids = collection_receipts(cls.tmp / "receipts")
        cls.cfg = D.derive_carrier_config(str(man), str(ids), "fixture", "p2", "admitted")
        cls.cfg_path = cls.tmp / "carrier.json"
        cls.cfg_path.write_text(json.dumps(cls.cfg))
        cls.scope = cls.tmp / "scope.json"
        cls.scope.write_text(json.dumps({"verification_scope": "full_original_manifest_bytes"}))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def receipt(self, dig=None, status="passed", flags=(True, True, True), legacy=None):
        dig = dig or self.dig
        a = {"source_alignment_status": status,
             "rank_npz": {s: {"source_row_alignment_verified": f} for s, f in zip(D.SPLITS, flags)},
             "decision_source": {k: dig[k] for k in ("version", "branches_sha256", "matches_sha256")}}
        if legacy is not None:
            a["root_content_manifest_sha256"] = legacy
        path = self.tmp / f"audit_{len(list(self.tmp.glob('audit_*')))}.json"
        path.write_text(json.dumps(a))
        return path

    def run_build(self, audit, branches=None, cfg=None, authoritative=True, scope=None, extra=()):
        out = self.tmp / f"data_{len(list(self.tmp.glob('data_*')))}"
        D.main(["--branches", str(branches or self.bdir), "--matches", str(self.mdir), "--out", str(out),
                "--audit", str(audit), "--manifest-receipt", str(scope or self.scope),
                "--carrier-config", str(cfg or self.cfg_path), *(["--authoritative"] if authoritative else []), *extra])
        return D.index_of(out)

    def test_passing_receipt(self):
        index = self.run_build(self.receipt(legacy=self.dig["root_content_manifest_sha256"]))
        self.assertTrue(index["authoritative"])
        self.assertEqual(index["source"]["digests"]["branches_sha256"], self.dig["branches_sha256"])
        self.assertTrue(index["carrier"]["config"]["verification"]["verified"])

    def test_foreign_receipt(self):
        other_b, other_m = build_fixture(self.tmp / "other", reps=1, seed=5)
        with self.assertRaises(SystemExit):
            self.run_build(self.receipt(D.source_digests(other_b, other_m)))

    def test_changed_source_bytes(self):
        copy_dir = self.tmp / "changed"
        shutil.copytree(self.bdir, copy_dir)
        f = sorted(copy_dir.rglob("*.json.gz"))[0]
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        row["note"] = "edited after the audit"
        with gzip.open(f, "wt", encoding="utf-8") as fh:
            json.dump(row, fh)
        with self.assertRaises(SystemExit):
            self.run_build(self.receipt(), branches=copy_dir)

    def test_contradictory_or_unpassed_receipts(self):
        for kw in ({"flags": (True, False, True)}, {"status": "pending_host_reaudit"},
                   {"status": "pending_host_reaudit", "flags": (True, True, True)}, {"legacy": "0" * 64}):
            with self.subTest(kw), self.assertRaises(SystemExit):
                self.run_build(self.receipt(**kw))
        with self.assertRaises(SystemExit):     # the real T0010 receipt is pending and has no source digests
            self.run_build(REPO / "bots/p_only/report/v_data/DATA_AUDIT.json")
        index = self.run_build(REPO / "bots/p_only/report/v_data/DATA_AUDIT.json", authoritative=False)
        self.assertFalse(index["authoritative"])
        self.assertTrue(any("pending" in r for r in index["exploratory_reasons"]))

    def test_selected_manifest_scope(self):
        adapter = REPO / "bots/p_only/report/v_data/MANIFEST_METADATA_ADAPTER.json"
        with self.assertRaises(SystemExit):
            self.run_build(self.receipt(), scope=adapter)
        index = self.run_build(self.receipt(), scope=adapter, extra=["--accept-selected-scope"])
        self.assertTrue(index["authoritative"] and index["receipts"]["accepted_selected_manifest_scope"])

    def test_carrier_config_conflicts(self):
        D.check_carrier_config(self.cfg, [CARRIER])
        cases = {
            "roots of another binary": (lambda c: c, ["d" * 64]),
            "policy file": (lambda c: c["opponent_policy"]["files"].update({"inference/models_fixture/weights_general.hpp": "5" * 64}), None),
            "omitted policy file": (lambda c: c["opponent_policy"]["files"].pop("inference/models_fixture/registry.hpp"), None),
            "budget": (lambda c: c["budget"].update(turn_ms=60), None),
            "search inputs": (lambda c: c["search"].update(inputs_sha256="8" * 64), None),
            "fixed samples": (lambda c: c["search"].update(fixed_samples=128), None),
            "value": (lambda c: c["value"].update(enabled=True), None),
            "missing budget": (lambda c: c.pop("budget"), None),
            "status": (lambda c: c.update(status="final"), None),
            "receipt hash": (lambda c: c["collection"].update(manifest_sha256="0" * 64), None),
        }
        for name, (fn, carriers) in cases.items():
            cfg = copy.deepcopy(self.cfg)
            fn(cfg)
            with self.subTest(name), self.assertRaises(SystemExit):
                D.check_carrier_config(cfg, carriers or [CARRIER])

    def test_config_binary_must_be_the_roots_carrier(self):
        man, ids = collection_receipts(self.tmp / "receipts_d", binary="d" * 64)
        cfg = self.tmp / "carrier_d.json"
        cfg.write_text(json.dumps(D.derive_carrier_config(str(man), str(ids), "fixture", "p2", "admitted")))
        for authoritative in (True, False):       # an inconsistent config is refused even for exploratory data
            with self.subTest(authoritative=authoritative), self.assertRaises(SystemExit):
                self.run_build(self.receipt(), cfg=cfg, authoritative=authoritative)

    def test_t0010_receipts(self):
        rep = "bots/p_only/report/v_data/"
        binary = "6cbd5d34610f7946eebee56713992eae81c283f3bbb6a6fc8a342a1dc3db47c1"
        cfg = D.derive_carrier_config(rep + "manifest.json", rep + "identities.json", "v12_ponly_xxl", "p2", "admitted")
        got = D.check_carrier_config(cfg, [binary])
        self.assertTrue(got["family_verified_from"].startswith("inference/models_xxl/weights_"))
        self.assertEqual((cfg["budget"], cfg["value"]["enabled"]), ({"first_ms": 2000, "turn_ms": 50}, False))
        wrong = D.derive_carrier_config(rep + "manifest.json", rep + "identities.json", "v12_ponly_xxl", "p3", "admitted")
        with self.assertRaises(SystemExit):
            D.check_carrier_config(wrong, [binary])


class Canonical(unittest.TestCase):
    """Support and canonicalization on a crafted root record."""

    def setUp(self):
        self.L = layout()
        rec = np.zeros(self.L.width, np.int16)
        rec[0], rec[1] = 30, 0
        o = lambda b: self.L.blocks[b][0]  # noqa: E731
        legal = np.zeros((3, 5), np.int16)
        legal[:, 0] = 1
        legal[0, 2] = 1                       # unit 0 may stay or go down
        act = np.zeros((3, 5, 4), np.int16)
        act[:, :, 0] = 1
        act[0, :, 3] = 1                      # unit 0 may attack
        cand = np.full((3, 8), -1, np.int16)
        cand[0] = [3, 4, 5, 6, 7, 8, 9, 10]    # slots full
        origin = np.array([5 * 20 + 5, 6 * 20 + 6, -1], np.int16)
        for name, arr in (("legal", legal), ("act_legal", act), ("cand", cand), ("origin", origin)):
            rec[o(name):o(name) + arr.size] = arr.reshape(-1)
        self.rec = rec
        units = [(i, i // 3, -1, -1, 0, -1, 0) for i in range(15)]
        units[0] = (0, 0, 5, 5, 100, -1, -1)
        units[1] = (1, 0, 6, 6, 100, -1, -1)
        units[11] = (11, 3, 6, 7, 100, -1, -1)  # hittable, beyond the full slots
        units[12] = (12, 4, 19, 19, 100, -1, -1)  # far away
        self.units = units

    def canon(self, line):
        return F.canonicalize(self.rec, self.L, line, 30, 0, self.units, 20)

    def test_flags(self):
        a = self.canon("30 D - S - S -")
        self.assertEqual((a["key"], a["flags"]), ("D-|S-|x", 0))
        b = self.canon("30 U - S - S -")                       # illegal move: settles as a stay
        self.assertEqual(b["key"], "S-|S-|x")
        self.assertTrue(b["flags"] & F.REQ_MOVE_ILLEGAL and b["flags"] & F.CANON_CHANGED)
        c = self.canon("30 S 4 S - S -")
        self.assertEqual((c["key"], int(c["slot"][0])), ("SA4|S-|x", 1))
        self.assertTrue(F.supported(c["flags"]))
        d = self.canon("30 S 11 S - S -")                       # outside the representation: unavailable
        self.assertTrue(d["flags"] & F.TARGET_OUT_OF_CAP)
        self.assertFalse(F.supported(d["flags"]))
        self.assertEqual(int(d["slot"][0]), -1)
        e = self.canon("30 S 12 S - S -")                       # cannot be hit: a wait
        self.assertEqual(e["key"], "S-|S-|x")
        self.assertTrue(e["flags"] & F.TARGET_INVALID)
        f = self.canon("30 S - S")
        self.assertFalse(F.supported(f["flags"]))
        g = self.canon("30 S - S - L 7")                        # off-board unit: its tokens do not matter
        self.assertEqual(g["key"], "S-|S-|x")

    def test_effective_duplicates(self):
        lines = ["30 S - S - S -", "30 U - S - S -", "30 D - S - S -"]
        cs = [self.canon(x) for x in lines]
        flags = [c["flags"] for c in cs]
        dup = F.mark_duplicates([c["key"] for c in cs], flags, incumbent=0)
        self.assertEqual(dup, [-1, 0, -1])
        self.assertFalse(F.supported(flags[1]))
        self.assertTrue(F.supported(flags[0]) and F.supported(flags[2]))

    def test_features_bind_the_order(self):
        a, b = self.canon("30 D - S - S -"), self.canon("30 S 4 S - S -")
        fa = F.candidate_features(self.rec, self.L, a["move"], a["act"], a["slot"])
        fb = F.candidate_features(self.rec, self.L, b["move"], b["act"], b["slot"])
        self.assertEqual(fa.shape, (F.CAND_DIM,))
        self.assertFalse(np.array_equal(fa, fb))
        self.assertEqual(fb[46], 1.0)


class Arithmetic(unittest.TestCase):
    def test_count_weights_and_clusters(self):
        np.testing.assert_allclose(E.count_weights(np.array([2, 2, 3])), [0.25, 0.25, 0.5])
        v = np.array([1.0, 1.0, -1.0, -1.0])
        ci = E.cluster_ci(v, np.full(4, 0.25), np.array([7, 7, 8, 8]))
        self.assertEqual(ci["clusters"], 2)
        self.assertAlmostEqual(ci["mean"], 0.0)

    def test_verdicts(self):
        self.assertEqual(E.verdict({"lo": 0.01, "hi": 0.03}, 0.002), "benefit")
        self.assertEqual(E.verdict({"lo": -0.03, "hi": -0.01}, 0.002), "harm")
        self.assertEqual(E.verdict({"lo": -0.001, "hi": 0.001}, 0.002), "negligible")
        self.assertEqual(E.verdict({"lo": -0.02, "hi": 0.03}, 0.002), "insufficient precision")

    def test_repeat_arithmetic(self):
        d = handmade([(2, 1, [("a", [0.0, 1.0], True), ("b", [1.0, 1.0, 0.5], True)]),
                      (3, 2, [("a", [0.5], True), ("b", [0.0], True)])])
        pooled = T.pooled_within_variance(d)
        self.assertAlmostEqual(pooled, (1 * 0.5 + 2 * (1 / 12)) / 3)
        items = T.gain_items(d, pooled)
        self.assertAlmostEqual(items["y"][0], (2.5 / 3) - 0.5)
        st = N.repeat_stats(d)
        self.assertAlmostEqual(st["within_action_variance"], pooled)
        one = handmade([(2, 1, [("a", [0.0], True), ("b", [1.0], True)])])
        self.assertIsNone(N.repeat_stats(one)["within_action_variance"])
        self.assertTrue(T.gain_items(one, None)["uncertainty"].startswith("none"))

    def test_rule_ties_and_lambda_zero(self):
        d = handmade([(2, 1, [("m", [0.5], True), ("z", [1.0], True), ("b", [0.0], True)])])
        g = np.array([0.0, 0.1, 0.1])
        o = "incumbent_override"
        self.assertEqual(int(E.apply_rule(d, g, 0.0, o)[0]), 0)
        self.assertEqual(int(E.apply_rule(d, g, 10.0, o)[0]), 0)     # 1 + 0 vs 0 + 10 * 0.1: tie, incumbent kept
        self.assertEqual(int(E.apply_rule(d, g, 100.0, o)[0]), 2)    # tie between alternatives: smallest key "b"
        d["cand_support"][2] = False
        self.assertEqual(int(E.apply_rule(d, np.array([0.0, 0.1, np.nan]), 100.0, o)[0]), 1)
        self.assertEqual(int(E.apply_rule(d, g, 100.0)[0]), 0)      # zscore without logged scores: the incumbent
        s = E.summarize(d, d["root_incumbent"].astype(np.int64), 0.002)
        self.assertEqual(s["primary"]["mean"], 0.0)

    def test_zscore_rule_uses_logged_scores(self):
        d = handmade([(2, 1, [("a", [0.5], True, 1.0), ("b", [0.0], True, 3.0), ("c", [1.0], True)]),
                      (2, 2, [("a", [0.5], True), ("b", [1.0], True, 5.0)])])
        g = np.array([0.0, 0.0, 9.0, 0.0, 9.0])
        choice = E.apply_rule(d, g, 0.0)
        self.assertEqual(int(choice[0]), 1)       # lambda 0: the carrier's top-scored candidate
        self.assertEqual(int(choice[1]), 3)       # the incumbent has no logged score: kept
        self.assertEqual(int(E.apply_rule(d, g, 100.0)[0]), 1)   # "c" has no logged score: never chosen
        g2 = np.array([0.0, -0.1, 0.0, 0.0, 0.0])
        self.assertEqual(int(E.apply_rule(d, g2, 100.0)[0]), 0)  # z(a) = -1, z(b) = +1: a gain of -0.1 x 100 flips it
        self.assertEqual(E.scored_roots(d), 1)

    def test_unsupported_incumbent_keeps_the_root(self):
        d = handmade([(2, 1, [("a", [0.5], False), ("b", [1.0], True)])])
        g = E.mask_gains(d, np.array([0.3, 0.9]))
        self.assertEqual(g[0], 0.0)
        self.assertTrue(np.isnan(g[1]))
        self.assertEqual(int(E.apply_rule(d, np.array([0.0, 0.9]), 1e6, "incumbent_override")[0]), 0)
        self.assertEqual(len(T.gain_items(d, None)["y"]), 0)
        self.assertEqual(N.coverage(d)["incumbent_unsupported"], 1)

    def test_headroom_uses_independent_repeats(self):
        d = handmade([(2, 1, [("a", [0.5, 0.5], True), ("b", [1.0, 0.0], True)]),
                      (2, 2, [("a", [0.5, 0.5], True), ("b", [1.0, 0.0], True)])])
        h = E.headroom(d, 0.002)
        self.assertAlmostEqual(h["headroom"]["mean"], -0.5)        # selected on the lucky repeat, evaluated on the other
        one = handmade([(2, 1, [("a", [0.5], True), ("b", [1.0], True)])])
        self.assertTrue(E.headroom(one, 0.002)["status"].startswith("unavailable"))


class ScorerAndSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.data = make_dataset(cls.tmp)
        cls.runs = {}
        for model in ("linear", "mlp"):
            out = cls.tmp / f"run_{model}"
            T.main(["--data", str(cls.data), "--model", model, "--hidden", "8", "--epochs", "3", "--exploratory",
                    "--seed", "1", "--out", str(out)])
            cls.runs[model] = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_zscore_refused_without_scores(self):
        data = make_dataset(self.tmp / "noscores", scores=False)
        run = self.tmp / "noscore_run"
        T.main(["--data", str(data), "--epochs", "1", "--exploratory", "--out", str(run)])
        args = ["select", "--data", str(data), "--run", f"linear={run}", "--primary", "linear", "--negligible", "0.002",
                "--out", str(self.tmp / "noscore_sel.json")]
        with self.assertRaises(SystemExit):
            E.main(args)
        E.main(args + ["--norm", "incumbent_override"])
        sel = json.loads((self.tmp / "noscore_sel.json").read_text())
        self.assertTrue(sel["rule"].startswith("EXPLORATORY FALLBACK"))

    def test_exploratory_needs_flag(self):
        with self.assertRaises(SystemExit):
            T.main(["--data", str(self.data), "--epochs", "1", "--out", str(self.tmp / "x")])

    def test_deterministic_training(self):
        out = self.tmp / "again"
        T.main(["--data", str(self.data), "--model", "linear", "--epochs", "3", "--exploratory", "--seed", "1", "--out", str(out)])
        a, b = np.load(out / "model.npz"), np.load(self.runs["linear"] / "model.npz")
        np.testing.assert_array_equal(a["W0"], b["W0"])

    def test_gains_and_support(self):
        d = D.load(self.data, "val")
        g = T.load_scorer(self.runs["mlp"], d["_index"]).gains(d)
        np.testing.assert_array_equal(g[d["root_incumbent"]], 0.0)
        self.assertTrue(np.isnan(g[~d["cand_support"]]).all())

    def test_model_kind_separation(self):
        from training import export_mixed, export_weights  # noqa: PLC0415
        from training.common import warm_start  # noqa: PLC0415
        from fusion.models import PolicyNet  # noqa: PLC0415
        run = self.runs["linear"]
        with self.assertRaises(SystemExit):
            export_weights.run_config(run, layout(), "policy")
        with self.assertRaises(SystemExit):
            export_weights.load_value(run, layout())
        with self.assertRaises(SystemExit):
            export_mixed.export_runs(self.tmp / "set", policy_run=run, value_tier="m")
        with self.assertRaisesRegex(ValueError, "unknown feature family 'own-action-1'"):
            warm_start(PolicyNet(layout(), np.random.default_rng(0), {"mover": [8], "squad": [8, 8], "act": [8], "target": [4]}, "p3"),
                       run, layout().schema)
        fake = self.tmp / "fake_policy"
        fake.mkdir(exist_ok=True)
        cfg = json.loads((run / "config.json").read_text())
        (fake / "config.json").write_text(json.dumps(dict(cfg, kind="policy")))
        shutil.copy(run / "model.npz", fake / "model.npz")
        with self.assertRaises(SystemExit):
            T.load_scorer(fake)

    def tiny_value_run(self) -> Path:
        """An untrained v3 V run: only the oracle's plumbing is tested, not any value."""
        from fusion import families  # noqa: PLC0415
        from fusion.models import ValueNet  # noqa: PLC0415
        out = self.tmp / "tiny_v"
        out.mkdir(exist_ok=True)
        net = ValueNet(layout(), np.random.default_rng(0), [8], "v3")
        np.savez(out / "model.npz", **net.state(), beta=np.array(1.0))
        (out / "config.json").write_text(json.dumps({"kind": "value", "hidden": [8],
                                                     "features": families.record(layout().schema, "v3")}))
        return out

    def test_frozen_selection(self):
        sel = self.tmp / "SELECTION.json"
        args = ["select", "--data", str(self.data), "--run", f"linear={self.runs['linear']}",
                "--run", f"mlp={self.runs['mlp']}", "--primary", "mlp", "--negligible", "0.002", "--out", str(sel)]
        E.main(args)
        frozen = json.loads(sel.read_text())
        self.assertEqual(frozen["primary"], "mlp - incumbent")
        self.assertEqual(frozen["norm"], "zscore")
        val = D.load(self.data, "val")
        for name, m in frozen["models"].items():          # the frozen rule reproduces its validation number
            g = T.load_scorer(Path(m["run"]), val["_index"]).gains(val)
            mean = E.summarize(val, E.apply_rule(val, g, m["lambda"], m["norm"]), 0.002)["primary"]["mean"]
            self.assertEqual(mean, next(r["mean"] for r in m["val_grid"] if r["lambda"] == m["lambda"]))
        rep_out = self.tmp / "test_report"
        E.main(["test", "--data", str(self.data), "--selection", str(sel), "--out", str(rep_out),
                "--oracle-value", str(self.tiny_value_run())])
        rep = json.loads(rep_out.with_suffix(".json").read_text())
        self.assertIn("mlp - incumbent", rep["comparisons"])
        self.assertTrue(rep["oracle"]["status"].startswith("DIAGNOSTIC ORACLE"))
        self.assertEqual(rep["rules"]["incumbent"]["primary"]["mean"], 0.0)
        self.assertFalse(rep["authoritative"])
        tampered = dict(frozen)
        tampered["models"] = copy.deepcopy(frozen["models"])
        tampered["models"]["mlp"]["lambda"] = 1e9
        bad = self.tmp / "tampered.json"
        bad.write_text(json.dumps(tampered))
        with self.assertRaises(SystemExit):
            E.main(["test", "--data", str(self.data), "--selection", str(bad), "--out", str(rep_out)])
        backup = (self.runs["linear"] / "model.npz").read_bytes()
        try:
            st = dict(np.load(self.runs["linear"] / "model.npz"))
            st["W0"] = st["W0"] + 1
            np.savez(self.runs["linear"] / "model.npz", **st)
            with self.assertRaises(SystemExit):
                E.main(["test", "--data", str(self.data), "--selection", str(sel), "--out", str(rep_out)])
        finally:
            (self.runs["linear"] / "model.npz").write_bytes(backup)


if __name__ == "__main__":
    unittest.main()
