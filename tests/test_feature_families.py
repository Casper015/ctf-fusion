"""Feature families p2/p3/v2/v3 (fusion/families.py, inference/combat.hpp, docs/FEATURE_FAMILIES.md).

  * the legacy families are unchanged: frozen digests, and the layout-2 columns of the new extractor are
    bit-identical to the pre-family extractor on recorded games;
  * what each family can see: the predicted team's own hit points reach legacy P unit features; a target's hit
    points reach the p3 combat block (and not the legacy mover/squad inputs); the v2 aliases the v3 extension
    separates;
  * masks, immunity, respawn, carriers, illegal moves and missing candidates;
  * incompatible feature/model combinations fail clearly (runs, shards, registries, C++ types and widths);
  * extraction and runtime encoding agree, through quantized p3/v3 models and the C++ parity driver.
"""
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import families, native, quant
from fusion.models import PolicyNet, QuantizedTier, ValueNet, policy_batch, policy_widths, value_inputs
from fusion.schema import Layout, layout
from training import export_weights

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
# sha256 (first 16 hex) of the layout-2 columns of every record, made with the extractor before families (7f30c80).
LEGACY_RECORDS = {"n04": "71f35595368675d7", "n15": "0cb8a2ad19038bb7"}
LEGACY_WIDTH = 644


def fixture_stream(name: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{name}.txt.gz").read_bytes()).decode()


class Stream:
    """A legal seat stream on an open map, one TURN block per call of turn()."""

    def __init__(self, teams=3, size=23, turns=30, bases=None, spots=((11, 11), (5, 15)), walls=()):
        self.teams, self.size = teams, size
        self.bases = bases or [(3, 3), (19, 3), (11, 19)][:teams]
        rows = [["."] * size for _ in range(size)]
        for x, y in walls:
            rows[y][x] = "#"
        self.lines = [f"INIT {teams} 0 {size} {turns} 100 34 10 15 8", "MAP", *["".join(r) for r in rows],
                      f"BASES {teams}", *[f"{x} {y}" for x, y in self.bases], f"SPOTS {len(spots)}", *[f"{x} {y}" for x, y in spots]]
        self.t = 0

    def default_units(self):
        """Every unit two rows in front of its base, full hit points."""
        out = {}
        for uid in range(3 * self.teams):
            bx, by = self.bases[uid // 3]
            out[uid] = dict(x=bx + uid % 3 - 1, y=by + 2, hp=100, flag=-1, respawn=-1)
        return out

    def turn(self, units=None, flags=None, score=None, attacks=()):
        self.t += 1
        units = units or self.default_units()
        flags = flags if flags is not None else [dict(status="home", x=11, y=11, holder=-1, ret=-1),
                                                 dict(status="home", x=5, y=15, holder=-1, ret=-1)]
        score = score or [0] * self.teams
        self.lines += [f"TURN {self.t}", "SCORE " + " ".join(map(str, score)), f"UNITS {len(units)}"]
        for uid in sorted(units):
            u = units[uid]
            self.lines.append(f"U {uid} {uid // 3} {u['x']} {u['y']} {u['hp']} {u['flag']} {u['respawn']}")
        self.lines.append(f"FLAGS {len(flags)}")
        for i, f in enumerate(flags):
            self.lines.append(f"F {i} {f['status']} {f['x']} {f['y']} {f['holder']} {f['ret']}")
        self.lines.append(f"EVENTS {len(attacks)}")
        self.lines += [f"E attack {a} {b}" for a, b in attacks]
        self.lines.append("END")
        return self

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def rows_of(stream: Stream, team: int) -> np.ndarray:
    rec = native.extract(stream.text())
    return rec[rec[:, 1] == team]


def single(team: int, *states) -> np.ndarray:
    """Team `team`'s turn-1 record of each one-turn stream (states: (units overrides, flags)); same turn, no history."""
    out = []
    for units, flags in states:
        s = Stream()
        out.append(rows_of(s.turn(s.default_units() | units, flags), team)[0])
    return np.stack(out)


def col(L: Layout, group: str, name: str) -> int:
    return L.names(group).index(name)


class TestLegacyFamilies(unittest.TestCase):
    def test_legacy_digests_are_frozen(self):
        self.assertEqual({k: families.available(native.schema())[k] for k in ("p2", "v2")}, families.LEGACY_DIGESTS)

    def test_layout2_columns_unchanged_on_recorded_games(self):
        L = layout()
        self.assertEqual(L.layout_version, 3)
        self.assertEqual(L.blocks["pc"][0], LEGACY_WIDTH)  # the new blocks start after every layout-2 block
        for name, want in LEGACY_RECORDS.items():
            rec = native.extract(fixture_stream(name))
            got = hashlib.sha256(np.ascontiguousarray(rec[:, :LEGACY_WIDTH]).tobytes()).hexdigest()[:16]
            self.assertEqual(got, want, f"{name}: the legacy encoder output changed")

    def test_legacy_runs_and_reports_map_to_legacy_families(self):
        self.assertEqual(families.run_features({"spec_version": 2}, "policy")["family"], "p2")
        self.assertEqual(families.run_features({"spec_version": 2}, "value")["digest"], families.LEGACY_DIGESTS["v2"])
        with self.assertRaises(SystemExit):
            families.run_features({"spec_version": 1}, "policy")
        identity = export_weights.artifact_identity  # legacy pairs keep the identity of before families
        self.assertEqual(export_weights.LEGACY_SCHEMA_DIGEST, json.loads((REPO / "inference" / "models_m" / "MANIFEST.json")
                                                                         .read_text())["general"]["identity"]["schema_digest"])
        self.assertTrue(callable(identity))


class TestWhatFamiliesSee(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = layout()

    def test_predicted_team_hp_reaches_legacy_unit_features(self):
        """P predicts team 1 from team 1's perspective: its own unit's hit points are a p2 unit feature."""
        unit = dict(x=18, y=5, flag=-1, respawn=-1)
        r = single(1, ({3: unit | {"hp": 100}}, None), ({3: unit | {"hp": 34}}, None))
        u = self.L.get(r, "u")
        hp = col(self.L, "unit", "hp")
        self.assertEqual((u[0, 0, hp], u[1, 0, hp]), (100, 34))
        X = policy_batch(r, self.L, normalize=False)
        self.assertFalse(np.array_equal(X.mover_in[0, 0], X.mover_in[1, 0]))

    def test_target_hp_reaches_the_combat_block_only(self):
        """An enemy two cells away: at 100 hp it needs three hits, at 34 one. The legacy mover and squad inputs of
        the shooter's team are identical; only the target head's candidate features and the p3 block change."""
        shooter = {0: dict(x=10, y=10, hp=100, flag=-1, respawn=-1)}
        far = dict(x=10, y=12, flag=-1, respawn=-1)
        r = single(0, (shooter | {3: far | {"hp": 100}}, None), (shooter | {3: far | {"hp": 34}}, None))
        p2 = policy_batch(r, self.L, normalize=False, family="p2")
        p3 = policy_batch(r, self.L, normalize=False, family="p3")
        np.testing.assert_array_equal(p2.mover_in[0], p2.mover_in[1])
        np.testing.assert_array_equal(p2.squad_in[0], p2.squad_in[1])
        self.assertFalse(np.array_equal(p2.cand_f[0], p2.cand_f[1]))
        self.assertFalse(np.array_equal(p3.mover_in[0], p3.mover_in[1]))
        pc = self.L.get(r, "pc")
        k1 = col(self.L, "combat", "kill1_stay")
        self.assertEqual(pc[0, 0, 0, k1], 0)  # staying at (10, 10): range 2, three hits needed
        self.assertEqual(pc[1, 0, 0, k1], 1)
        self.assertEqual(pc[1, 0, 2, k1], 1)  # moving down to (10, 11): still in range

    def test_carrier_focus_keeps_attributes_together(self):
        """Two enemy carriers in range: the focus is the one with fewer hits; its delivery distance comes with it."""
        flags = [dict(status="carried", x=-1, y=-1, holder=3, ret=-1), dict(status="carried", x=-1, y=-1, holder=6, ret=-1)]
        units = {0: dict(x=10, y=10, hp=100, flag=-1, respawn=-1), 3: dict(x=10, y=12, hp=100, flag=0, respawn=-1),
                 6: dict(x=12, y=10, hp=34, flag=1, respawn=-1)}
        r = single(0, (units, flags))
        pc = self.L.get(r, "pc")[0, 0, 0]
        names = self.L.names("combat")
        f = {n: int(pc[i]) for i, n in enumerate(names)}
        self.assertEqual((f["focus_present"], f["focus_hits"], f["kill1_carrier_stay"]), (1, 1, 1))
        # unit 6 (team 2) carries; its base is (11, 19): BFS distance from (12, 10) to the nearest base cell.
        self.assertEqual(f["focus_home"], 8)

    def test_v2_alias_separated_by_v3(self):
        """Team 0's view of an enemy carrier at 100 hp and at 34 hp: identical v2 inputs, different v3."""
        flags = [dict(status="carried", x=-1, y=-1, holder=3, ret=-1), dict(status="home", x=5, y=15, holder=-1, ret=-1)]
        carrier = dict(x=11, y=10, flag=0, respawn=-1)
        r = single(0, ({3: carrier | {"hp": 100}}, flags), ({3: carrier | {"hp": 34}}, flags))
        v2 = value_inputs(r, self.L, normalize=False, family="v2")
        v3 = value_inputs(r, self.L, normalize=False, family="v3")
        np.testing.assert_array_equal(v2[0], v2[1])
        names = self.L.value_names("v3")
        self.assertEqual((v3[0, names.index("ec0_hits")], v3[1, names.index("ec0_hits")]), (3, 1))

    def test_v2_alias_flag_return_timing(self):
        """A dropped flag returning in 14 turns or in 1: v2 sees one dropped flag either way."""
        home = dict(status="home", x=5, y=15, holder=-1, ret=-1)
        r = single(0, *[({}, [dict(status="dropped", x=8, y=8, holder=-1, ret=1 + wait), home]) for wait in (14, 1)])
        v2 = value_inputs(r, self.L, normalize=False, family="v2")
        np.testing.assert_array_equal(v2[0], v2[1])
        names = self.L.value_names("v3")
        v3 = value_inputs(r, self.L, normalize=False, family="v3")
        self.assertEqual((v3[0, names.index("dropped_return_min")], v3[1, names.index("dropped_return_min")]), (14, 1))


class TestMasksAndRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = layout()
        cls.rec = native.extract(fixture_stream("n15"))

    def test_illegal_moves_and_absent_units_have_zero_blocks(self):
        pc = self.L.get(self.rec, "pc")
        legal = self.L.get(self.rec, "legal").astype(bool)
        self.assertTrue((pc[~legal] == 0).all())
        active = self.L.get(self.rec, "u")[..., col(self.L, "unit", "active")] > 0
        self.assertTrue((pc[~active] == 0).all())

    def test_units_that_cannot_shoot_have_no_own_fire(self):
        u = self.L.get(self.rec, "u")
        pc = self.L.get(self.rec, "pc")
        cannot = (u[..., col(self.L, "unit", "carrying")] > 0) | (u[..., col(self.L, "unit", "immune")] > 0)
        own_fire = [col(self.L, "combat", n) for n in ("kill1_stay", "kill1_carrier_stay", "volley_stay",
                                                      "volley_carrier_stay", "focus_present", "focus_hits", "focus_home")]
        self.assertTrue(cannot.any())
        self.assertTrue((pc[cannot][..., own_fire] == 0).all())
        immune = u[..., col(self.L, "unit", "immune")] > 0
        exposure = [col(self.L, "combat", n) for n in ("exposure_reach", "lethal_stay", "lethal_reach")]
        self.assertTrue((pc[immune][..., exposure] == 0).all())

    def test_counts_are_consistent(self):
        pc = self.L.get(self.rec, "pc").astype(int)
        c = {n: pc[..., i] for i, n in enumerate(self.L.names("combat"))}
        self.assertTrue((c["kill1_stay"] <= c["volley_stay"]).all())          # a one-hit kill is a volley kill
        self.assertTrue((c["kill1_carrier_stay"] <= c["kill1_stay"]).all())
        self.assertTrue((c["volley_carrier_stay"] <= c["volley_stay"]).all())
        self.assertTrue(((c["focus_present"] == 1) == (c["focus_hits"] > 0)).all())
        self.assertTrue((c["lethal_stay"] <= c["lethal_reach"]).all())        # staying shooters are among those that reach
        self.assertTrue((c["kill1_stay"] > 0).any() and (c["volley_carrier_stay"] > 0).any())
        m = self.L.get(self.rec, "m").astype(int)
        targets = m[..., col(self.L, "move", "targets")]
        self.assertTrue((c["volley_stay"] <= targets).all())

    def test_respawning_units_change_no_combat_feature(self):
        """A unit respawning this turn is immune: it can neither shoot nor be shot, so the combat block of a unit
        next to its spawn cell is the same as when it is still waiting off the board."""
        near = {0: dict(x=18, y=6, hp=34, flag=-1, respawn=-1)}
        r = single(0, (near | {3: dict(x=-1, y=-1, hp=0, flag=-1, respawn=1)}, None),   # respawns now (turn 1)
                      (near | {3: dict(x=-1, y=-1, hp=0, flag=-1, respawn=5)}, None))   # still waiting
        origin = self.L.get(r, "origin")
        self.assertEqual(origin[1, 0], 6 * 23 + 18)
        np.testing.assert_array_equal(self.L.get(r, "pc")[0], self.L.get(r, "pc")[1])
        self.assertNotIn(3, self.L.get(r, "cand")[0, 0].tolist())  # missing candidates stay -1, never an immune unit
        v3 = value_inputs(r, self.L, normalize=False, family="v3")
        names = self.L.value_names("v3")
        self.assertEqual(v3[0, names.index("enemy_respawning")] + 1, v3[1, names.index("enemy_respawning")])


class TestIncompatible(unittest.TestCase):
    def test_family_digest_mismatch_is_refused(self):
        s = native.schema()
        good = families.record(s, "p3")
        families.check(s, good, "run")
        with self.assertRaises(SystemExit):
            families.check(s, good | {"digest": "0" * 16}, "run")
        with self.assertRaises(SystemExit):
            families.run_features({"features": good}, "value")  # a policy family where a value family is needed

    def test_old_shards_serve_legacy_families_only(self):
        old = json.loads(json.dumps(native.schema()))
        old["layout"] = [b for b in old["layout"] if b[0] not in ("pc", "vx")]
        for k in ("layout_version", "combat_version", "combat", "value_ext"):
            old.pop(k)
        families.check_shards(old, ("p2", "v2"), "old shards")
        for fam in ("p3", "v3"):
            with self.assertRaises(SystemExit):
                families.check_shards(old, (fam,), "old shards")
        moved = json.loads(json.dumps(native.schema()))
        moved["layout"] = [[b, o + (1 if b == "v" else 0), sh] for b, o, sh in moved["layout"]]
        with self.assertRaises(SystemExit):
            families.check_shards(moved, ("v2",), "shifted shards")

    def test_registry_holds_one_family_pair(self):
        with tempfile.TemporaryDirectory() as d:
            man = {"n02": {"header": "a.hpp"}, "n03": {"header": "b.hpp", "features": {"policy": {"family": "p3"},
                                                                                           "value": {"family": "v2"}}}}
            with self.assertRaises(SystemExit):
                export_weights.registry(Path(d), man)

    def test_quantized_tier_checks_widths(self):
        L = layout()
        rng = np.random.default_rng(0)
        policy, value = PolicyNet(L, rng, family="p3"), ValueNet(L, rng, family="v3")
        self.assertEqual(export_weights.model_width(policy, value), 448)  # the p3 squad input (447) exceeds 256
        nets = export_weights.quantize_model(policy, value, 1.0, L, export_weights.model_width(policy, value))
        flat = export_weights.flat_arrays(nets)
        tier = QuantizedTier(flat, layout=L)
        self.assertEqual((tier.policy_family, tier.value_family), ("p3", "v3"))
        with self.assertRaises(ValueError):
            QuantizedTier(flat, "p2", None, layout=L)

    def test_cpp_refuses_crossed_families(self):
        """p2 features cannot reach a p3 model, and a p2-shaped net cannot be declared p3 (static_assert)."""
        L = layout()
        rng = np.random.default_rng(1)
        p2_nets = export_weights.quantize_model(PolicyNet(L, rng), ValueNet(L, rng), 1.0, L)
        text = export_weights.header("crossed", p2_nets, {}, {}, policy_family="p3", value_family="v2")
        self.assertIn('static_assert(40 == fusion::kMoverInP3', text)
        with tempfile.TemporaryDirectory() as d:
            hdr = Path(d) / "weights_crossed.hpp"
            hdr.write_text(text, encoding="utf-8")
            src = Path(d) / "use.cpp"
            src.write_text(f'#include "{hdr.as_posix()}"\nint main() {{ return 0; }}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                native.build(src, "crossed_family", deps=[hdr], includes=[REPO / "inference"])
            wrong = Path(d) / "wrong.cpp"
            wrong.write_text('#include "policy_p3.hpp"\nint main() { fusion::PolicyModelP3 m{}; fusion::PolicyFeatures f{};\n'
                             '  fusion::PolicyOutput o; fusion::jointPolicy(m, f, o); return 0; }\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                native.build(wrong, "crossed_types", includes=[REPO / "inference"])


class TestRuntimeAgreesWithExtraction(unittest.TestCase):
    """A quantized p3/v3 model: the Python emulation on extractor records equals the C++ runtime encoders plus
    inference (infer_check), bit for bit; the runtime path allocates nothing after turn 1 and the v3 accumulator
    matches the full pass (runtime_check)."""

    def test_p3_v3_parity(self):
        L = layout()
        rng = np.random.default_rng(3)
        hidden = {"mover": [16], "squad": [32, 16], "act": [16], "target": [8]}
        policy, value = PolicyNet(L, rng, hidden, "p3"), ValueNet(L, rng, (64, 16), "v3")
        nets = export_weights.quantize_model(policy, value, 1.3, L, export_weights.model_width(policy, value))
        self.assertEqual(nets["mover"]["in"], policy_widths(L, "p3")["mover"])
        flat = export_weights.flat_arrays(nets)
        with tempfile.TemporaryDirectory() as d:
            hdr = Path(d) / "weights_famtest.hpp"
            hdr.write_text(export_weights.header("famtest", nets, {}, {}, policy_family="p3", value_family="v3"), encoding="utf-8")
            self.assertEqual(native.model_families(hdr), ("p3", "v3"))
            stream = fixture_stream("n04")
            rec = native.extract(stream)
            cpp = native.run_infer_check(native.infer_check(hdr, "fusion::models::famtest"), stream, len(rec))
            py = QuantizedTier(flat, "p3", "v3", layout=L).raw(rec, L)
            np.testing.assert_array_equal(cpp, py)
            r = native.run_tool(native.model_tool("runtime_check", hdr, "fusion::models::famtest"), stream.encode())
            self.assertEqual(r.returncode, 0, r.stdout.decode() + r.stderr.decode())

    def test_quantized_bound_holds_for_wide_first_layers(self):
        """The 4XL value shape's quantization proves int32 safety (shape only; small rows for speed)."""
        L = layout()
        net = ValueNet(L, np.random.default_rng(4), (8192, 512), "v3").net
        lo, hi = L.value_bounds("v3")
        q = quant.quantize(net, L.value_scale("v3"), lo, hi, out_rows=[0])
        self.assertLessEqual(max(layer["bound"] for layer in q["layers"]), quant.ACC_LIMIT)
        quant.check_widths(q, "value", 8192)
        x = np.random.default_rng(5).integers(lo, hi + 1, size=(16, len(lo)))
        quant.forward_raw(q, x)  # raises if an accumulator leaves the proven range


if __name__ == "__main__":
    unittest.main()


def fake_run(path: Path, kind: str, family: str, hidden, seed: int = 0) -> Path:
    """A training-run directory as train_policy / train_value write it (random weights)."""
    L = layout()
    rng = np.random.default_rng(seed)
    model = PolicyNet(L, rng, hidden, family) if kind == "policy" else ValueNet(L, rng, hidden, family)
    path.mkdir(parents=True, exist_ok=True)
    state = model.state() | ({} if kind == "policy" else {"beta": np.array(1.0)})
    np.savez(path / "model.npz", **state)
    (path / "config.json").write_text(json.dumps({"kind": kind, "hidden": hidden, "features": families.record(native.schema(), family)}),
                                      encoding="utf-8")
    return path


class TestExportRuns(unittest.TestCase):
    """training.export_mixed with runs: a frozen committed P paired with a trained V (written once, shared), and a
    trained p3 policy paired with a committed V."""

    def test_frozen_p_with_value_run(self):
        from training import export_mixed  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d) / "value_run", "value", "v3", [64, 16])
            out = export_mixed.export_runs(Path(d) / "set", policy_tier="m", value_run=run)
            man = json.loads((out / "MANIFEST.json").read_text())
            src = json.loads((REPO / "inference" / "models_m" / "MANIFEST.json").read_text())
            self.assertEqual(sorted(man), sorted(src))
            self.assertEqual(len(list(out.glob("weights_shared_value.hpp"))), 1)
            for model, e in man.items():
                self.assertEqual((e["features"]["policy"]["family"], e["features"]["value"]["family"]), ("p2", "v3"))
                self.assertEqual(e["sources"]["policy"]["weights_digest"], src[model]["identity"]["weights_digest"])
                text = (out / e["header"]).read_text()
                self.assertIn("fusion::models::shared_value::kValue;  // shared", text)
                self.assertNotIn("value_w0", text)  # the value arrays live in the shared header only
                z, ref = np.load(out / f"weights_{model}.npz"), np.load(REPO / "inference" / "models_m" / f"weights_{model}.npz")
                for k in z.files:
                    np.testing.assert_array_equal(z[k], ref[k])  # P copied exactly
            r = native.run_tool(native.registry_check(out))
            self.assertEqual(r.returncode, 0, r.stderr.decode())
            self.assertIn("2 n02", r.stdout.decode())

    def test_p3_policy_run_with_committed_value(self):
        from training import export_mixed  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d) / "policy_run", "policy", "p3", {"mover": [16], "squad": [32, 16], "act": [16], "target": [8]})
            out = export_mixed.export_runs(Path(d) / "set", policy_run=run, value_tier="m")
            man = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(list(man), ["general"])
            self.assertEqual(export_weights.manifest_families(man["general"]), ("p3", "v2"))
            self.assertEqual(native.model_families(out / "weights_general.hpp"), ("p3", "v2"))
            r = native.run_tool(native.registry_check(out))
            self.assertEqual(r.returncode, 0, r.stderr.decode())
            self.assertIn("15 general", r.stdout.decode())
