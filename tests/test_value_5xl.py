"""Actual 16384 x 1024 contracts, chunk regeneration and paired decision statistics."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from fusion import native, quant
from fusion.models import ValueNet
from fusion.paths import REPO
from fusion.schema import layout
from training import evaluate_value, export_weights
from training.branch_collect import candidates
from training.value_cache import cache_identity
from training.header_chunks import FILE_LIMIT, HeaderWriter, verify_chunks


class TestValue5XL(unittest.TestCase):
    def test_complete_candidates_are_bounded_deterministic_and_cover_actions(self):
        from fusion.protocol import parse_reply
        block = ("TURN 5\nSCORE 0 0\nUNITS 6\n"
                 "U 0 0 2 2 100 -1 -1\nU 1 0 2 3 100 -1 -1\nU 2 0 7 7 100 0 -1\n"
                 "U 3 1 4 2 100 -1 -1\nU 4 1 15 15 100 -1 -1\nU 5 1 16 16 100 -1 -1\n"
                 "FLAGS 0\nEVENTS 0\nEND\n")
        game = SimpleNamespace(size=20, walls=set(), flags=[{"status": "home", "pos": [2, 2]}],
                               can_attack=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 2)
        line = "5 S - S - S -"
        result = candidates(game, block, line, 0, 5, 25, "root", "complete")
        self.assertEqual(result, candidates(game, block, line, 0, 5, 25, "root", "complete"))
        self.assertLessEqual(len(result), 25)
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(result[0], line)
        orders = [parse_reply(s, 5) for s in result]
        self.assertTrue(all(order is not None for order in orders))
        self.assertTrue(any(sum(r.move != 0 for r in order) >= 2 for order in orders))
        self.assertTrue(any(r.act == 1 for order in orders for r in order))
        self.assertTrue(any(r.act == 2 for order in orders for r in order))
        self.assertTrue(any(r.act == 3 for order in orders for r in order))

    def test_cache_identity_rejects_map_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta.json").write_text("{}")
            for split in ("train", "val"):
                (root / split).mkdir()
                np.save(root / split / "map.npy", [123])
            with self.assertRaisesRegex(ValueError, "map leakage"):
                cache_identity(root)
            np.save(root / "val" / "map.npy", [456])
            self.assertEqual(cache_identity(root)["maps"], {"train": [123], "val": [456]})

    def test_value_cost_feeds_concurrent_copies(self):
        from tools.host import value_cost
        barrier = threading.Barrier(2)

        class Process:
            returncode = 0

            def communicate(self, data):
                barrier.wait(timeout=3)
                return json.dumps({"stream": data.decode()}).encode(), b""

        with mock.patch.object(value_cost.subprocess, "Popen", side_effect=lambda *a, **kw: Process()):
            self.assertEqual(value_cost.run_copies(Path("unused"), [b"a", b"b"], 1, 2), [{"stream": "a"}, {"stream": "b"}])

    def test_actual_shape_quantization_native_and_chunks(self):
        L = layout()
        model = ValueNet(L, np.random.default_rng(43), [16384, 1024], "v3")
        self.assertEqual(sum(w.size + b.size for w, b in zip(model.net.W, model.net.b)), 18_567_171)
        lo, hi = L.value_bounds("v3")
        q = quant.quantize(model.net, L.value_scale("v3"), lo, hi, out_rows=[0])
        self.assertEqual(export_weights.macs(q), 18_547_712)
        quant.check_widths(q, "value", 16384)
        with self.assertRaises(ValueError):
            quant.check_widths(q, "value", 8192)
        for i, layer in enumerate(q["layers"]):
            xmax = np.maximum(np.abs(lo), np.abs(hi)).astype(np.int64) if i == 0 else np.full(layer["w"].shape[1], 128)
            bound = (np.abs(layer["w"].astype(np.int64)) @ xmax + np.abs(layer["b"].astype(np.int64))).max()
            self.assertEqual(int(bound), layer["bound"])
            self.assertLessEqual(bound, quant.ACC_LIMIT)
        rng = np.random.default_rng(47)
        random = rng.integers(lo, hi + 1, size=(3, len(lo)), dtype=np.int16)
        x = np.concatenate([lo[None], hi[None], random, random[-1:]]).astype(np.int16)
        raw = quant.forward_raw(q, x)
        real = quant.to_real(q, raw)[:, 0]
        floating = model.logits((x / L.value_scale("v3")).astype(np.float32))[:, 0]
        self.assertLess(float(np.max(np.abs(real - floating))), 0.1)
        with tempfile.TemporaryDirectory(dir=REPO / "build") as temp:
            header = Path(temp) / "weights_capacity5xl.hpp"
            writer = HeaderWriter(header)
            chunks = writer.write(export_weights.shared_value_header("capacity5xl", q, "v3", {}, array_writer=writer.array))
            self.assertGreater(len(chunks), 30)
            self.assertTrue(all(c["bytes"] < FILE_LIMIT for c in chunks))
            verify_chunks(header.parent, chunks)
            before = header.read_bytes(), chunks
            writer2 = HeaderWriter(header)
            chunks2 = writer2.write(export_weights.shared_value_header("capacity5xl", q, "v3", {}, array_writer=writer2.array))
            self.assertEqual(before, (header.read_bytes(), chunks2))
            deps = native.model_dependencies(header)
            self.assertTrue(all((header.parent / c["path"]).resolve() in deps for c in chunks))
            binary = native.build(REPO / "tools" / "cpp" / "value_capacity_check.cpp", "capacity5xl_contract", deps=deps,
                                  defines=[f'FUSION_MODEL_HEADER="{header.as_posix()}"',
                                           "FUSION_MODEL_NS=fusion::models::capacity5xl", "FUSION_MAX_WIDTH=16384"],
                                  includes=[REPO / "inference"])
            result = native.run_tool(binary, x.astype("<i2").tobytes())
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            native_raw = np.frombuffer(result.stdout, "<i4").reshape(-1, 3)
            np.testing.assert_array_equal(native_raw, np.repeat(raw, 3, axis=1))
            part = header.parent / chunks[0]["path"]
            part.write_bytes(part.read_bytes() + b"// changed\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_chunks(header.parent, chunks)

    def test_chunk_regeneration_removes_only_own_stale_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights_test.hpp"
            writer = HeaderWriter(path, 4)
            writer.write(writer.array("value_w0", "int16_t", np.arange(12)))
            directory = path.parent / "weights_test_parts"
            foreign = directory / "keep.txt"
            foreign.write_text("unrelated")
            writer = HeaderWriter(path, 4)
            chunks = writer.write(writer.array("value_w0", "int16_t", np.arange(6)))
            self.assertEqual(len(chunks), 2)
            self.assertFalse((directory / "value_w0_0002.hpp").exists())
            self.assertTrue(foreign.exists())

    def test_incumbent_comparison_is_paired_by_root_with_ties(self):
        # Two roots/map: incumbent regret .2, model regret 0, identical paired advantage per root.
        cand_y = np.tile([0.5, 0.7], 6).astype(np.float32)
        roots = np.repeat(np.arange(6), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rank.npz"
            np.savez(path, cand_rec=np.zeros((12, 923), np.int16), cand_root=roots, cand_y=cand_y,
                     cand_on_policy=np.tile([True, False], 6), root_map=np.repeat([101, 102, 103], 2),
                     pair_a=np.arange(0, 12, 2), pair_b=np.arange(1, 12, 2), pair_y=np.zeros(6, np.int8),
                     pair_w=np.ones(6), cand_n=np.full(12, 3))
            model = mock.Mock(family="v3")
            with mock.patch.object(evaluate_value, "probs", return_value=cand_y):
                result = evaluate_value.rank_metrics(model, 1, {"path": path}, layout())
            self.assertEqual(result["regret_vs_incumbent"]["clusters"], 3)
            self.assertAlmostEqual(result["regret_vs_incumbent"]["mean"], -0.2)
            self.assertEqual(result["regret_vs_incumbent"]["verdict"], "lower")
            self.assertEqual(result["pair_majority_baseline"], 1)
            with mock.patch.object(evaluate_value, "probs", return_value=np.full(12, 0.5)):
                result = evaluate_value.rank_metrics(model, 1, {"path": path}, layout())
            self.assertEqual(result["pair_accuracy"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
