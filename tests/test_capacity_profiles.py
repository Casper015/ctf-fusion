"""Architecture/large-export correctness, never development-host latency evidence."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fusion import capacity, native, quant, synthetic
from fusion.schema import layout
from training import export_weights, train_policy, train_value
from training.common import resolve_hidden


def args(tier=None, max_width=0, **overrides):
    return argparse.Namespace(capacity=tier, max_width=max_width, hidden=None,
                              **({"hidden_"+h: None for h in capacity.POLICY_HEADS} | overrides))


class TestCapacityProfiles(unittest.TestCase):
    def test_default_and_preset_override_are_separate(self):
        a = args()
        hidden, name = resolve_hidden(a, "policy")
        self.assertEqual(hidden, {h: capacity.TIERS["S"][h] for h in capacity.POLICY_HEADS})
        self.assertEqual((name, a.max_width), ("S", 256))
        a = args("XXXL", hidden_target="8")
        hidden, name = resolve_hidden(a, "policy")
        self.assertEqual(hidden["squad"], [2048, 512])
        self.assertEqual(hidden["target"], [8])
        self.assertEqual((name, a.max_width), ("XXXL", 2048))
        self.assertEqual(capacity.preset("XXXL")["target"], [64])

    def test_p_and_v_can_choose_capacity_independently(self):
        p, p_name = resolve_hidden(args("XXXXL"), "policy")
        v_args = args("S")
        v, v_name = resolve_hidden(v_args, "value")
        self.assertEqual(p["squad"], [4096, 1024])
        self.assertEqual(v, [128, 16])
        self.assertEqual((p_name, v_name), ("XXXXL", "S"))

    def test_explicit_undersized_or_legacy_custom_limit_fails_before_data(self):
        with self.assertRaises(SystemExit):
            resolve_hidden(args("XXXXL", max_width=2048), "policy")
        with self.assertRaises(SystemExit):
            resolve_hidden(args(hidden_squad="512,32"), "policy")
        with self.assertRaises(SystemExit):
            resolve_hidden(args("S", max_width=-1), "value")
        for trainer, command in (
            (train_policy, ["--capacity", "XXXL", "--max-width", "256"]),
            (train_value, ["--capacity", "XXXXL", "--max-width", "256"]),
        ):
            with self.subTest(trainer=trainer.__name__), mock.patch.object(trainer, "load") as load:
                with self.assertRaises(SystemExit):
                    trainer.main(["--model", "general", "--out", "unused", *command])
                load.assert_not_called()

    def test_shape_only_verification_capacity(self):
        from tools import capacity_bench as cb
        with mock.patch.object(cb, "tier_static", side_effect=AssertionError("allocated synthetic export")):
            self.assertEqual(cb.required_width(capacity.preset("XXXL"), layout()), 4096)
            self.assertEqual(cb.required_width(capacity.preset("XXXXL"), layout()), 8192)
        self.assertIs(cb.TIERS, capacity.TIERS)
        with self.assertRaises(SystemExit):
            cb.parse_tier("bad squad=0,16")
        with self.assertRaises(SystemExit):
            cb.verify({"S": capacity.preset("S")}, [2], turns=2)
        with self.assertRaises(SystemExit):
            cb.verify({"S": capacity.preset("S")}, [2], value_calls=0)

    def test_native_width_preamble_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"header.hpp"
            for text, expected in (("// FUSION_MAX_WIDTH_REQUIRED=8192\n", 8192),
                                   ("// Widest layer 520: build wider.\n", 576),
                                   ("// old narrow header\n", 256)):
                p.write_text(text)
                self.assertEqual(native.model_buffer_width(p), expected)


class TestLargeNativeArtifact(unittest.TestCase):
    def test_export_above_xxl_compiles_and_matches_integer_emulation(self):
        spec = layout()
        inputs = export_weights.input_specs(spec)
        # A low-cost but genuinely >XXL first layer, including a long inner dot
        # in its tail. Deterministic bounded integers; not trained model weights.
        width = 3072

        def layer(w, hidden=False):
            return {"w": w, "b": np.zeros(w.shape[0], np.int32), "shift": 0, "hidden": hidden}

        nets = {}
        for head, out in (("mover", 1), ("squad", 125), ("act", 4), ("target", 1)):
            w = np.zeros((out, len(inputs[head][0])), np.int16)
            nets[head] = {"layers": [layer(w)], "out_exp": 0}
        p_in = len(inputs["squad"][0])
        p_first = np.zeros((width, p_in), np.int16)
        p_first[np.arange(width), np.arange(width) % p_in] = 1
        p_tail = np.repeat(np.where(np.arange(125)[:, None] % 2, -1, 1), width, axis=1).astype(np.int16)
        nets["squad"] = {"layers": [layer(p_first, True), layer(p_tail)], "out_exp": 0}
        n_in = len(inputs["value"][0])
        w = np.zeros((width, n_in), np.int16)
        w[np.arange(width), np.arange(width) % n_in] = 1
        tail = np.ones((1, width), np.int16)
        nets["value"] = {"layers": [layer(w, True), layer(tail)], "out_exp": 0}
        for head, net in nets.items():
            quant.check_widths(net, head, limit=width)
        with self.assertRaises(ValueError):
            quant.check_widths(nets["value"], "value", limit=2048)

        text = export_weights.header("general", nets, {}, {}, spec_version=spec.schema["spec_version"])
        self.assertIn("kRequiredMaxWidth = 3072", text)
        stream = synthetic.stream(3, turns=4)
        rec = native.extract(stream)
        expected = quant.forward_raw(nets["value"], spec.get(rec, "v"))[:, 0]
        p_inputs = np.concatenate([spec.get(rec, n).reshape(len(rec), -1) for n in ("g", "u", "m")], axis=1)
        expected_p = quant.forward_raw(nets["squad"], p_inputs)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            header = folder/"weights_general.hpp"
            header.write_text(text)
            namespace = "fusion::models::general"
            actual = native.run_infer_check(native.infer_check(header, namespace), stream, len(rec))
            np.testing.assert_array_equal(actual[:, -1], expected)
            np.testing.assert_array_equal(actual[:, 15:140], expected_p)
            runtime = native.run_tool(native.model_tool("runtime_check", header, namespace), stream.encode())
            self.assertEqual(runtime.returncode, 0, runtime.stderr.decode())
            self.assertIn("allocating_turns 0", runtime.stdout.decode())
            self.assertIn("accumulator_mismatches 0", runtime.stdout.decode())
            export_weights.registry(folder, {"general": {"header": header.name, "widest_layer": width}})
            check = native.run_tool(native.registry_check(folder))
            self.assertEqual(check.returncode, 0, check.stderr.decode())
            self.assertEqual(len(check.stdout.decode().splitlines()), 14)


if __name__ == "__main__":
    unittest.main()
