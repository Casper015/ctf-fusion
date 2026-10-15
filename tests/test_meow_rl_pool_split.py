"""The Meowmind RL train/test split of the AI_Test bots (task T0058, docs/MEOWMIND_RL_POOL_SPLIT.md)."""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

from training.meow_rl.pool import Pool

REPO = Path(__file__).resolve().parents[1]
POOL_V2 = REPO / "profiles" / "meow_rl_training_pool_v2.json"
POOL_V1 = REPO / "profiles" / "meow_rl_training_pool.json"
SCREEN = REPO / "profiles" / "meow_rl_screen.json"
# every file of a held-out lineage that exists in the arena or AI_Test (judge and stress-pack copies)
HELD_OUT_FILES = {"fable-5.1.cpp", "mimo-v2.6-pro.cpp", "kimi-k3.cpp", "glm-5.3.cpp", "Player2.cpp", "Player4.cpp", "Player6.cpp"}


class Split(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(POOL_V2.read_text(encoding="utf-8"))
        cls.pool = Pool(cls.doc)

    def test_pool_loads_under_the_cap(self):
        self.assertEqual(len(self.pool.families), 9)
        self.assertLessEqual(max(self.pool.shares.values()), self.pool.cap)
        self.assertEqual(self.pool.cap, 0.15)
        for b in self.doc["bots"]:
            self.assertRegex(b.get("sha256", ""), r"^[0-9a-f]{64}$", b["name"])

    def test_no_training_bot_is_held_out(self):
        held = set(self.doc["held_out"])
        self.assertEqual(held, {"Fable", "MiMo", "Kimi", "GLM", "Player2", "Player4", "Player6"})
        names = {b["name"] for b in self.doc["bots"]}
        self.assertFalse(names & held)
        files = {Path(b["path"]).name for b in self.doc["bots"]}
        self.assertFalse(files & HELD_OUT_FILES)
        self.assertNotIn("fable", self.pool.families)

    def test_one_copy_of_identical_files(self):
        pins = [b["sha256"] for b in self.doc["bots"]]
        self.assertEqual(len(pins), len(set(pins)))

    def test_t0055_pool_is_contained(self):
        if not POOL_V1.is_file():
            self.skipTest("T0055's pool is not merged yet")
        v1 = {(b["name"], b["path"], b["sha256"]) for b in json.loads(POOL_V1.read_text(encoding="utf-8"))["bots"]}
        v2 = {(b["name"], b["path"], b["sha256"]) for b in self.doc["bots"]}
        self.assertTrue(v1 <= v2, v1 - v2)

    def test_screen_opponents_are_held_out(self):
        if not SCREEN.is_file():
            self.skipTest("the screen spec (T0056) is not merged yet")
        roster = set(json.loads(SCREEN.read_text(encoding="utf-8"))["opponents"]["reference"])
        self.assertTrue(roster <= set(self.doc["held_out"]), roster - set(self.doc["held_out"]))

    @unittest.skipUnless(os.environ.get("AI_TEST_ROOT"), "set AI_TEST_ROOT to the AI_Test checkout to verify the pins")
    def test_pins_match_the_checkout(self):
        root = Path(os.environ["AI_TEST_ROOT"])
        checked = 0
        for b in self.doc["bots"]:
            path = REPO / b["path"][len("@repo/"):] if b["path"].startswith("@repo/") else root / b["path"]
            if not path.is_file():
                continue                                           # e.g. judge copies of another arena layout
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), b["sha256"], b["name"])
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
