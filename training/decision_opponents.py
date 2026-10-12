"""Expanded opponent registry and fixture sampler for designated-carrier decision data (task T0035).

    python -m training.decision_collect list --out build/decision/registry.json      # availability, no build

The registry (profiles/decision_opponents.json, profiles/decision_variants.json) has three components:

  reference  the twelve frozen T0010 references, by name and source SHA-256 from their schedule;
  addition   six fingerprint-screened representatives, by stable source id and SHA-256 from the training-selection
             CSV;
  neural     every P-only and P+V entry of the P-only variant catalogs (controls and hybrids are not P-only and are
             excluded by role); the catalogs include each other, so they are merged by name, and a name defined twice
             must be defined identically.

Sources resolve in this order, and only to the recorded SHA-256: a repository path (@repo/...), the arena checkout
(AI_TEST_ROOT), then the hash-named vendored copy roster/bots/src/<sha256>.cpp. Every use of a path other than the
recorded one is a substitution and is recorded; a different source is never substituted. A neural entry resolves by
bundling it (bots/p_only/tools/bundle.py): its transitive includes and model registry must exist, so a wrapper whose
weights are missing is disabled ("missing model registry") with zero sampling mass, never compiled raw.

Lineage comes from the roster inventory (roster/new_bots/inventory.json) by source SHA-256, following byte and code
aliases; a neural group takes the inventory lineage of its carrier's origin source (named in the variants config),
not its architecture name. Exposure units: byte- or code-identical sources share one unit, and neural entries with
the same bundle text (before any PONLY_STATS define) share one; a unit is drawn at most once per game. Reserved
lineages (and any alias of their sources) are excluded wherever they appear. Every exclusion, disabled entry,
quarantine and alias stays in the registry, and any of them changes the effective pool identity.

Mixture: composition weights over the components present (default 60/20/20), equal over units inside reference and
addition, equal over neural groups (C08, v12) and then over units inside a group. An optional lineage cap scales any
lineage above it down (roster.sampling.cap_lineages; relaxed to 1/lineages when infeasible, which is reported). A
game's n - 1 opponent seats are drawn without replacement by that distribution, so realized per-count shares differ
from the targets; simulate() measures them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from fusion.paths import REPO

SCHEMA_REGISTRY = "ctf-fusion-decision-registry-1"
SCHEMA_SCHEDULE = "ctf-fusion-decision-schedule-1"
COMPONENTS = ("reference", "addition", "neural")
TEAM_COUNTS = tuple(range(2, 16))
OPPONENTS = REPO / "profiles" / "decision_opponents.json"
VARIANTS = REPO / "profiles" / "decision_variants.json"
VENDORED = REPO / "roster" / "bots" / "src"
STATS_DEFINE = "#define PONLY_STATS 1\n"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(*parts, bits=31) -> int:
    """The same derivation as roster.strength.derive (fixture seeds are reproducible from the seed)."""
    h = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << bits) - 1)


def load_json(path) -> dict:
    p = Path(path)
    return json.loads((p if p.is_absolute() else REPO / p).read_text(encoding="utf-8"))


def repo_path(rel: str) -> Path:
    return REPO / rel[len("@repo/"):] if rel.startswith("@repo/") else REPO / rel


# ---------------------------------------------------------------- seeds

def consumed_seeds(digest_md: Path = REPO / "docs" / "DIGEST.md") -> set[int]:
    """Every seed in the DIGEST seed ledger (single seeds and ranges); a new schedule may not reuse one."""
    out: set[int] = set()
    for line in Path(digest_md).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d{8})(?:\s*[–-]\s*(\d{8}))?\s*\|", line)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            out.update(range(lo, hi + 1))
    return out


# ---------------------------------------------------------------- inventory and lineage

class Inventory:
    def __init__(self, path: str | Path):
        doc = load_json(path)
        self.path = str(path)
        self.sha256 = sha256_file(path if Path(path).is_absolute() else REPO / path)
        self.by_id = {b["id"]: b for b in doc["bots"]}
        self.by_sha = defaultdict(list)
        for b in doc["bots"]:
            self.by_sha[b["source_sha256"]].append(b)

    def root_id(self, entry: dict) -> str:
        """The id an entry aliases to (byte alias first, then code alias), followed to the end of the chain."""
        seen = set()
        while entry["id"] not in seen:
            seen.add(entry["id"])
            nxt = entry.get("byte_alias_of") or entry.get("code_alias_of")
            if not nxt or nxt not in self.by_id:
                break
            entry = self.by_id[nxt]
        return entry["id"]

    def lookup_sha(self, sha: str) -> dict | None:
        hits = self.by_sha.get(sha, [])
        return hits[0] if hits else None

    def lineage_of_sha(self, sha: str) -> tuple[str | None, str | None]:
        """(lineage, alias root id) of a source; (None, None) when the inventory does not know it."""
        e = self.lookup_sha(sha)
        if e is None:
            return None, None
        root = self.by_id[self.root_id(e)]
        return root.get("lineage_group") or e.get("lineage_group"), root["id"]

    def reserved_ids(self, ids: list[str]) -> set[str]:
        """Reserved source ids plus every inventory entry that aliases one of them."""
        roots = {self.root_id(self.by_id[i]) for i in ids if i in self.by_id}
        return {i for i, e in self.by_id.items() if self.root_id(e) in roots} | set(ids)


# ---------------------------------------------------------------- catalogs

def load_catalog(path: str | Path, seen: set | None = None) -> list[dict]:
    """A P-only variant catalog with its includes (repository-relative), in order, without the name check of
    bots/p_only/tools/build.py: merging happens in merge_catalogs."""
    doc = load_json(path)
    out = []
    for inc in doc.get("include", []):
        out += load_catalog(inc)
    out += [dict(v, _catalog=str(path)) for v in doc["variants"]]
    return out


def merge_catalogs(paths: list[str]) -> list[dict]:
    """Every entry of the catalogs once; a name defined twice must be defined identically (refused otherwise)."""
    merged: dict[str, dict] = {}
    for p in paths:
        for v in load_catalog(p):
            body = {k: x for k, x in v.items() if k != "_catalog"}
            if v["name"] in merged:
                old = {k: x for k, x in merged[v["name"]].items() if k != "_catalog"}
                if old != body:
                    raise SystemExit(f"variant {v['name']} is defined differently in {merged[v['name']]['_catalog']} and {v['_catalog']}")
                continue
            merged[v["name"]] = v
    return list(merged.values())


def neural_group(v: dict) -> str:
    return {"c08": "C08", "v12": "v12"}.get(str(v.get("carrier", "")).lower(), str(v.get("carrier")))


_CODE_CACHE: dict[str, str] = {}


def file_code_sha(path: Path, raw_sha: str) -> str:
    """Code identity of one bundled file: the comment- and layout-free token stream (roster.inventory.normalized_code)
    for hand-written files; generated model headers (inference/models*/) by their bytes."""
    if raw_sha not in _CODE_CACHE:
        rel = Path(path).resolve().relative_to(REPO).as_posix()
        if rel.startswith("inference/models"):
            _CODE_CACHE[raw_sha] = raw_sha
        else:
            from roster.inventory import normalized_code  # noqa: PLC0415
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            _CODE_CACHE[raw_sha] = hashlib.sha256(normalized_code(text).encode()).hexdigest()
    return _CODE_CACHE[raw_sha]


def bundle_variant(v: dict) -> dict:
    """Bundle identity of a variant: sha256 of the bundle text (without PONLY_STATS), the code identity (aliases:
    the same code in the same bundle order, comments and layout ignored), and the inputs with hashes.
    SystemExit when an include (for example a model registry) is missing."""
    tools = str(REPO / "bots" / "p_only" / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from bundle import bundle  # noqa: PLC0415
    text, files = bundle(REPO / v["source"])
    inputs = [{"path": Path(f).resolve().relative_to(REPO).as_posix(), "sha256": sha256_file(f)} for f in files]
    code = digest([file_code_sha(f, i["sha256"]) for f, i in zip(files, inputs)])
    return {"bundle_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "code_sha256": code, "inputs": inputs}


# ---------------------------------------------------------------- source resolution

def resolve_source(paths: list[str], sha: str, root: Path | None) -> dict:
    """The first candidate path whose bytes have exactly `sha`: repository, arena checkout, then vendored copy."""
    tried = []
    cands = []
    for p in paths:
        if p.startswith("@repo/"):
            cands.append(("repository", repo_path(p), p))
        elif root is not None:
            cands.append(("arena", Path(root) / p, p))
    cands.append(("vendored", VENDORED / f"{sha}.cpp", f"@repo/roster/bots/src/{sha}.cpp"))
    for how, path, label in cands:
        if path.is_file():
            got = sha256_file(path)
            if got == sha:
                return {"status": "resolved", "how": how, "path": str(path), "label": label,
                        "substituted": label != paths[0] if paths else True, "recorded_path": paths[0] if paths else None}
            tried.append(f"{label}: sha256 {got[:12]} differs")
        else:
            tried.append(f"{label}: missing")
    return {"status": "unresolved", "tried": tried}


# ---------------------------------------------------------------- the registry

def build_registry(root: Path | None = None, opponents: str | Path = OPPONENTS, variants: str | Path = VARIANTS,
                   resolve_neural: bool = True) -> dict:
    cfg, vcfg = load_json(opponents), load_json(variants)
    inv = Inventory(cfg["inventory"])
    reserved = inv.reserved_ids(cfg["reserved_sources"])
    reserved_lineages = set(cfg["reserved_lineages"])
    quarantine = cfg.get("quarantine", {}) | vcfg.get("quarantine", {})
    members = []

    def lineage_for(sha: str, name: str) -> tuple[str, str | None, list[str]]:
        lin, root_id = inv.lineage_of_sha(sha)
        warn = [] if lin else [f"{name}: source not in the inventory; lineage unknown"]
        return lin or f"unknown:{name}", root_id, warn

    warnings: list[str] = []
    # references
    sched = load_json(cfg["references"]["schedule"])
    pool = {e["name"]: e for e in load_json(cfg["references"]["pool"])["bots"]}
    for ref in sched["reference"]:
        lin, rid, w = lineage_for(ref["source_sha256"], ref["name"])
        warnings += w
        members.append({"name": ref["name"], "component": "reference", "kind": "source", "group": ref["name"],
                        "family": pool.get(ref["name"], {}).get("family", ref["name"].lower()),
                        "source_sha256": ref["source_sha256"], "paths": [ref["path"]], "lineage": lin,
                        "inventory_root": rid})
    # additions
    with open(REPO / cfg["additions"]["csv"], encoding="utf-8", newline="") as fh:
        rows = {r["original_pool_name"]: r for r in csv.DictReader(fh)}
    for a in cfg["additions"]["bots"]:
        row = rows.get(a["pool_name"])
        if row is None or row["id"] != a["source_id"] or row["source_sha256"] != a["source_sha256"]:
            raise SystemExit(f"addition {a['name']}: the selection CSV does not list id {a['source_id']} with that SHA-256")
        lin, rid, w = lineage_for(a["source_sha256"], a["name"])
        warnings += w
        members.append({"name": a["name"], "component": "addition", "kind": "source", "group": a["name"],
                        "family": f"addition:{a['name']}", "source_sha256": a["source_sha256"], "paths": [a["source_id"]],
                        "lineage": lin, "inventory_root": rid, "selection_lineage": row["lineage"]})
    # neural
    origins = {}
    for group, o in vcfg["origins"].items():
        e = inv.by_id.get(o["source_id"])
        if e is None:
            raise SystemExit(f"origin of {group}: {o['source_id']} is not in the inventory")
        origins[group] = {"source_id": o["source_id"], "lineage": inv.by_id[inv.root_id(e)].get("lineage_group") or e["lineage_group"],
                          "basis": o["basis"]}
    catalog = merge_catalogs(vcfg["catalogs"])
    excluded_roles = set(vcfg["exclude_roles"])
    for v in catalog:
        m = {"name": v["name"], "component": "neural", "kind": "variant", "group": neural_group(v), "catalog": v["_catalog"],
             "role": v["role"], "source": v["source"], "models": v.get("models"), "tier": v.get("tier"),
             "value_on": v.get("role") == "ponly_v" or bool(v.get("value_on")) or bool(v.get("value_tier")),
             "family": f"neural:{neural_group(v)}"}
        if v["role"] in excluded_roles:
            m |= {"status": "not_registered", "reason": f"role {v['role']}: not a P-only or P+V entry"}
            members.append(m)
            continue
        origin = origins.get(m["group"])
        m["lineage"] = origin["lineage"] if origin else f"unknown:{m['group']}"
        if origin is None:
            warnings.append(f"{v['name']}: no origin for carrier group {m['group']}")
        if resolve_neural:
            try:
                b = bundle_variant(v)
                m |= b | {"model_registry": [i["path"] for i in b["inputs"] if i["path"].endswith("registry.hpp")]}
            except SystemExit as exc:
                msg = str(exc).replace(str(REPO) + "/", "").replace(str(REPO) + "\\", "")
                m |= {"status": "disabled", "reason": f"missing model registry or include: {msg}"}
        else:
            m |= {"status": "unresolved", "reason": "not bundled (listing without neural resolution)"}
        members.append(m)
    # resolution, exclusion and quarantine
    for m in members:
        if m.get("status"):
            continue
        if m["kind"] == "source":
            m["resolution"] = resolve_source(m["paths"], m["source_sha256"], root)
            if m["resolution"]["status"] != "resolved":
                m |= {"status": "disabled", "reason": "source not found with its recorded SHA-256"}
                continue
            e = inv.lookup_sha(m["source_sha256"])
            if e and (e.get("local_includes") or e.get("unresolved_includes")):
                m |= {"status": "disabled", "reason": "the source has local includes; compile it only through a bundle "
                                                      "that hashes them"}
                continue
        reserved_hit = m["lineage"] in reserved_lineages or (m.get("inventory_root") in reserved) or \
            any(p in reserved for p in m.get("paths", []))
        if reserved_hit:
            m |= {"status": "excluded", "reason": "reserved lineage (unseen-opponent evaluation)"}
        elif m["name"] in quarantine:
            m |= {"status": "quarantined", "reason": quarantine[m["name"]]}
        else:
            m["status"] = "enabled"
    # exposure units: aliases share one
    for m in members:
        if m["status"] != "enabled":
            continue
        if m["kind"] == "source":
            m["unit"] = "src:" + (m.get("inventory_root") or m["source_sha256"])
        else:
            m["unit"] = "code:" + m["code_sha256"]
    units = defaultdict(list)
    for m in members:
        if m.get("unit"):
            units[m["unit"]].append(m["name"])
    for m in members:
        if m.get("unit"):
            m["aliases"] = sorted(units[m["unit"]])
            m["representative"] = sorted(units[m["unit"]])[0] == m["name"]
    designated = designated_identity(cfg, catalog)
    for m in members:
        if m.get("code_sha256") and designated.get("code_sha256") == m["code_sha256"]:
            m["same_policy_as_designated"] = True
    body = {"schema": SCHEMA_REGISTRY, "config_sha256": sha256_file(REPO / opponents if not Path(opponents).is_absolute() else opponents),
            "variants_config_sha256": sha256_file(REPO / variants if not Path(variants).is_absolute() else variants),
            "inventory": {"path": inv.path, "sha256": inv.sha256}, "designated": designated,
            "composition": cfg["composition"], "lineage_cap": cfg.get("lineage_cap"),
            "reserved_lineages": sorted(reserved_lineages), "origins": origins,
            "members": [{k: x for k, x in m.items() if k != "inputs"} | ({"inputs_sha256": digest(m["inputs"])} if "inputs" in m else {})
                        for m in members],
            "warnings": warnings}
    body["pool_identity"] = pool_identity(body)
    body["counts"] = {f"{c}:{st}": n for (c, st), n in sorted(Counter((m["component"], m["status"]) for m in members).items())}
    body["_inputs"] = {m["name"]: m["inputs"] for m in members if "inputs" in m}
    return body


def designated_identity(cfg: dict, catalog: list[dict]) -> dict:
    d = dict(cfg["designated"])
    v = next((x for x in catalog if x["name"] == d["variant"]), None)
    if v is None:
        raise SystemExit(f"designated carrier {d['variant']} is not in the catalogs")
    try:
        b = bundle_variant(v)
    except SystemExit as exc:
        raise SystemExit(f"designated carrier {d['variant']} does not resolve: {exc}") from exc
    d |= {"bundle_sha256": b["bundle_sha256"], "code_sha256": b["code_sha256"], "inputs_sha256": digest(b["inputs"]),
          "source": v["source"]}
    ref = cfg.get("designated_reference")
    if ref:   # the incumbent of an earlier collection: compare the build inputs file by file (paths normalized)
        man = load_json(ref["manifest"])["variants"][ref["variant"]]
        then = {i["path"].replace("\\", "/"): i["sha256"] for i in man["inputs"]}
        now = {i["path"]: i["sha256"] for i in b["inputs"]}
        d["reference"] = {"manifest": ref["manifest"], "variant": ref["variant"], "binary_sha256": man["binary_sha256"],
                          "same_inputs": then == now,
                          "differences": sorted(set(then.items()) ^ set(now.items()))[:10]}
    return d


def pool_identity(reg: dict) -> str:
    """What changes the effective pool: every member's status, unit, lineage and source/bundle identity, the
    composition, the cap, the reserved lineages and the designated carrier."""
    keep = ("name", "component", "status", "unit", "lineage", "source_sha256", "bundle_sha256", "code_sha256", "group")
    return digest({"members": sorted(({k: m.get(k) for k in keep} for m in reg["members"]), key=lambda x: x["name"]),
                   "composition": reg["composition"], "lineage_cap": reg["lineage_cap"],
                   "reserved": reg["reserved_lineages"], "designated": reg["designated"]})


# ---------------------------------------------------------------- mixture and sampling

def units_of(reg: dict) -> dict[str, dict]:
    out = {}
    for m in reg["members"]:
        if m.get("status") == "enabled" and m.get("representative"):
            out[m["unit"]] = {"unit": m["unit"], "name": m["name"], "component": m["component"], "group": m["group"],
                              "lineage": m["lineage"], "aliases": m["aliases"]}
    return out


def mixture(reg: dict) -> dict:
    """Unit distribution: composition over components, groups then units; then the optional lineage cap."""
    from roster.sampling import cap_lineages  # noqa: PLC0415
    units = units_of(reg)
    by_comp = defaultdict(list)
    for u in units.values():
        by_comp[u["component"]].append(u)
    comp = {c: float(reg["composition"].get(c, 0.0)) for c in COMPONENTS if by_comp[c] and reg["composition"].get(c, 0) > 0}
    total = sum(comp.values())
    comp = {c: w / total for c, w in comp.items()}
    rho = {}
    for c, w in comp.items():
        groups = defaultdict(list)
        for u in by_comp[c]:
            groups[u["group"] if c == "neural" else u["unit"]].append(u)
        for g in groups.values():
            for u in g:
                rho[u["unit"]] = w / len(groups) / len(g)
    lineage_of = {k: u["lineage"] for k, u in units.items()}
    pre = lineage_mass(rho, lineage_of)
    cap = reg.get("lineage_cap")
    report = {"requested_cap": cap, "lineages": len(set(lineage_of.values())), "pre_cap": pre}
    if cap:
        effective = max(cap, 1.0 / len(set(lineage_of.values())))
        rho = cap_lineages(rho, lineage_of, cap)
        report |= {"effective_cap": effective, "relaxed": effective > cap, "binding": any(v > cap + 1e-12 for v in pre.values())}
    report["post_cap"] = lineage_mass(rho, lineage_of)
    comp_post = defaultdict(float)
    for k, p in rho.items():
        comp_post[units[k]["component"]] += p
    return {"rho": rho, "units": units, "composition_target": comp, "composition_after_cap": dict(comp_post), "cap": report}


def lineage_mass(rho: dict, lineage_of: dict) -> dict:
    out = defaultdict(float)
    for k, p in rho.items():
        out[lineage_of[k]] += p
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def draw_opponents(rng: random.Random, mix: dict, k: int) -> list[str]:
    """k distinct units, sequentially without replacement, each by rho renormalized over the units left."""
    left = dict(mix["rho"])
    out = []
    for _ in range(min(k, len(left))):
        names = sorted(n for n, p in left.items() if p > 0)
        if not names:
            break
        pick = rng.choices(names, [left[n] for n in names])[0]
        out.append(pick)
        del left[pick]
    if len(out) < k:
        raise SystemExit(f"only {len(out)} enabled exposure units for {k} opponent seats")
    return out


def make_schedule(reg: dict, seed: int, maps_per_count: int, teams=TEAM_COUNTS, allow_consumed: bool = False) -> dict:
    if not allow_consumed and seed in consumed_seeds():
        raise SystemExit(f"seed {seed} is in the DIGEST seed ledger: never reuse a seed")
    mix = mixture(reg)
    fixtures = []
    for n in teams:
        seats = sorted(range(n), key=lambda k: derive(seed, n, k, "seat"))
        for m in range(maps_per_count):
            rng = random.Random(derive(seed, n, m, "opponents", bits=63))
            us = draw_opponents(rng, mix, n - 1)
            u = [mix["units"][x] for x in us]
            fixtures.append({"fixture_id": f"s{seed}-n{n:02d}-m{m:05d}", "teams": n, "map_seed": derive(seed, n, m, "map"),
                             "flag_seed": derive(seed, n, m, "flag", bits=63), "seat": seats[m % n],
                             "opponents": [x["name"] for x in u], "units": us, "components": [x["component"] for x in u],
                             "groups": [x["group"] for x in u], "lineages": [x["lineage"] for x in u]})
    sched = {"schema": SCHEMA_SCHEDULE, "seed": seed, "maps_per_count": maps_per_count, "team_counts": list(teams),
             "pool_identity": reg["pool_identity"], "designated": reg["designated"], "fixtures": fixtures}
    sched["schedule_identity"] = digest({k: v for k, v in sched.items() if k != "created"})
    return sched


def simulate(reg: dict, games: int = 2000, seed: int = 0, teams=TEAM_COUNTS) -> dict:
    """Realized seat shares per count (components, lineages) of the without-replacement sampler."""
    mix = mixture(reg)
    out = {"composition_target": mix["composition_target"], "composition_after_cap": mix["composition_after_cap"],
           "cap": mix["cap"], "per_count": {}}
    for n in teams:
        rng = random.Random(derive(seed, n, "simulate"))
        comp, lin, inc = Counter(), Counter(), Counter()
        for _ in range(games):
            us = draw_opponents(rng, mix, n - 1)
            inc.update(us)
            for u in us:
                comp[mix["units"][u]["component"]] += 1
                lin[mix["units"][u]["lineage"]] += 1
        seats = games * (n - 1)
        out["per_count"][str(n)] = {
            "component_share": {c: round(comp[c] / seats, 4) for c in mix["composition_target"]},
            "top_lineages": {k: round(v / seats, 4) for k, v in lin.most_common(4)},
            "units_never_drawn": sorted(set(mix["units"]) - set(inc))}
    return out
