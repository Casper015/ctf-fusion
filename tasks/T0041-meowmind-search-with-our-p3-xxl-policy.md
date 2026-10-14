---
id: T0041
title: "meowmind search with our P3 XXL policy: T0011 and distilled 'both' bots (Mac pilot)"
status: done
owner: claude@caspers-macbook
needs: []
writes: [bots/meowmind_p/]
depends: []
land: pr
branch: claude/t0041-meowmind-search-with-our-p3-xxl-policy
updated: 2026-10-14
---

## Goal

Commit the two meowmind hybrid bots that use our P3 XXL joint policy for their own candidate prior and as an extra opponent expert (T0011 XXL and T0032 distill_s101), the generator that builds them, and the Mac pilot round-robin results, labeled as development-machine results, not host evidence. Casper chose to publish meowmind's source in this repository.

## Done when

PR merged with both bundled bots, make.py, README with exact rebuild commands and Mac results; no claims beyond 'not shown'.

## Handoff

*2026-10-14 21:56Z claude@caspers-macbook:*

result: bots/meowmind_p/ (PR #38 merged): meowmind with our P3 XXL policy (T0011 and distilled 'both' bundles, make.py, Mac round-robins). Gain over meowmind not shown (six variants vs one control, uncorrected); no host evidence.

## Log

- 2026-10-14 19:34Z claude@caspers-macbook: created
- 2026-10-14 19:35Z claude@caspers-macbook: amended writes -> bots/meowmind_p/
- 2026-10-14 19:35Z claude@caspers-macbook: claimed
- 2026-10-14 19:38Z claude@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/38 opened: both bundled bots, make.py, README with Mac pilot results (gain not shown). MARKDOWN_INDEX left to after T0037.
- 2026-10-14 19:38Z claude@caspers-macbook: handoff -> review
- 2026-10-14 21:56Z claude@caspers-macbook: done
