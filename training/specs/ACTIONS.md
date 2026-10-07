# Action indexing contract

One contract shared by the labels (`fusion/labels.py`), the models (`fusion/models.py`) and the C++ heads (`inference/policy.hpp`).

## Moves

| Index | Protocol letter | (dx, dy) | Engine name |
|---:|---|---|---|
| 0 | `S` | (0, 0) | WAIT |
| 1 | `U` | (0, -1) | N |
| 2 | `D` | (0, 1) | S |
| 3 | `L` | (-1, 0) | W |
| 4 | `R` | (1, 0) | E |

`y` grows downward, as in the protocol. A move is *legal* when the destination is open; walls include the border.

## Joint moves (125)

For the three units of a team, sorted by id (`3*team + k`, k = 0, 1, 2):

```text
joint = m0 * 25 + m1 * 5 + m2        m_k in 0..4
m0 = joint / 25,  m1 = (joint / 5) % 5,  m2 = joint % 5
```

**Legal support:** `legal[0][m0] && legal[1][m1] && legal[2][m2]`.

**Units off the board** (dead, not respawning this turn) have only `S` legal. The joint distribution therefore puts all of their mass on `S`, and they contribute nothing to the loss.

**Units respawning this turn** move from their predicted spawn cell (`origin`). Their legality follows that cell.

**Logits:** `mover(0,m0) + mover(1,m1) + mover(2,m2) + squad[joint]`, summed in exactly that order in both languages. Illegal entries get probability 0.

## Action heads (conditioned on a unit's move)

| Index | Act | Effective (`act_legal[k][m][a]`) | Protocol |
|---:|---|---|---|
| 0 | wait | always, for a unit on the board | `-` |
| 1 | pickup | unladen, and the destination holds an available flag | `P` |
| 2 | drop | carrying | `X` |
| 3 | attack | unladen, not respawning, at least one candidate target | target id |

**Target head.** A pointer over `cand[k][0..7]`: the hittable enemies within Manhattan 4 of the unit's origin, nearest first, ties broken by id. Farther targets cannot come within range 2 this turn. Its input for a move `m` uses the candidate's `hit_<m>` bit.

**What the search consumes.** For every destination of an enemy unit:
- `P(attack | destination)` from the act head, which becomes `search::Rival::fire`;
- the target distribution, which spreads the unit's attention.

## Label canonicalization (requested, not resolved)

Labels come from the reply line as sent, never from position changes. A move stopped by a collision keeps its requested direction; `blocked` marks it for diagnostics.

**Requests the engine would ignore** are mapped to their effect:

| Request | Label |
|---|---|
| move into a wall | `S` |
| `P` where the destination holds no available flag | wait |
| `X` without a flag | wait |
| attack while carrying, while respawning, on a teammate, on an off-board unit or beyond Manhattan 4 | wait |
| attack on a valid target that did not fit the 8 candidate slots | attack, target label -1 (excluded) |

**Missing or malformed replies** (timeout, parse error, dead process) give no policy label for that row. Value labels never depend on replies.

## Value targets

| Head | Target | Exported |
|---|---|---|
| 0 | pairwise final outcome of the row's team: mean over other teams of win 1 / draw 0.5 / loss 0 | yes |
| 1 | the team captures within the next 20 settlements | no (training only) |
| 2 | a unit of the team dies within the next 10 settlements | no (training only) |
