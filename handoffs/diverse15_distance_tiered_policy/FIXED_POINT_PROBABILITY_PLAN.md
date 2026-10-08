# Fixed-point softmax, value and probability computation on the CPU

The floating-point tail of our networks can be replaced selectively with fixed-point computation. The useful opportunities are removing unnecessary sigmoid calls, replacing repeated exponentials with validated approximations, and reusing integer sampling tables. Actual speedup must be measured on the target CPU: integer arithmetic alone does not guarantee faster inference.

This is an implementation proposal, not a benchmark result or a deployed change. The inspected source is ctf-fusion `3d47a01ccc5d05682168f4d4b5c8839ec948ae45`. Preserve the 2-second startup limit, complete-turn 50ms deadline, existing feature/adapter contracts and unchanged legacy fallback.

## Current implementation and proposed representations

The network already produces integer accumulators. `inference/policy.hpp` converts logits to double and calls `exp` for softmax; `inference/value.hpp` converts the value logit to double and applies sigmoid. `inference/adapter.hpp` mixes probabilities and samples through floating-point cumulative sums.

| Use | Proposed representation | Condition |
|---|---|---|
| Ranking one V prediction per candidate | INT32 raw logit, or aligned INT64 logit | Same model, calibration and output scale; no averaging or additional rewards. |
| Expected value across scenarios | Fixed-point probability with INT64 intermediates | Apply sigmoid before averaging; retain probability semantics. |
| Neural categorical sampling | Nonnegative integer weights and an integer CDF | Sample the actual declared weight distribution. |
| Marginal probabilities and mixtures | UINT32 probability with 24 fractional bits | Use consistent scale, rounding, mass and adapter semantics. |

## 1. Omit sigmoid only when ordering is sufficient

Sigmoid is strictly increasing:

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))>0.
$$

For a single prediction from the same V model, comparing logits preserves the mathematical ordering of sigmoid outputs. The existing `valueRaw()` is a useful starting point. The floating reference can round extreme outputs to the same probability, so exact action equivalence also needs saturation and tie-breaking tests. Raw outputs from different models or scales require alignment and appropriate calibration before comparison.

Expected value is different:

$$
\mathbb E[\sigma(z)]\ne\sigma(\mathbb E[z])
$$

For example, candidate A has equally likely logits -8 and 2: its mean logit is -3, but its mean probability is approximately 0.4406. Candidate B has logit -1 in both scenarios: its mean logit is -1, but its mean probability is approximately 0.2689. Averaging logits selects B; averaging probabilities selects A.

Therefore, scenario averages, chance-node expectations and mixtures with realized rewards must use value in the intended probability/utility space. Terminal pairwise outcomes also stay in that space; they cannot be compared directly with raw logits. For these paths, a sigmoid lookup/interpolation kernel can produce fixed-point probabilities without a runtime exponential. Its range, saturation and error must be declared and validated.

## 2. Integer softmax and sampling

For legal actions, subtract the maximum logit and approximate the exponential with a lookup table or fixed-point polynomial:

$$
w_i\approx K\exp(z_i-z_{max}),\qquad W=\sum_i w_i.
$$

Assign zero weight to illegal actions. Logits from mover and squad heads must first share a defined fixed-point scale; their raw accumulators are not necessarily directly additive. Widen differences/intermediates where required, and bound every sum and product.

When a consumer only needs a categorical draw, normalized probabilities need not be materialized. Generate an unbiased integer r in [0,W), then return the first index whose cumulative weight exceeds r. Its sampling probability is exactly `w_i/W`, for the integer weights actually produced. The exponential approximation is the source of approximation to the original softmax.

Prepare and cache the CDF once for an unchanged prediction, then reuse it across search samples. Benchmark the cost of lookup, setup, bounded random-number generation and sampling. Integer division can be expensive, so a version that normalizes every element using integer division may lose to floating point.

If analytical marginals are also required, compute them from the same canonical weights. There are two coherent contracts:

- **Weight contract:** the sampler uses w/W; analytical marginals may be fixed-point approximations of sums of weights divided by W. Offline likelihood must score the actual w/W sampler, and the analytical rounding policy is part of the deployment identity.
- **Fixed-mass contract:** normalize to integer masses summing to S, then use those exact same masses for the CDF and marginal sums. Offline likelihood scores mass/S. Define deterministic residual redistribution and preserve zero mass for illegal actions.

Choose one contract explicitly. Mixing a weight-based sampler with a gate that scores a different rounded distribution recreates the offline/runtime mismatch already fixed in this project.

## 3. Probability precision and accumulators

An initial candidate is UINT32 storage with 24 fractional bits:

$$
S=2^{24}=16{,}777{,}216,\qquad P=\operatorname{round}(pS).
$$

Thus 1.0 is S, 0.5 is 8,388,608, and the representable step is approximately 5.96e-8. This is a proposed precision to evaluate, not a universal optimum. One probability unit does not bound the error from exponential approximation, tail truncation, mixing and repeated rounding.

Use INT64/UINT64 where required for products, normalization, cumulative weights and expected-value sums. A Q24 probability product can reach 2^48 before rescaling; sums need their own bounds. Use wide intermediates for mixture alpha and keep sum-to-one behavior deterministic. A long product of tiny branch probabilities can still underflow at fixed precision; preserve appropriate precision or use a separately evaluated log-domain/alternative calculation for that consumer.

Avoid low-bit probability outputs that erase relevant rare joint actions. Test discarded probability mass, held-out zero-probability labels and low-probability attack/drop/interception events. Illegal actions must remain impossible. Approximation error can change decisions whose value margins are small, even when average loss barely moves.

## 4. Preserve the adapter and fallback

Joint-v2 keeps one mixture-component choice per opponent team and scenario; a neural draw then samples conditional action/target heads. Marginal-v1 samples independent mixed marginals. Integer kernels must preserve whichever declared adapter is selected.

Changing random-number mapping, rounding or the legacy distribution can alter action streams. Keep the original legacy path when alpha is zero, and verify stream equivalence. Treat integerizing the legacy sampler as a separate experiment. Include the tail-kernel version, LUT/polynomial coefficients, logit/probability scales, rounding rules, sampler contract, calibration, weights and routing configuration in the artifact identity.

## 5. Implementation order and acceptance

1. Measure current softmax, sigmoid, CDF construction and sampling within a representative complete turn. Compare double, FP32 and proposed fixed-point kernels on the same CPU/workload.
2. Remove sigmoid from eligible same-model single-value ranking paths. Verify exact ordering, ties and terminal-state handling.
3. Cache immutable prediction CDFs and conditional-head results. Verify invalidation when state, move, model or legal support changes.
4. Add integer exponential/sigmoid approximations and the chosen canonical probability contract. Test masked/all-stay/one-legal-action cases, tails, scale alignment, overflow and RNG bias.
5. Recalibrate where necessary and evaluate the actual deployed adapter through the integer-artifact gate. Report NLL, event calibration, decision regret and paired play; numerical agreement alone does not establish strength.
6. Measure complete-turn p99/p99.9/max and faults under the real deadline. Retain the floating reference and fallback for configurations that lose accuracy, strength or latency.

The priority is removing repeated nonlinear work while retaining decision semantics. A fully integer tail is optional; a measured mixed implementation may be preferable on some CPUs. Offline training can remain FP32/CUDA and export this CPU representation.

## References and verification status

Arm CMSIS-NN provides established integer softmax kernels, including [fixed-point exponential/reciprocal operations](https://github.com/ARM-software/CMSIS-NN/blob/main/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c) and [a lookup-based INT16 softmax](https://github.com/ARM-software/CMSIS-NN/blob/main/Source/SoftmaxFunctions/arm_softmax_s16.c). These are implementation references, not measured speedups for this CTF workload.

The sigmoid-ordering argument follows from its derivative. The ranking counterexample and Q24 arithmetic were checked numerically. No replacement kernel, target-host benchmark, training or match experiment was run for this report.
