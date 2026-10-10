# Adaptive policy compute under a hard CPU deadline

Research checked on October 8, 2026. This document proposes experiments; it does not claim a measured CTF speedup or authorize a model deployment. No training or runtime benchmarks were executed for this research.

The strongest applicable idea is **selective spending of compute**, rather than a universally larger P network. Start with independently trained small/base/large models, select a default using spawn distance, and override that choice using current tactical relevance and a shared turn budget. A shared-width network is a worthwhile second experiment. Do not assume that image-classification speedups transfer to tiny quantized MLPs.

## Primary evidence and its limits

The six sources below demonstrate different kinds of efficiency. Reported latency, arithmetic counts, and amortized training cost must remain separate.

### 1. Learned selection between existing networks

Bolukbasi et al., *Adaptive Neural Networks for Efficient Inference*, ICML 2017, learn early-exit and network-selection policies using a cost/accuracy objective. Their ImageNet AlexNet/GoogLeNet/ResNet50 selection system reports **2.8× speedup over ResNet50 within one percentage point of top-5 accuracy**. Timing uses Caffe, Titan X Pascal and CuDNN 5, includes policy overhead, and excludes loading/preprocessing. This is GPU evidence for a cascade of substantially different CNNs, not a CPU guarantee. The useful analogy is learning whether an expensive prediction is worth its extra cost instead of using a confidence threshold alone. [Paper, sections 5–5.1](https://proceedings.mlr.press/v70/bolukbasi17a/bolukbasi17a.pdf).

### 2. Early exits with measured CPU gains

Teerapittayanon et al., *BranchyNet*, ICPR 2016, attach intermediate classifiers and exit according to prediction entropy. With batch size one on a **3.0 GHz CPU with 20 MB L3**, CIFAR-10 AlexNet changes from **9.56 to 6.32 ms**, with accuracy **78.38% to 79.19%**; MNIST LeNet changes from **3.37 to 0.62 ms**, with accuracy **99.20% to 99.25%**. The CPU ResNet point is faster but loses 1.53 accuracy points, illustrating that some operating points buy speed through degradation. The authors also examine cache misses. Our shallow MLPs have much less depth to skip. [Paper, Table I and section V](https://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf), [author implementation](https://github.com/kunglab/branchynet).

### 3. One budget shared across many inputs

Huang et al., *Multi-Scale Dense Networks for Resource Efficient Image Classification*, ICLR 2018, study anytime prediction and budgeted batch classification: spend unequal compute across easier and harder examples while reusing intermediate work. For ImageNet, their dynamic evaluation uses **about 2–3× fewer multiply-adds than DenseNets at comparable accuracy**; at approximately 1.7 billion operations it reaches approximately 75% top-1, roughly six points above the plotted ResNet baseline. These are arithmetic-budget comparisons, not measured CPU latency. ImageNet batches contain 128 images. The scheduling analogy is useful for several opponent teams competing for one turn budget. [Paper, section 5.2/Figure 7](https://arxiv.org/pdf/1703.09844), [author implementation](https://github.com/gaohuang/MSDNet).

### 4. Shared weights across widths

Yu and Huang, *Universally Slimmable Networks*, ICCV 2019, train a network executable at different widths using the sandwich rule and in-place distillation. ImageNet MobileNet-v1 at width 0.25 uses **41M operations and obtains 55.7% top-1**, versus 49.8% for the individually trained baseline; width 1.0 uses **569M operations and obtains 71.8%**, versus 70.9%. MobileNet-v2 is not uniformly better at every width. These comparisons do not measure our CPU latency. The paper also demonstrates width sharing in Atari actor-critic training without batch normalization. Reuse the training ideas, but first verify whether shared capacity hurts CTF calibration or prediction. [Paper, Table 1 and section 5.1](https://arxiv.org/pdf/1903.05134), [author implementation](https://github.com/JiahuiYu/slimmable_networks).

### 5. Hardware-aware specialization

Cai et al., *Once-for-All*, ICLR 2020, progressively shrink a supernetwork and specialize subnets using hardware costs. A Google Pixel1 configuration obtains **80.1% ImageNet top-1 at 143 ms**, reported **2.6× faster than EfficientNet-B2 with 0.3 points higher accuracy**. Phone timing uses TF-Lite, batch size one. The full training takes about **1,200 V100 GPU-hours**, amortized over deployment scenarios. This supports measuring the actual hardware frontier; it does not justify that investment for three tiny MLP tiers. Mobile-specific model selection is relevant later. [Paper, sections 4.1–4.2/Figure 9](https://arxiv.org/pdf/1908.09791), [author implementation](https://github.com/mit-han-lab/once-for-all).

### 6. Dynamic skipping and router overhead

Wang et al., *SkipNet*, ECCV 2018, learn per-input residual-block gates. Their ImageNet experiments report **12–30% arithmetic reduction** with accuracy matched to the full networks; the much larger savings on CIFAR/SVHN should not be described as ImageNet results. Counts include gate computation within their stated convolution multiply-add accounting. Figure 4 includes cases where a feed-forward gate increases cost. This is a useful warning: routing can consume the work it saves, especially in small networks. SkipNet's recurrent gates and reinforcement-learning machinery are unnecessary for the first CTF router. [Paper, sections 4.1–4.3](https://arxiv.org/pdf/1711.09485), [author implementation](https://github.com/ucbdrive/skipnet).

## Two-layer selection rule

These are **two decision rules**, not two neural layers, and they do not increase neural mixture alpha merely because an opponent is close.

**Layer 1: static distance prior.** Cache wall-aware initial spawn-to-spawn distances. Rank opponent teams deterministically: nearby teams default to large, middle teams to base, remote teams to small. In a duel, the sole opponent defaults to large. This preserves the requested proximity preference without rerunning geometry. Spawn distance estimates likely interaction, not current danger.

**Layer 2: dynamic relevance and admission.** Current contact, a vulnerable carrier, imminent scoring, or a contested flag can promote any team. Rank requested upgrades by expected decision improvement per additional measured cost, with urgent tactical events taking priority. Downgrade less consequential teams when the shared budget is insufficient. Add hysteresis for ordinary transitions; urgent events bypass it. Always retain a legal action and a valid lower-cost prediction fallback.

Entropy can be one router feature, but should not be its sole target. A distant opponent with random movement may have high irreducible entropy yet almost no effect on our best action. A predictable nearby carrier may have low entropy but a large consequence if predicted incorrectly. More capacity cannot reveal information absent from the observation. Evaluate upgrades using held-out labels and downstream decision changes, rather than assuming large always helps.

An optional cheap-small-first cascade must be compared with direct selection. For independent models, a promoted query costs **small plus large plus routing**, not just large. Immediate threats can go directly to large. If expensive queries are common, small-first can be strictly slower than direct large even when the small model is very cheap.

## Mathematical allocation proposal

For opponent team i and tier k, estimate cost c(i,k) and expected reduction in decision regret g(i,k), relative to an approved fallback. Select one tier per team to maximize

$$
\sum_i g(i,k_i),\qquad \sum_i c(i,k_i)+c_{router}\leq B_P.
$$

This is a proposed multiple-choice budget allocation, not a theorem or reported result from these papers. The additive benefit is an approximation: coordinated enemy threats and changes in our chosen action can couple upgrades. With at most 14 opponent teams and three tiers, a small deterministic dynamic program is feasible, but its bookkeeping must be measured. A greedy upgrade ratio is a cheaper approximation; it need not be globally optimal. Estimate benefits on tuning data using tactical features, then freeze and confirm the entire router on independent games.

For intuition, in a fixed one-step simulator with utility in [0,1], changing an enemy distribution from p to q changes any action's expected utility by at most TV(p,q). Choosing with q instead of p therefore incurs at most **2 TV(p,q)** regret. This bound follows from the definition of total variation and two action-value errors. It is loose and does not say a large model is closer to p. It explains why probability accuracy matters most where competing own actions have small value margins. Longer-horizon rollout utility requires its own range and model-error accounting.

Do not label teacher-based regret as ground truth. A fixed high-budget reference can rank experimental routers, but its mistakes remain. Confirm actual strength with paired, independent games using the evaluation package.

## Independent tiers versus shared or nested networks

Independent tiers are the first reference: simple exports, separate calibration, and clean measurements. Their extra weight memory is acceptable here. Distill the large model's full joint and conditional distributions into smaller students while retaining supervised action labels; confirm that teacher errors are not merely copied.

A shared-width network reduces duplicated storage and may reduce training duplication. Train all deployed widths deliberately, including smallest and largest. Preserve 125 joint moves and identical action/target semantics at every width. Smaller hidden layers do not shrink the fixed output space: conditional heads, masking, softmax and sampling may dominate, so scaling hidden widths by four does not guarantee a sixteenfold end-to-end saving.

Width sharing is **not automatic incremental continuation**. In an ordinary dense MLP, wider first-layer activations change inputs to later layers, including their narrow prefix. Cached narrow second-layer activations usually cannot be reused unchanged. Exact progressive reuse needs an explicitly nested dependency structure or residual refinement design, with its own training and strength tradeoff. Prefer depth exits on a genuinely shared prefix if continuation is required; do not retrofit them without measurements.

Each tier or exit needs a validated integer export, its own temperature/alpha calibration and a gate for the deployed exposure. Quantization scales can differ; splicing hidden activations across incompatible scales is invalid. The scheduler, weights, feature schema and joint-v2 sampler together form the artifact. Calibrate the actual integer adapter, including joint correlations, rather than a floating-point surrogate or independent marginals.

## Memory should eliminate repeated work

Precompute static distance/attack geometry within the two-second startup contract or use a bounded on-demand fallback. Build one turn representation, then share geometry and stable primitives between opponent encoders. Preallocate aligned model matrices, features, outputs and sampling buffers.

Reuse predictions exactly when full encoded input, model identity, conditioning context and legal masks match. Confirm full keys after hashing. Keep bounded caches and measure hits, lookup cost and evictions. Branch states that change scores, carriers, immunity or targets invalidate affected entries. Reuse fixed root CDFs across rollout samples; do not silently reuse stale root predictions as fresh branch predictions.

First-layer contribution caches help only if sufficiently many expensive contributions repeat. Weight-plus-feature lookup tables may exceed cache and lose to SIMD multiplication. Unlimited RAM increases possible storage, not bandwidth or locality. Measure cold and warm working sets; use separate native and browser memory profiles.

## Deadline and experiment order

Set B_P only after measuring complete-turn costs on the target CPU. Include feature extraction, heads, calibration, dispatch, cache work and output. Admit work using conservative measured costs and bounded interruption points; keep time for search and flushing. Tail measurements do not prove a hard worst-case bound, so the deadline guard remains necessary. First-turn initialization must also fit two seconds. GPU training does not relax either CPU deadline.

1. Train independently quantized small/base/large general P, then representative n02/n05/n15 tiers. Measure each tier's prediction and complete cost on the same workload.
2. Compare fixed-small, fixed-large, distance-only and distance-plus-dynamic routing with equal search work. Test far-born urgent threats explicitly.
3. Add exact geometry/input/sampler caches; verify parity before speed comparisons. Then compare useful search work under the real deadline.
4. Test small-first upgrading and supervised benefit routing. Retain simple routing unless complexity improves confirmed regret/strength per millisecond.
5. Compare a slimmable/distilled model only after the independent-tier baseline is stable. Expand count coverage only for useful tiers.

CUDA can accelerate offline batched model training and teacher generation; it should remain optional. Benchmark examples/second including transfers and data preparation against CPU training, since these MLPs can be launch-bound. Native inference and eventual WASM use the same CPU integer contract. Release criteria include p99.9/max complete-turn latency, deadline/fault counts, tier exposure, cache effectiveness, useful search scenarios and paired strength intervals. None of the cited speedups substitutes for those measurements.
