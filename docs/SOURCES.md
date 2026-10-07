# Sources

The design brief cites a `HEURISTIC_POLICY_SOURCES.md` that was not provided. These are the works it names, plus the techniques this repository implements.

| Topic | Reference | Used for |
|---|---|---|
| KL-regularized policy improvement (MPO E-step) | A. Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation*, ICLR 2018, arXiv:1806.06920 | `mixture.kl_fusion` closed form |
| Reference implementation | DeepMind RLax, `rlax/_src/mpo_ops.py`, https://github.com/google-deepmind/rlax | Same |
| Regularized MDPs | M. Geist, B. Scherrer, O. Pietquin, *A Theory of Regularized Markov Decision Processes*, ICML 2019, arXiv:1901.11275 | Background for the KL fusion |
| Safe policy improvement with a baseline | R. Laroche, P. Trichelair, R. Tachet des Combes, *Safe Policy Improvement with Baseline Bootstrapping*, ICML 2019, arXiv:1712.06924; code: https://github.com/RomainLaroche/SPIBB | Brief section 5; not implemented |
| Temperature scaling | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017, arXiv:1706.04599 | `mixture.fit_temperature` |
| Efficiently updatable networks (NNUE) | Y. Nasu, *Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi*, 2018 | Clipped ReLU, int16 weights, fixed-point activations (`quant.py`); the sparse accumulator itself is deferred ([REVIEW.md](REVIEW.md) 3.4) |
| Search genes already in AI_Test | Karnin, Koren, Somekh (ICML 2013); Danihelka et al. (ICLR 2022); Moraes and Lelis (AAAI 2018); Churchill and Buro (CIG 2013) | See AI_Test `nextgen/search/SEARCH_API.md` |
