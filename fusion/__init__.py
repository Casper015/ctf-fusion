"""ctf-fusion: opponent policy and board value models for the CTF arena bots.

Library layout:
    paths      locate this repository and the AI_Test checkout (engine, match runner, bots)
    native     compile and run the C++ tools (feature extractor, parity driver, legacy forecast dump)
    schema     extractor record layout and feature scales (read from the C++ encoder)
    matchlog   raw match files: write, read, rebuild any seat's input stream
    protocol   light parser for TURN blocks and reply lines (labels only; features come from C++)
    labels     requested-action labels, value targets, strata
    nn         numpy multilayer perceptrons, clipped ReLU, Adam
    models     policy and value networks over extractor records
    quant      integer export format and its bit-exact emulation of inference/mlp.hpp
    mixture    held-out mixture weight, temperature calibration, KL-regularized fusion
    metrics    log-loss, calibration, ranking accuracy
"""

SPEC_VERSION = 1
