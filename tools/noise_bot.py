"""Random-movement noise opponent for data collection (pool entry "Noise").

Every unit takes a uniformly random open move; it picks up a flag it steps on,
sometimes drops one, and shoots a random enemy within reach. Its seats add
unpredictable movement to everyone else's states. Its own requests are not
opponent-policy labels (pool.json: "policy_labels": false); value training
uses its rows like any other.
"""
from random_bot import main

if __name__ == "__main__":
    main(greed=0.0)
