"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(logits):
    max_val = np.max(logits, axis = -1, keepdims=True)
    reduced_logits = logits - max_val
    exp_logits = np.exp(reduced_logits)
    return exp_logits/np.sum(exp_logits, axis=-1, keepdims=True)

