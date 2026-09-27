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

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    if temperature <= 0.0:
        return logits
    return logits/temperature

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    V = logits.shape[-1]
    if k >= V:
        return logits
    top_k = np.partition(logits, -k, axis=-1)
    return np.where(logits >= top_k[...,-k, None], logits, -np.inf)

