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
    top_k = np.partition(logits, -k, axis=-1)[..., -k, None]
    return np.where(logits >= top_k, logits, -np.inf)

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    probs = stable_softmax(logits)
    indicies = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, indicies, axis=-1)

    cumsum = np.cumsum(sorted_probs, axis=-1)
    remove = cumsum > p
    remove[...,1:] = remove[...,:-1]
    remove[...,0] = False
    sorted_logits = np.take_along_axis(logits, indicies, axis=-1)
    sorted_logits = np.where(remove, -np.inf, sorted_logits)

    out = np.empty_like(logits)
    # np.put_along_axis(out, indices, sorted_logits, axis=-1) takes values from sorted_logits and 
    # writes them into out at positions specified by indices along the last axis.
    np.put_along_axis(out, indicies, sorted_logits, axis=-1)
    return out

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    return int(rng.choice(len(probs), p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    return np.argmax(logits)

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    id_to_token = special_tokens.copy()
    unique_chars = set()
    for text in corpus:
        for c in text:
            unique_chars.add(c)
    ordered_chars = sorted(unique_chars)
    id_to_token.extend(ordered_chars)
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token
    }

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    ids = []
    if add_bos and '<bos>' in vocab['token_to_id']:
        ids.append(0)
    for char in text:
        if char in vocab['token_to_id']:
            ids.append(vocab['token_to_id'][char])
        else:
            ids.append(2)
    return ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    chars = []
    for token_id in token_ids:
        if 0 <= token_id <=1:
            if not skip_special:
                chars.append(vocab['id_to_token'][token_id])
        else:
            chars.append(vocab['id_to_token'][token_id])
    return ''.join(chars)

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    return embedding_matrix[token_ids]

