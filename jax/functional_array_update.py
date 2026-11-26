import jax
import jax.numpy as jnp

def weighted_histogram_vectorized(counts: jax.Array, indices: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Returns a new float32 array of shape (K,) holding counts plus every weight added into the bin named by its index, with out-of-range indices ignored.
    """
    valid = (indices >= 0) & (indices < counts.shape[0])
    weights = jnp.where(valid, weights, 0)
    return counts.at[indices].add(weights, mode="drop")

def weighted_histogram(counts: jax.Array, indices: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Returns a new float32 array of shape (K,) holding counts plus every weight added into the bin named by its index, with out-of-range indices ignored.
    """
    if indices.shape[0] == 0:
        return counts.copy()
    def add(i, output):
        idx = indices[i]
        valid = (idx >= 0) & (idx < output.shape[0])
        delta = jnp.where(valid, weights[i], 0.0)
        return output.at[indices[i]].add(delta)
    return jax.lax.fori_loop(0, indices.shape[0], add, counts)
    
