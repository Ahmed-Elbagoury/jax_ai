import jax
import jax.numpy as jnp

"""
Add a batch of weighted updates into a histogram and return the new histogram. JAX arrays cannot be changed in place, so the result has to be built as a new array.

counts holds the starting value of each of the K bins. Update number n adds weights[n] to the bin whose number is indices[n], so for every bin k
"""

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
