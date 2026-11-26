import jax
import jax.numpy as jnp

def masked_mean(x: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """
    Returns a float32 array with the shape of x without the given axis, holding the mean of the valid entries.
    """
    valid_sum = jnp.sum(x, where=mask, axis= axis)
    count = jnp.sum(mask, axis=axis)
    return jnp.where(count != 0, valid_sum / count, 0)
    
