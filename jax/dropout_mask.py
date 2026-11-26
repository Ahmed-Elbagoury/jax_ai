import jax
import jax.numpy as jnp

def dropout_masks(seed: int, num_layers: int, shape: list[int], keep_prob: float) -> jax.Array:
    """
    Returns a bool array of shape (num_layers, followed by the entries of shape) holding one dropout keep mask per layer.
    """
    output = jnp.zeros((num_layers, *shape), dtype=bool)
    key = jax.random.key(seed)
    for i in range(num_layers):
        key, subkey = jax.random.split(key)
        output = output.at[i].set(jax.random.bernoulli(subkey, p=keep_prob, shape=shape))
    return output
