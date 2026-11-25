import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, N: int) -> jax.Array:
    input = input - jnp.max(input)
    input_exp = jnp.exp(input)
    return input_exp / jnp.sum(jnp.exp(input))
