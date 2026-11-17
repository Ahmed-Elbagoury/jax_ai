import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, lo: float, hi: float, N: int) -> jax.Array:
    # return output tensor directly
    # input = jnp.where(input < lo, lo, input)
    # return jnp.where(input > hi, hi, input)
    return jnp.maximum(jnp.minimum(input, hi), lo)
