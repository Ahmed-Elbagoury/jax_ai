import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    return input[: input.shape[0]//2] * jax.nn.gelu(input[input.shape[0]//2:], approximate = False)
