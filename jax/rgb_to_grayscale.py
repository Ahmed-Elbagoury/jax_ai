import jax
import jax.numpy as jnp


# input is a tensor on device
@jax.jit
def solve(input: jax.Array, width: int, height: int) -> jax.Array:
    # return output tensor directly
    return 0.299 * input[0::3] + 0.587 * input[1::3] + 0.114 * input[2::3]
    # input = input.reshape(-1, 3) # N/3 x 3
    # weights = jnp.array([0.299, 0.587, 0.114])
    # result = input @ weights 
    # return result.reshape(-1)
