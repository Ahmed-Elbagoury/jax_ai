import jax
import jax.numpy as jnp


# A, B are tensors on device
@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    # total_len = A.shape[0] + B.shape[0]
    # output = jnp.zeros(total_len)
    # output = output.at[0:total_len:2].set(A)
    # return output.at[1:total_len:2].set(B)
    return jnp.stack([A, B], axis=1).reshape(-1)
