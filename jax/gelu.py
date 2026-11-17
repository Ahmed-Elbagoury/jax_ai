import jax
import jax.numpy as jnp


# X is a tensor on device
@jax.jit
def solve(X: jax.Array, N: int) -> jax.Array:
    # return output tensor directly
    return X * (1 + jax.lax.erf(X / jnp.sqrt(2))) / 2
