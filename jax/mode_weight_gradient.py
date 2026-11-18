import jax
import jax.numpy as jnp


# X, dY, group_offsets are tensors on device
@jax.jit
def solve(
    X: jax.Array,
    dY: jax.Array,
    group_offsets: jax.Array,
    G: int,
    M_total: int,
    K: int,
    N: int,
) -> jax.Array:
    # return output tensor directly
    dw = jnp.zeros((group_offsets.shape[0]-1, X.shape[1], dY.shape[1]))
    num_rows = dY.shape[0]
    for i in range(group_offsets.shape[0]-1):
        mask = (jnp.arange(num_rows) >= group_offsets[i]) & (jnp.arange(num_rows) < group_offsets[i+1])
        dY_masked = jnp.where(mask[:, None], dY, 0)
        mult = X.T @ dY_masked
        dw = dw.at[i].set(X.T @ dY_masked)
    return dw
