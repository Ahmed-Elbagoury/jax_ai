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
    # dw = jnp.zeros((group_offsets.shape[0]-1, X.shape[1], dY.shape[1]))
    # num_rows = dY.shape[0]
    # for i in range(group_offsets.shape[0]-1):
    #     mask = (jnp.arange(num_rows) >= group_offsets[i]) & (jnp.arange(num_rows) < group_offsets[i+1])
    #     dY_masked = jnp.where(mask[:, None], dY, 0)
    #     mult = X.T @ dY_masked
    #     dw = dw.at[i].set(X.T @ dY_masked)
    # return dw
    output = jnp.zeros((group_offsets.shape[0]-1, X.shape[1], dY.shape[1]))
    def one_expert(i, output):
        y_mask = dY
        rows = jnp.arange(dY.shape[0])
        cols = jnp.arange(dY.shape[1])
        row_mask = (rows >= group_offsets[i]) & (rows < group_offsets[i+1])
        mask_2d = row_mask[:, None]
        y_mask = jnp.where(mask_2d, y_mask, 0)
        return output.at[i].set(X.T @ y_mask)
    return jax.lax.fori_loop(0, G, one_expert, output)
