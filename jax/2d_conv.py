import jax
import jax.numpy as jnp
from functools import partial

# input, kernel are tensors on device
@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def solve(
    input: jax.Array,
    kernel: jax.Array,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
) -> jax.Array:
    kernel = kernel.reshape(kernel_rows, kernel_cols, 1, 1)
    input = input.reshape(1, input_rows, input_cols, 1)
    out = jax.lax.conv_general_dilated(input, kernel,
                               window_strides=(1,1),
                               padding='VALID', 
                               dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return out.reshape(-1)
