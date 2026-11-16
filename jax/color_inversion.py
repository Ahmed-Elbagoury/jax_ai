import jax
import jax.numpy as jnp


# image is a tensor on device
@jax.jit
def solve(image: jax.Array, width: int, height: int) -> jax.Array:
    # return output tensor directly
    # [255, 255, 255, 0, 255, 255, 255, 0 ....]
    x = image.reshape(-1, 4)
    return x.at[:, 0:3].set(255-x[:, 0:3]).reshape(-1)
