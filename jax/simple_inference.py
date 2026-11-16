import jax
import jax.numpy as jnp


# input and model are on device
def solve(input: jax.Array, model) -> jax.Array:
    # return output tensor directly
    x = input @ jnp.asarray(model.weight.T.detach().numpy())
    if model.bias is None:
        return x
    return x + jnp.array(model.bias.detach().numpy())
