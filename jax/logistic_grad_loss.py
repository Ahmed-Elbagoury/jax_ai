import jax
import jax.numpy as jnp

def logistic_value_and_grad(params, x, y, l2):
    """
    Returns: dict with the scalar loss and a grads dict shaped like params
    """
    def loss_fn(p):
        logits = x @ p["w"] + p["b"]
        data_loss = jnp.mean(jax.nn.softplus(logits) - y * logits)
        return data_loss + 0.5 * l2 * jnp.sum(p["w"] ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return {"loss": loss, "grads": grads}
