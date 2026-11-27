import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnames=("L",))
def _pad_step(tokens, lengths, pad_id, L):
    start_indices = jnp.cumsum(lengths) - lengths # B x 1
    token_idx = start_indices[:, None] + jnp.arange(L) # B X 1 + 1 X L
    
    output = tokens[token_idx]
    
    mask = jnp.where(jnp.arange(L)[None, :] < lengths[:, None], True, False)
    
    return {"tokens": jnp.where(mask, output, pad_id), "mask": mask}

def pad_to_bucket(tokens: jax.Array, lengths: jax.Array, buckets: list[int], pad_id: int) -> dict:
    """
    Returns a dict with tokens, an int32 JAX array of shape (B, L), and mask, a bool JAX array of shape (B, L).
    """
    L = min(b for b in buckets if b >= int(jnp.max(lengths)))
    return _pad_step(tokens, lengths, pad_id, L)
