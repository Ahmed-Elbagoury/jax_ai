import jax
import jax.numpy as jnp
import math
from functools import partial

# G, Buf are tensors on device
@partial(jax.jit, static_argnums=(2, 3))
def solve(G: jax.Array, Buf: jax.Array, M: int, N: int, momentum: float) -> jax.Array:
    V = momentum * Buf + G
    transposed = M > N
    
    X = V / (jnp.linalg.norm(V, ord="fro") +1e-7)

    if transposed:
        X = X.T

    a, b, c = 3.4445, -4.775, 2.0315
    for i in range(5):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return jnp.sqrt(jnp.maximum(M, N)) * X
