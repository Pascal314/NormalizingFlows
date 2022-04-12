import glow
import numpy as np
import haiku as hk
import jax

x= np.random.normal(size=(1, 10, 10, 4))
def forward(x):
    conv = glow.Invertable1x1Conv(1, 4, 2)
    return conv.forward_and_log_det(x)

def backward(x):
    conv = glow.Invertable1x1Conv(1, 4, 2)
    return conv.inverse_and_log_det(x)

fwd = hk.without_apply_rng(hk.transform(forward))
params = fwd.init(jax.random.PRNGKey(42), x)
y, logdet = fwd.apply(params, x)


bwd = hk.without_apply_rng(hk.transform(backward))
inv, inv_logdet = bwd.apply(params, y)

assert np.allclose(x - inv, 0, atol=1e-5)
assert np.allclose(logdet + inv_logdet, 0, atol=1e-5), (logdet + inv_logdet)

x= np.random.normal(size=(1, 10, 4))
def forward2(x):
    conv = glow.Invertable1x1Conv(1, 4, 1)
    return conv.forward_and_log_det(x)

def backward2(x):
    conv = glow.Invertable1x1Conv(1, 4, 1)
    return conv.inverse_and_log_det(x)

fwd = hk.without_apply_rng(hk.transform(forward2))
params = fwd.init(jax.random.PRNGKey(42), x)
y, logdet = fwd.apply(params, x)


bwd = hk.without_apply_rng(hk.transform(backward2))
inv, inv_logdet = bwd.apply(params, y)

assert np.allclose(x - inv, 0, atol=1e-5)
assert np.allclose(logdet + inv_logdet, 0, atol=1e-5), (logdet + inv_logdet)