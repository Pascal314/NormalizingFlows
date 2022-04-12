import haiku as hk
import distrax
import numpy as np
import jax
import jax.numpy as jnp


class Invertable1x1Conv(distrax.Bijector):
    def __init__(self, event_ndims_in, n_channels, num_spatial_dims):
        super().__init__(event_ndims_in)
        self.conv = hk.ConvND(num_spatial_dims, n_channels, 1, with_bias=False)
        self.num_spatial_dims = num_spatial_dims
    
    def forward_and_log_det(self, x):
        y = self.conv(x)
        W = next(iter(self.conv.params_dict().values()))
        logdet = x.shape[1] * jnp.log(jnp.linalg.det(W))
        return y, logdet

    def inverse_and_log_det(self, y):
        W = next(iter(self.conv.params_dict().values()))
        W_inv = jnp.linalg.inv(W)

        # (0, n, n-1, ... 1,)
        input_axis_perm = (0, *range(self.num_spatial_dims + 1, 0, -1))
        # (n, n-1, n-2, ..., 0)
        filter_axis_perm = (*range(self.num_spatial_dims + 1, -1, -1),)

        window_strides = (1,) * self.num_spatial_dims
        y = jax.lax.conv(jnp.transpose(x, input_axis_perm), jnp.transpose(W_inv, filter_axis_perm) , window_strides, 'same')
        logdet = x.shape[1] * jnp.log(jnp.linalg.det(W_inv))
        return x, logdet


class ActnormModule(hk.Module):
    eps = 1e-8
    def __call__(self, x, reverse=False):
        # notation taken from GLOW
        b = hk.get_parameter('b', (1,) * (len(x.shape) - 1) + (x.shape[-1],), init=hk.initializers.Constant(0.))
        s = hk.get_parameter('s', (1,) * (len(x.shape) - 1) + (x.shape[-1],), init=hk.initializers.Constant(1.))
        logdet = jnp.sum(jnp.log(jnp.abs(s)))

        if not reverse:
            y = s * (x + b)
        else:
            y = x / (b + self.eps) - s
            logdet = -logdet
        return y, logdet

class ActNormBijector(distrax.Bijector):
    def __init__(self, event_ndims_in):
        super().__init__(event_ndims_in=event_ndims_in)
        self.actnorm = ActnormModule()
    
    def forward_and_log_det(self, x):
        return self.actnorm(x, reverse=False)

    def inverse_and_log_det(self, y):
        return self.actnorm(y, reverse=True)

def make_flowblock(input_shape, coupling_conditioner):
    event_ndims_in = len(input_shape) - 1
    num_spatial_dims = len(input_shape) - 2

    actnorm = ActNormBijector(event_ndims_in)
    conv = Invertable1x1Conv(event_ndims_in, input_shape[-1], num_spatial_dims)

    split_index = input_shape[-1] // 2
    bijector = lambda params: distrax.ScalarAffine(params['shift'], params['scale'])

    def conditioner(x):
        output = coupling_conditioner(x)
        return {'shift': output[..., 0], 'scale': output[..., 1]}

    coupler = distrax.SplitCoupling(
        split_index=split_index, 
        event_ndims=event_ndims_in, 
        conditioner=conditioner, 
        bijector=bijector)

    return distrax.Chain([actnorm, conv, coupler])


if __name__ == "__main__":
    shape = (1, 10, 10, 4)
    x= np.random.normal(size=(1, 10, 10, 4))
    
    def coupler(x):
        net = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP( (32, 32, 2 // 2 * np.prod(shape[1:])) ),
            hk.Reshape( (*shape[1:-1], 2, 2))
        ])
        return net(x)

    def forward(x):
        conv = make_flowblock(shape, coupler)
        return conv.forward_and_log_det(x)

    def backward(x):
        conv = make_flowblock(shape, coupler)
        return conv.inverse_and_log_det(x)

    fwd = hk.without_apply_rng(hk.transform(forward))
    params = fwd.init(jax.random.PRNGKey(42), x)
    y, logdet = fwd.apply(params, x)


    bwd = hk.without_apply_rng(hk.transform(backward))
    inv, inv_logdet = bwd.apply(params, y)

    assert np.allclose(x - inv, 0, atol=1e-5), (x - inv)
    assert np.allclose(logdet + inv_logdet, 0, atol=1e-5), (logdet + inv_logdet)