import haiku as hk
import distrax
import numpy as np
import jax
import jax.numpy as jnp

class RandomRotationMatrix(hk.initializers.Initializer):
    def __call__(self, shape, dtype):
        # Assumes the final 2 dimensions are the (in_channels, out_channels) dims
        perm = jax.random.permutation(hk.next_rng_key(), shape[-1])
        W = jnp.zeros((shape[-1], shape[-1]), dtype=dtype)
        W = W.at[jnp.arange(shape[-1]), perm].set(1.)
        return jnp.broadcast_to(W, shape)

class Invertable1x1Conv(distrax.Bijector):
    def __init__(self, event_ndims_in, n_channels, num_spatial_dims):
        super().__init__(event_ndims_in)
        self.conv = hk.ConvND(num_spatial_dims, n_channels, 1, with_bias=False, w_init=hk.initializers.Orthogonal())
        self.num_spatial_dims = num_spatial_dims
    
    def forward_and_log_det(self, x):
        y = self.conv(x)
        W = next(iter(self.conv.params_dict().values()))
        logdet = jnp.log(jnp.abs(jnp.linalg.det(W)))
        return y, logdet

    def inverse_and_log_det(self, y):
        W = next(iter(self.conv.params_dict().values()))
        W_inv = jnp.linalg.inv(W)

        # Move channels to the front
        # (0, 3, 1, ..., n)
        input_axis_perm = (0, self.num_spatial_dims + 1, *range(1, self.num_spatial_dims + 1))
        # Move channels to the back
        # (0, 2, ..., n, 1)
        output_axis_perm = (0, *range(2, self.num_spatial_dims + 2), 1)
        # Move spatial dimensions to the back (is this really correct? seems like I am also reversing the batch/spatial dims)
        # (n, n-1, n-2, ..., 0)
        filter_axis_perm = (*range(self.num_spatial_dims + 1, -1, -1),)

        window_strides = (1,) * self.num_spatial_dims
        x = jax.lax.conv(jnp.transpose(y, input_axis_perm), jnp.transpose(W_inv, filter_axis_perm) , window_strides, 'same')
        logdet = jnp.log(jnp.abs(jnp.linalg.det(W_inv)))

        x = jnp.transpose(x, output_axis_perm)
        return x, logdet


class ActnormModule(hk.Module):
    eps = 1e-12
    def __call__(self, x, reverse=False):
        # notation taken from GLOW
        b = hk.get_parameter('b', (1,) * (len(x.shape) - 1) + (x.shape[-1],), init=hk.initializers.Constant(0.))
        s = hk.get_parameter('s', (1,) * (len(x.shape) - 1) + (x.shape[-1],), init=hk.initializers.Constant(1.))
        logdet = jnp.sum(jnp.log(jnp.abs(s)))

        if not reverse:
            y = s * x + b
        else:
            y = (x - b) / (s + self.eps)
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

def make_affine_coupler(split_index, event_ndims_in, coupling_conditioner):
    bijector = lambda params: distrax.ScalarAffine(params['shift'], params['scale'])

    def conditioner(x):
        output = coupling_conditioner(x)
        return {'shift': output[..., 0], 'scale': jax.nn.sigmoid(output[..., 1] + 2.)}

    coupler = distrax.SplitCoupling(
        split_index=split_index, 
        event_ndims=event_ndims_in, 
        conditioner=conditioner, 
        bijector=bijector)
    return coupler


def make_flowblock(input_shape, coupling_conditioner):
    event_ndims_in = len(input_shape) - 1
    num_spatial_dims = len(input_shape) - 2

    actnorm = ActNormBijector(event_ndims_in)
    conv = Invertable1x1Conv(event_ndims_in, input_shape[-1], num_spatial_dims)

    split_index = input_shape[-1] // 2
    bijector = lambda params: distrax.ScalarAffine(params['shift'], params['scale'])
    coupler = make_affine_coupler(split_index, event_ndims_in, coupling_conditioner)

    return distrax.Chain([actnorm, conv, coupler])
    # return distrax.Chain([actnorm, conv])
    # return distrax.Chain([actnorm, coupler])
    # return distrax.Chain([coupler])





if __name__ == "__main__":
    n = 1
    k = 10
    shape = (32, 10, 16)
    x= np.random.normal(size=shape)
    
    def make_coupler():
        net = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP( (32, 32), activate_final=True),
            hk.Linear( 2 // 2 * np.prod(shape[1:])),
            hk.Reshape( (*shape[1:-1], shape[-1] // 2, 2))
        ])
        return net

    def forward(x):
        blocks = []
        for i in range(n):
            coupler = make_coupler()
            block = make_flowblock(shape, coupler)
            blocks.append(block)
        return distrax.Chain(blocks).forward_and_log_det(x)

    def backward(x):
        blocks = []
        for i in range(n):
            coupler = make_coupler()
            block = make_flowblock(shape, coupler)
            blocks.append(block)
        return distrax.Chain(blocks).inverse_and_log_det(x)

    fwd = hk.without_apply_rng(hk.transform(forward))
    params = fwd.init(jax.random.PRNGKey(42), x)

    bwd = hk.without_apply_rng(hk.transform(backward))

    y = x
    logdet = 0
    for i in range(k):
        y, ld = fwd.apply(params, y)
        logdet += ld 

    inv = y
    for i in range(k):
        inv, ld = bwd.apply(params, inv)
        logdet += ld

    print(np.mean(x**2), np.mean(y**2), np.mean(inv**2), np.mean((x - inv)**2), np.mean(np.abs(logdet)))