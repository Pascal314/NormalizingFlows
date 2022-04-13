import glow
import numpy as np
import haiku as hk
import jax
import chex
import distrax
from absl.testing import parameterized, absltest


class ConvTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1d', (1, 10, 4)),
        ('case_2d', (1, 10, 10, 4)),

    )
    def test(self, shape):
        x= np.random.normal(size=shape)
        event_ndims_in = len(shape) - 1
        spatial_dims = len(shape) - 2
        def forward(x):
            conv = glow.Invertable1x1Conv(event_ndims_in, 4, spatial_dims)
            return conv.forward_and_log_det(x)

        def backward(x):
            conv = glow.Invertable1x1Conv(event_ndims_in, 4, spatial_dims)
            return conv.inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)

        assert np.allclose(x, inv, atol=1e-8), (x - inv)
        assert np.allclose(logdet + inv_logdet, 0, atol=1e-8), (logdet, inv_logdet)

class ActNormTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1d', (5, 5)),
        ('case_2d', (1, 10, 10)),
        ('case_8d', (1, 5, 7, 5, 7, 5, 7, 5, 7)),
    )
    def test(self, shape):
        x = np.random.normal(size=shape)
        event_ndims_in = len(shape) - 1
        
        def forward(x):
            actnorm = glow.ActNormBijector(event_ndims_in)
            return actnorm.forward_and_log_det(x)
        
        def backward(x):
            actnorm = glow.ActNormBijector(event_ndims_in)
            return actnorm.inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)

        assert np.allclose(x, inv, atol=1e-8), np.max(np.abs(x - inv))
        assert np.allclose(logdet + inv_logdet, 0, atol=1e-8), (logdet, inv_logdet)

class FlowTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1', 1),
        ('case_3', 3),
        ('case_10', 10),

    )
    def test(self, n):
        shape = (1, 10, 10, 4)
        x= np.random.normal(size=(1, 10, 10, 4))
        
        def make_coupler():
            net = hk.Sequential([
                hk.Flatten(),
                hk.nets.MLP( (32, 32, 2 // 2 * np.prod(shape[1:])) ),
                hk.Reshape( (*shape[1:-1], 2, 2))
            ])
            return net

        def forward(x):
            blocks = []
            for i in range(n):
                coupler = make_coupler()
                block = glow.make_flowblock(shape, coupler)
                blocks.append(block)
            return distrax.Chain(blocks).forward_and_log_det(x)

        def backward(x):
            blocks = []
            for i in range(n):
                coupler = make_coupler()
                block = glow.make_flowblock(shape, coupler)
                blocks.append(block)
            return distrax.Chain(blocks).inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)

        # assert np.allclose(x, inv, rtol=1e-4), x - inv
        assert np.allclose(x, inv, rtol=1e-4), np.max(np.abs((x - inv)))

        assert np.allclose(logdet, inv_logdet, rtol=1e-4), (logdet + inv_logdet)

if __name__ == "__main__":
    absltest.main()