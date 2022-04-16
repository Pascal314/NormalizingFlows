import glow
import numpy as np
import haiku as hk
import jax
import chex
import distrax
from absl.testing import parameterized, absltest


class ConvTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1d_n1', (1, 10, 4), 1),
        ('case_2d_n1', (2, 10, 10, 4), 1),
        ('case_1d_n5', (2, 10, 4), 5),
        ('case_2d_n5', (1, 10, 10, 4), 5),
    )
    def test(self, shape, n):
        x= np.random.normal(size=shape)
        event_ndims_in = len(shape) - 1
        spatial_dims = len(shape) - 2
        def forward(x):
            convs = []
            for i in range(n):
                conv = glow.Invertable1x1Conv(event_ndims_in, 4, spatial_dims)
                convs.append(conv)
            return distrax.Chain(convs).forward_and_log_det(x)

        def backward(x):
            convs = []
            for i in range(n):
                conv = glow.Invertable1x1Conv(event_ndims_in, 4, spatial_dims)
                convs.append(conv)
            return distrax.Chain(convs).inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)

        # careful, as these convs are initialized with orthogonal matrices
        # A better test would override this initialization with something else
        # to test for non-orthogonal determinant calculations (i.e. det != 0)
        assert np.allclose(x, inv, atol=1e-6 ), np.max(np.abs((x - inv)))
        assert np.allclose(logdet + inv_logdet, 0, atol=1e-6 * np.prod(shape[1:-1])), (logdet, inv_logdet)
        assert logdet.shape == shape[0:1], logdet.shape

class ActNormTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1d', (5, 5), 1),
        ('case_2d', (1, 10, 10), 1),
        ('case_8d', (1, 5, 7, 5, 7, 5, 7, 5, 7), 1),
        ('case_8d_n9', (1, 5, 7, 5, 7, 5, 7, 5, 7), 9),

    )
    def test(self, shape, n):
        x = np.random.normal(size=shape)
        event_ndims_in = len(shape) - 1
        
        def forward(x):
            blocks = []
            for i in range(n):
                actnorm = glow.ActNormBijector(event_ndims_in)
                blocks.append(actnorm)
            return distrax.Chain(blocks).forward_and_log_det(x)
        
        def backward(x):
            blocks = []
            for i in range(n):
                actnorm = glow.ActNormBijector(event_ndims_in)
                blocks.append(actnorm)
            return distrax.Chain(blocks).inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)  

        assert logdet.shape == shape[0:1], logdet.shape
        assert np.allclose(x, inv, atol=1e-6), np.max(np.abs(x - inv))
        assert np.allclose(logdet, -inv_logdet, atol=1e-6), (logdet, inv_logdet)
        mean, std = np.mean(y), np.std(y)
        assert np.allclose([mean, std], [0., 1.], atol=1e-6), (mean, std, np.mean(x), np.std(x))

class FlowTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1', 1),
        ('case_2', 2),
        ('case_3', 3),
        ('case_4', 4),
        ('case_5', 5),
        ('case_6', 6),
        ('case_7', 7),

    )
    def test(self, n):
        shape = (1, 10, 10, 16)
        x= np.random.normal(size=shape)
        
        def make_conditioner():
            net = hk.Sequential([
                hk.Flatten(),
                hk.nets.MLP( (32, 2 // 2 * np.prod(shape[1:]))),
                hk.Reshape( (*shape[1:-1], shape[-1] // 2, 2))
            ])
            return net

        def forward(x):
            blocks = []
            for i in range(n):
                coupler = make_conditioner()
                block = glow.make_flowblock(shape, coupler)
                blocks.append(block)
            return distrax.Chain(blocks).forward_and_log_det(x)

        def backward(x):
            blocks = []
            for i in range(n):
                coupler = make_conditioner()
                block = glow.make_flowblock(shape, coupler)
                blocks.append(block)
            return distrax.Chain(blocks).inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        # print(jax.tree_map(lambda p: p.shape, params))
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)

        assert logdet.shape == shape[0:1], logdet.shape
        assert np.allclose(x, inv, atol=1e-5), np.max(np.abs((x - inv)))
        assert np.allclose(logdet, -inv_logdet, atol=1e-6 * np.prod(shape[1:-1])), (logdet, inv_logdet)


class CouplerTest(chex.TestCase):
    @parameterized.named_parameters(
        ('case_1', 1),
        ('case_2', 2),
        ('case_5', 5),
        ('case_9', 9),
    )
    def test(self, n):
        shape = (1, 10, 10, 4)
        x= np.random.normal(size=(1, 10, 10, 4))
        split_index = x.shape[-1] // 2
        event_ndims_in = len(shape) - 1

        
        def make_conditioner():
            net = hk.Sequential([
                hk.Flatten(),
                hk.nets.MLP( (32, 2 // 2 * np.prod(shape[1:])) ),
                hk.Reshape( (*shape[1:-1], 2, 2))
            ])
            return net

        def forward(x):
            blocks = []
            for i in range(n):
                conditioner = make_conditioner()
                coupler = glow.make_affine_coupler(split_index, event_ndims_in, conditioner)
                blocks.append(coupler)
            return distrax.Chain(blocks).forward_and_log_det(x)

        def backward(x):
            blocks = []
            for i in range(n):
                conditioner = make_conditioner()
                coupler = glow.make_affine_coupler(split_index, event_ndims_in, conditioner)
                blocks.append(coupler)
            return distrax.Chain(blocks).inverse_and_log_det(x)

        fwd = hk.without_apply_rng(hk.transform(forward))
        params = fwd.init(jax.random.PRNGKey(42), x)
        y, logdet = fwd.apply(params, x)


        bwd = hk.without_apply_rng(hk.transform(backward))
        inv, inv_logdet = bwd.apply(params, y)
        
        assert logdet.shape == shape[0:1], logdet.shape
        assert np.allclose(x, inv, rtol=1e-4), np.max(np.abs((x - inv)))
        assert np.allclose(logdet, -inv_logdet), (logdet + inv_logdet, logdet, inv_logdet) 

if __name__ == "__main__":
    absltest.main()