import jax
import haiku as hk
import jax.numpy as jnp 
import numpy as np
import distrax
import matplotlib.pyplot as plt

import haiku as hk
import glow
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import distrax
from scipy.stats import norm
import optax
from scipy import integrate

    


if __name__ == "__main__":
    n = 1000
    np.random.seed(0)
    y = np.random.normal(0, 3, size=(n, 1, 1))
    x = np.random.normal(0, np.minimum(np.exp(y / 2), 5))

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(x, y, alpha=0.1, label='data')
    shape = (n, 1, 2)
    data = np.concatenate([x, y], axis=-1)
    data = data.reshape(shape)
    
    def make_coupler():
        net = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP( (32, 32), activate_final=True),
            hk.Linear( 2 // 2 * np.prod(shape[1:]), w_init=hk.initializers.Constant(0.)),
            hk.Reshape( (*shape[1:-1], shape[-1] // 2, 2))
        ])
        return net

    def make_flow_model():
        blocks = []
        for i in range(4):
            coupler = make_coupler()
            block = glow.make_flowblock(shape, coupler)
            blocks.append(block)

        normal = distrax.Independent(
            distrax.Normal(jnp.zeros(shape[1:]), jnp.ones(shape[1:])),
            reinterpreted_batch_ndims=2
        )

        flow = distrax.Inverse(distrax.Chain(blocks[::-1]))
        return distrax.Transformed(normal, flow), flow

    @hk.without_apply_rng
    @hk.transform
    def log_prob(x):
        model, _ = make_flow_model()
        return model.log_prob(x)

    def loss_fn(params):
        loss = -jnp.mean(log_prob.apply(params, data))
        return loss

    opt = optax.adam(3e-4)

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    key = jax.random.PRNGKey(42)

    params = log_prob.init(key, data)
    print(sum(jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))))



    assert (log_prob.apply(params, data).shape) == (n,)


    print('loss:', loss_fn(params))

    opt_state = opt.init(params)

    for i in range(5000):
        loss, params, opt_state = update(params, opt_state)
        if i % 1000 == 0:
            print('loss:', loss)

    
    @hk.without_apply_rng
    @hk.transform
    def round_trip(x):
        model, bijection = make_flow_model()
        y, det = bijection.inverse_and_log_det(x) 
        inv, inv_det = bijection.forward_and_log_det(y)
        return y, det, inv, inv_det, model.log_prob(x)

    latent, det, inv, inv_det, prob = round_trip.apply(params, data)

    @hk.transform
    def sample(n):
        latent = jax.random.normal(hk.next_rng_key(), (n, 1, 2))
        model, bijection = make_flow_model()
        return bijection.forward(latent)
    
    samples = sample.apply(params, jax.random.PRNGKey(42), 1000)
    axes[0].set_title('samples')
    axes[0].scatter(samples[:, 0, 0], samples[:, 0, 1], alpha=0.1, label='sampled')
    axes[0].legend()
    axes[0].set_xlim((-10, 10))
    axes[0].set_ylim((-10, 10))
    
    axes[1].set_title('latent space')
    axes[1].scatter(latent[:, 0, 0], latent[:, 0, 1], label='latent', alpha=0.1)
    fig.set_size_inches(10, 4)
    plt.savefig('neals_funnel.pdf')
    plt.show()
