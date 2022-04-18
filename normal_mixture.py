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
    n, k = (100, 2)
    shape = (n, 1, k)
    data = 0.1 * np.random.normal(size=(n,))
    data += np.random.choice([-0.5, 0.5], size=data.shape)
    # plt.hist(data, density=True, bins=20)
    # x = np.linspace(-2, 2, 100)
    # plt.plot(x, 0.5 * (norm(-.5, .1).pdf(x) + norm(.5, .1).pdf(x)))
    # plt.show()

    data= data.reshape(-1, 1)
    def make_coupler():
        net = hk.Sequential([
            hk.Flatten(),
            hk.nets.MLP( (32, 32), activate_final=True),
            hk.Linear( 2 // 2 * np.prod(shape[1:]), w_init=hk.initializers.Constant(0.05)),
            hk.Reshape( (*shape[1:-1], shape[-1] // 2, 2))
        ])
        return net

    def make_flow_model():
        blocks = []
        for i in range(8):
            coupler = make_coupler()
            block = glow.make_flowblock(shape, coupler)
            blocks.append(block)

        normal = distrax.Independent(
            distrax.Normal(jnp.zeros(shape[1:]) - 2, jnp.ones(shape[1:])),
            reinterpreted_batch_ndims=2
        )

        # blocks.append(distrax.Block(distrax.Sigmoid(), 2))
        flow = distrax.Inverse(distrax.Chain(blocks[::-1]))
        return distrax.Transformed(normal, flow), flow
        # return distrax.Transformed(uniform, (distrax.Chain(blocks)))
        # return distrax.Chain(blocks[::-1])

    
    @hk.transform
    def log_prob(x):
        model, _ = make_flow_model()
        u = jax.random.uniform(hk.next_rng_key(), shape=(x.shape[0], 1, k-1))
        u = jax.random.normal(hk.next_rng_key(), shape=(x.shape[0], 1, k-1))
        x = jnp.concatenate([x[..., None], u], axis=-1)
        return model.log_prob(x) + distrax.Independent(distrax.Normal(jnp.zeros(u.shape), jnp.ones(u.shape)), 2).log_prob(u)

    def loss_fn(params, key):
        loss = -jnp.mean(log_prob.apply(params, key, data))
        return loss

    opt = optax.adam(3e-4)

    @jax.jit
    def update(params, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, 2)
    params = log_prob.init(subkey, data)
    print(jax.tree_map(lambda x: x.shape, params))

    key, subkey = jax.random.split(key, 2)

    print(log_prob.apply(params, subkey, data))
    key, subkey = jax.random.split(key, 2)

    print(loss_fn(params, subkey))

    opt_state = opt.init(params)

    for i in range(10000):
        key, subkey = jax.random.split(key, 2)

        loss, params, opt_state = update(params, opt_state, subkey)
        if i % 1000 == 0:
            print(loss)

        
        if i % 3000 == 0:
            x = np.linspace(-200, 200, 10000).reshape(-1, 1)
            key, subkey = jax.random.split(key, 2)
            approximate_pdf = log_prob.apply(params, subkey, x)
            I1 = integrate.simpson(np.exp(approximate_pdf), x.flatten())
            print(np.log(I1), I1)
    
    x = np.linspace(-20, 20, 10000).reshape(-1, 1)
    key, subkey = jax.random.split(key, 2)
    approximate_pdf = log_prob.apply(params, subkey, x)
    plt.plot(x, 0.5 * (norm(-.5, .1).pdf(x) + norm(.5, .1).pdf(x)))
    
    I1 = integrate.simpson(np.exp(approximate_pdf), x.flatten())
    print(np.log(I1))
    print(I1)

    I2 = integrate.simpson( (0.5 * (norm(-.5, .1).pdf(x) + norm(.5, .1).pdf(x)) ).flatten(), x.flatten())
    print(np.log(I2))
    print(I2)
 

    @hk.transform
    def round_trip(x):
        model, bijection = make_flow_model()
        # u = jax.random.uniform(hk.next_rng_key(), shape=(x.shape[0], 1, k-1))
        u = jax.random.normal(hk.next_rng_key(), shape=(x.shape[0], 1, k-1))

        x = jnp.concatenate([x[..., None], u], axis=-1)
        y, det = bijection.inverse_and_log_det(x) 
        inv, inv_det = bijection.forward_and_log_det(y)
        return y, det, inv, inv_det, model.log_prob(x)

    key, subkey = jax.random.split(key, 2)
    y, det, inv, inv_det, prob = round_trip.apply(params, subkey, x)
    print(x)
    print(y[:, 0, 0])
    print(inv[:, 0, 0])
    print(det)

    print(inv_det)
    print(det + inv_det)
    print(prob)

    # print(params)
    plt.plot(x, np.exp(approximate_pdf))
    plt.show()
    print(approximate_pdf)

    # # @hk.without_apply_rng
    # @hk.transform
    # def forward(x):
    #     model = make_flow_model()
    #     x = jnp.stack([x] * k, axis=-1)
    #     return model.forward_and_log_det(x)

    # params = forward.init(jax.random.PRNGKey(42), data)
    # print(jax.tree_map(lambda x: x.shape, params))
    # print(forward.apply(params, data))
    