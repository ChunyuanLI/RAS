import tensorflow as tf

from rllab.core.serializable import Serializable

from softqlearning.misc.nn import feedforward_net

from .nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self, env_spec, hidden_layer_sizes, squash=True):
        Serializable.quick_init(self, locals())

        self._action_dim = env_spec.action_space.flat_dim
        self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation',
        )

        self._actions = self.actions_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, 'policy')

    def actions_for(self,
                    observations,
                    n_action_samples=1,
                    reuse=tf.AUTO_REUSE):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope('policy', reuse=reuse):
            raw_actions, l1out = feedforward_net(
                observations,
                latents,
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None,
                is_policy_net=True,)
        
        
        return tf.tanh(raw_actions) if self._squash else raw_actions, l1out
