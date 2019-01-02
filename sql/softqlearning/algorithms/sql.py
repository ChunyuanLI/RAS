import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
import tensorflow.contrib.distributions as ds
from .rl_algorithm import RLAlgorithm

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class SQL(RLAlgorithm, Serializable):
    """Soft Q-learning (SQL).

    Example:
        See `examples/mujo co_all_sql.py`.

    Reference:
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """
    
    def __init__(
            self,
            base_kwargs,
            env,
            pool,
            qf,
            policy,
            # add by me
            df,
            vf,
            kernel_n_particles,
            
            plotter=None,
            policy_lr=5E-4,
            qf_lr=5E-4,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            # kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=1,
            save_full_state=False,
            df_lr=5E-4,
            old_model=None,
            rf_lr=5E-4,
            vf_lr=5E-4,
            dist = 'beta'
    ):
        """
        Args:
            base_kwargs (dict): Dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`rllab.Env`): rllab environment object.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            qf (`NNQFunction`): Q-function approximator.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            qf_lr (`float`): Learning rate used for the Q-function approximator.
            value_n_particles (`int`): The number of action samples used for
                estimating the value of next state.
            td_target_update_interval (`int`): How often the target network is
                updated to match the current Q-function.
            kernel_fn (function object): A function object that represents
                a kernel function.
            kernel_n_particles (`int`): Total number of particles per state
                used in SVGD updates.
            kernel_update_ratio ('float'): The ratio of SVGD particles used for
                the computation of the inner/outer empirical expectation.
            discount ('float'): Discount factor.
            reward_scale ('float'): A factor that scales the raw rewards.
                Useful for adjusting the temperature of the optimal Boltzmann
                distribution.
            save_full_state ('boolean'): If true, saves the full algorithm
                state, including the replay buffer.
        """
        Serializable.quick_init(self, locals())
        super().__init__(**base_kwargs)
        
        self.env = env
        self.pool = pool
        self.qf = qf
        self.policy = policy
        self.plotter = plotter
        
        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._reward_scale = reward_scale
        
        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval
        
        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        
        self._save_full_state = save_full_state
        
        self._observation_dim = self.env.observation_space.flat_dim
        self._action_dim = self.env.action_space.flat_dim
        
        self._create_placeholders()
        
        self._training_ops = []
        self._target_ops = []

        self.dist = dist
        
        self._create_td_update()
        self._create_target_ops()
        
        # my own initialization
        self.df = df
        self.vf = vf
        # self._init_ref_dis()  # the reference distribution. (single gaussian)
        self.df_lr = df_lr
        self.vf_lr = vf_lr
        self._create_gd_update()
        self._sess.run(tf.global_variables_initializer())
    
    @overrides
    def train(self):
        """Start the Soft Q-Learning algorithm."""
        self._train(self.env, self.policy, self.pool)
    
    def _create_placeholders(self):
        """Create all necessary placeholders."""
        
        self._observations_ph = tf.placeholder(
                tf.float32,
                shape=[None, self._observation_dim],
                name='observations')
        
        self._next_observations_ph = tf.placeholder(
                tf.float32,
                shape=[None, self._observation_dim],
                name='next_observations')
        
        self._actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._action_dim], name='actions')
        
        self._next_actions_ph = tf.placeholder(
                tf.float32, shape=[None, self._action_dim], name='next_actions')
        
        self._rewards_pl = tf.placeholder(
                tf.float32, shape=[None], name='rewards')
        
        self._terminals_pl = tf.placeholder(
                tf.float32, shape=[None], name='terminals')
        
        self._df_lr_pl = tf.placeholder(
                tf.float32, shape=[]
        )
        self._policy_lr_pl = tf.placeholder(
                tf.float32, shape=[]
        )
    
    def _init_ref_dis(self):
        self.ref_dis = ds.Uniform(low=np.ones(self.env.action_dim) * -1, high=np.ones(self.env.action_dim) * 1)
    
    def gen_sample(self, n):
        return np.random.uniform(low=-1, high=1, size=n * self.env.action_dim * self._kernel_n_particles). \
            reshape(n, self._kernel_n_particles, self.env.action_dim).astype('float32')

    def tf_gen_sample(self, values,var):
        if(self.dist=='norm'):
            std = tf.sqrt(var+1e-4)
            self.ref_dis = ds.MultivariateNormalDiag(loc=values,scale_diag=std)
            self.ref_tile_dis = ds.MultivariateNormalDiag(loc=tf.tile(values[:, None, :], [1, self._kernel_n_particles, 1]),
                                                          scale_diag= tf.tile(std[:, None, :],
                                                                             [1, self._kernel_n_particles, 1]))
            return tf.transpose(self.ref_dis.sample(self._kernel_n_particles), [1, 0, 2])
        elif(self.dist=='beta'):
            var = var + 1e-4
            a = values * (values * (1 - values) / var - 1)
            b = (1 - values) * (values * (1 - values) / var - 1)
            masked = (var < values * (1 - values))
            a = tf.where(masked, a, a * 0 + 2)
            b = tf.where(masked, b, b * 0 + 2)
            self.a = a
            self.b = b
            self.values = values
            self.var = var
            self.ref_dis_beta = ds.Beta(a, b)
            self.ref_tile_dis_beta = ds.Beta(tf.tile(a[:, None, :], [1, self._kernel_n_particles, 1]),
                                             tf.tile(b[:, None, :], [1, self._kernel_n_particles, 1]))
            return (tf.transpose(self.ref_dis_beta.sample(self._kernel_n_particles), [1, 0, 2]) * 2 - 1) * 0.99
        else:
            print('methods has not been implemented')
            return None

    def _create_td_update(self):
        """Create a minimization operation for Q-function update."""
        
        with tf.variable_scope('target'):
            # The value of the next state is approximated with uniform samples.
            target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(
                    observations=self._next_observations_ph[:, None, :],
                    actions=target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])
        
        self._q_values = self.qf.output_for(
                self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(self._q_values, [None])
        
        # Equation 10:
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])
        
        # Importance weights add just a constant to the value.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)
        
        # \hat Q in Equation 11:
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])
        
        # Equation 11:
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values) ** 2)
        
        self.td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                loss=bellman_residual, var_list=self.qf.get_params_internal())
        
        self._training_ops.append(self.td_train_op)
        self._bellman_residual = bellman_residual

    def _create_gd_update(self):  # build a gan. update the discriminator and the generator.
        # generate the actions.
        actions, noise = self.policy.actions_for(
                observations=self._observations_ph,
                n_action_samples=self._kernel_n_particles)
        if (self._kernel_n_particles > 1):
            assert_shape(actions,
                         [None, self._kernel_n_particles, self._action_dim])
        else:
            assert_shape(actions,
                         [None, self._action_dim])
        
        # reference distribution
        tmp_action = tf.stop_gradient(actions)
        if(self.policy=='norm'):
            mean_action, var_action = tf.nn.moments(tmp_action,axes=1)
        elif(self.policy=='beta'):
            mean_action, var_action = tf.nn.moments(tmp_action / 2 + 0.5, axes=1)

        obs = self._observations_ph[:, None, :]
        obs = tf.tile(obs, [1, self._kernel_n_particles, 1])
        
        ref = self.tf_gen_sample(mean_action,var_action)

        # train the disciminator
        D_fake = self.df.output_for(
                tf.concat([actions, obs], axis=-1), reuse=True)
        D_ref = self.df.output_for(tf.concat([ref, obs], axis=-1), reuse=True)
        
        D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_ref, labels=tf.zeros_like(D_ref)))
        D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        
        if(self.dist=='norm'):
            ref_pro = self.ref_tile_dis.log_prob(actions)
        elif(self.dist=='beta'):
            ref_pro = tf.reduce_sum(self.ref_tile_dis_beta.log_prob(actions/2+0.5), axis=-1,keep_dims=False) - self._action_dim*tf.log(2.0)
        

        self.d_var_list = self.df.get_params_internal()
        real_pro = self.qf.output_for(obs, actions, reuse=True)
        self.loss_log_density = tf.reduce_mean(ref_pro - real_pro)
        
        self.D_loss = D_loss_real + D_loss_fake  # Discriminator
        self.D_fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        self.policy_loss = tf.reduce_mean(D_fake) + self.loss_log_density  # Generator
        self.D_training_op = tf.train.AdamOptimizer(self._df_lr_pl).minimize(self.D_loss,
                                                                             var_list=self.d_var_list)
        
        self.policy_training_op = tf.train.AdamOptimizer(self._policy_lr_pl).minimize(self.policy_loss,
                                                                                      var_list=self.policy.get_params_internal())
        self._training_ops.append(self.D_training_op)
        self._training_ops.append(self.policy_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')
        
        self._target_ops = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]
    
    @overrides
    def _init_training(self, env, policy, pool):
        super()._init_training(env, policy, pool)
        self._sess.run(self._target_ops)
    
    @overrides
    def _do_training(self, itr, batch):
        """Run the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(batch)
        self._sess.run([self.td_train_op,self.D_training_op, self.policy_training_op], feed_dict)

        if itr % self._qf_target_update_interval == 0:
            self._sess.run(self._target_ops)
    
    def _get_feed_dict(self, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""
        
        feeds = {
            self._observations_ph: batch     ['observations'],
            self._actions_pl: batch          ['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch          ['rewards'],
            self._terminals_pl: batch        ['terminals'],
            self._policy_lr_pl:              self._policy_lr,
            self._df_lr_pl:                  self.df_lr,
        }
        
        return feeds
    
    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information.

        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.

        Also call the `draw` method of the plotter, if plotter is defined.
        """
        
        feeds = self._get_feed_dict(batch)
        qf, bellman_residual, d_loss, q_loss, fkl, lld = self._sess.run(
                [self._q_values, self._bellman_residual, self.D_loss, self.policy_loss,
                 self.D_fake_loss, self.loss_log_density], feeds)
        
        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('mean-sq-bellman-error', bellman_residual)
        logger.record_tabular('d_loss', d_loss)
        logger.record_tabular('gen_loss', q_loss)
        logger.record_tabular('self.D_fake_loss', fkl)
        logger.record_tabular('loss_log_density', lld)
        
        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()
    
    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """
        
        if self._save_full_state:
            return {'epoch': epoch, 'algo': self}
        
        return {
            'epoch':        epoch,
            'policy':       self.policy,
            'qf':           self.qf,
            'env':          self.env,
            'discrimintor': self.df,
        }
    
    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""
        
        state = Serializable.__getstate__(self)
        state.update({
            'qf-params':     self.qf.get_param_values(),
            'policy-params': self.policy.get_param_values(),
            'df-params':     self.df.get_params_internal(),
            'pool':          self.pool.__getstate__(),
            'env':           self.env.__getstate__(),
        })
        return state
    
    def __setstate__(self, state):
        """Set Serializable state fo the RLAlgorithm instance."""
        
        Serializable.__setstate__(self, state)
        self.qf.set_param_values(state['qf-params'])
        self.df.set_param_values(state['df-params'])
        self.policy.set_param_values(state['policy-params'])
        self.pool.__setstate__(state['pool'])
        self.env.__setstate__(state['env'])
