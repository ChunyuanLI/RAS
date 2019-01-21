import tensorflow as tf

from rllab.core.serializable import Serializable

from softqlearning.misc.nn import MLPFunction


class NNVFunction(MLPFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100, 100),
                 name='value_function'):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')

        super(NNVFunction, self).__init__(
            self._observations_ph,
            name=name,
            hidden_layer_sizes=hidden_layer_sizes,
            out_neuron=1)

    def eval(self, observations):
        return super(NNVFunction, self)._eval(observations)

    def output_for(self, observations, reuse=False):
        return super(NNVFunction, self)._output_for(observations, reuse=reuse)


class NNQFunction(MLPFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100, 100),
                 name='q_function'):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._Do], name='observations')
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._Da], name='actions')

        super(NNQFunction, self).__init__(
            self._observations_ph,
            self._actions_ph,
            name=name,
            hidden_layer_sizes=hidden_layer_sizes,
            out_neuron=1)

    def output_for(self, observations, actions, reuse=False):
        return super(NNQFunction, self)._output_for(
            observations, actions, reuse=reuse)

    def eval(self, observations, actions):
        return super(NNQFunction, self)._eval(observations, actions)


class VFunction(MLPFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100, 100),
                 name='v_function'):
        Serializable.quick_init(self, locals())
        
        self._Do = env_spec.observation_space.flat_dim
        
        self._observations_ph = tf.placeholder(
                tf.float32, shape=[None, self._Do], name='observations')

        super(VFunction, self).__init__(
                self._observations_ph,
                name=name,
                hidden_layer_sizes=hidden_layer_sizes,
                out_neuron=1)
    
    def output_for(self, observations, reuse=False):
        return super(VFunction, self)._output_for(
                observations, reuse=reuse)
    
    def eval(self, observations):
        return super(VFunction, self)._eval(observations)


class DFunction(MLPFunction):
    def __init__(self,
                 env_spec,
                 hidden_layer_sizes=(100,100),
                 name='d_function'):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim
        self._obs_act_ph = tf.placeholder(
            tf.float32, shape=[None, self._Da + self._Do], name='disc_act')

        super(DFunction, self).__init__(
            self._obs_act_ph,
            name=name,
            hidden_layer_sizes=hidden_layer_sizes,
            out_neuron=1)

    def output_for(self, actions, reuse=False,sn=True):
        return super(DFunction, self)._output_for(actions, reuse=reuse,sn=sn)

    def eval(self, actions):
        return super(DFunction, self)._eval(actions)
    

# reverse function in GAN, no use
# class RFunction(MLPFunction):
#     def __init__(self,
#                  env_spec,
#                  hidden_layers_sizes=(100,100),
#                  name='r_function'):
#
#         Serializable.quick_init(self, locals())
#
#         self._Do = env_spec.observation_space.flat_dim
#         self._Da = env_spec.action_space.flat_dim
#         self._actions_ph = tf.placeholder(
#                 tf.float32, shape=[None, self._Da], name='reverse')
#
#         super(RFunction, self).__init__(
#                 self._actions_ph,
#                 name=name,
#                 hidden_layer_sizes=hidden_layers_sizes,
#                 out_neuron= hidden_layers_sizes[0]
#         )
#
#     def output_for(self, actions, reuse=False):
#         return super(RFunction, self)._output_for(actions, reuse=reuse)
#
#     def eval(self, actions):
#         return super(RFunction, self)._eval(actions)