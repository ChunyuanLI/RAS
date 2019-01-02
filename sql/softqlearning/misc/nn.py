import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized

from softqlearning.misc import tf_utils
from softqlearning.misc import snlayer

def feedforward_net(
        *inputs,
        layer_sizes,
        activation_fn=tf.nn.relu,
        output_nonlinearity=None,
        is_policy_net = False,
        sn = False):

    def bias(n_units):
        return tf.get_variable(
            name='bias',
            shape=n_units,
            initializer=tf.zeros_initializer())

    def linear(x, n_units, postfix=None,sn = False):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        # `tf.tensordot` supports broadcasting
        if(sn):
            return tf.tensordot(x, snlayer.spectral_norm(weight), axes=[-1, 0])
        else:
            return tf.tensordot(x, (weight), axes=[-1, 0])

    out = 0
    l1out = 0
    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('layer_{i}'.format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    with tf.variable_scope('input' + str(j)):
                        out += linear(input_tensor, layer_size, j,sn=sn)
                        if(is_policy_net):
                            l1out = out
            else:
                if(is_policy_net):
                    out_new = linear(out, layer_size)
                    # out_new += linear(tf.random_normal(tf.shape(out))*0.1,layer_size,str(2))
                    out = out_new
                else:
                    out = linear(out, layer_size)

            out += bias(layer_size)

            if i < len(layer_sizes) - 1 and activation_fn:
                out = activation_fn(out)


    if output_nonlinearity:
        out = output_nonlinearity(out)

    if(is_policy_net):
        return out, l1out
    else:
        return out


class MLPFunction(Parameterized, Serializable):
    def __init__(self, *inputs, name, hidden_layer_sizes, out_neuron):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._inputs = inputs
        self._layer_sizes = list(hidden_layer_sizes) + [out_neuron]

        self._output = self._output_for(*self._inputs)
        self._out_neuron = out_neuron

    def _output_for(self, *inputs, reuse=tf.AUTO_REUSE,sn=False):
        with tf.variable_scope(self._name, reuse=reuse):
            out = feedforward_net(
                *inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes,
                    sn=sn,
            )

        if(self._layer_sizes[-1]>1):
            return out
        else:
            return out[..., 0]

    def _eval(self, *inputs):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope += '/' + self._name if scope else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

# class DiscFunction(Parameterized, Serializable):
#     def __init__(self, *inputs, name, hidden_layer_sizes):
#         Parameterized.__init__(self)
#         Serializable.quick_init(self, locals())
#
#         self._name = name
#         self._inputs = inputs
#         self._layer_sizes = list(hidden_layer_sizes) + [1] #input?
#         self._output = self._output_for(*self._inputs)
#
#     def _output_for(self, *inputs, reuse=tf.AUTO_REUSE): # In Tensorflow, what does reuse mean?
#         with tf.variable_scope(self._name, reuse=reuse):
#             out = feedforward_net(
#                 *inputs,
#                 output_nonlinearity=None,
#                 layer_sizes=self._layer_sizes,
#             )
#
#         return out[..., 0]
#
#     # def _eval(self, *inputs):
#     def get_params_internal(self, scope='', **tags):
#         if len(tags) > 0:
#             raise NotImplementedError
#
#         scope += '/' + self._name if scope else self._name
#         return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)