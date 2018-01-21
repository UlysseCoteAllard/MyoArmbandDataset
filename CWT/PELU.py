import lasagne
from lasagne import nonlinearities
from lasagne import init
import theano

class ParametricExponentialLayer(lasagne.layers.Layer):
    """
    lasagne.layers.ParametricExponentialLayer(incoming,
    alpha=init.Constant(0.25), shared_axes='auto', **kwargs)

    A layer that applies parametric exponential nonlinearity to its input
    following [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    alpha : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the alpha values. The
        shape must match the incoming shape, skipping those axes the alpha
        values are shared over (see the example below).
        See :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the parameters of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the parameter over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, which corresponds to a single scalar parameter.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

     References
    ----------
    .. Trottier, Ludovic, Philippe Giguere, and Brahim Chaib-draa.
    "Parametric exponential linear unit for deep convolutional neural networks."
    arXiv preprint arXiv:1605.09332 (2016).
    """
    def __init__(self, incoming, alpha=init.Constant(1.), beta=init.Constant(1.), shared_axes='auto',
                 **kwargs):
        super(ParametricExponentialLayer, self).__init__(incoming, **kwargs)
        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = self.add_param(alpha, shape, name="alpha",
                                    regularizable=False)
        self.beta = self.add_param(beta, shape, name="beta",
                                   regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.alpha.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(input.ndim)]
        alpha = self.alpha.dimshuffle(pattern)
        beta = self.beta.dimshuffle(pattern)

        alpha = theano.tensor.switch(alpha >= .1, alpha, .1)
        beta = theano.tensor.switch(beta >= .1, beta, .1)
        delta = 1./beta
        return theano.tensor.switch(input >= 0, alpha*delta*input,
                                    alpha*theano.tensor.exp(delta*input) - 1)



def pelu(layer, **kwargs):
    """
    Convenience function to apply pelu to a given layer's output.
    Will set the layer's nonlinearity to identity if there is one and will
    apply the parametric exponential instead.

    Parameters
    ----------
    layer: a :class:`Layer` instance
        The `Layer` instance to apply the parametric exponential layer to;
        note that it will be irreversibly modified as specified above

    **kwargs
        Any additional keyword arguments are passed to the
        :class:`ParametericRectifierLayer`

    Examples
    --------
    Note that this function modifies an existing layer, like this:

    from lasagne.layers import InputLayer, DenseLayer, pelu
    layer = InputLayer((32, 100))
    layer = DenseLayer(layer, num_units=200)
    layer = pelu(layer)

    In particular, :func:`pelu` can *not* be passed as a nonlinearity.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    return ParametricExponentialLayer(layer, **kwargs)
