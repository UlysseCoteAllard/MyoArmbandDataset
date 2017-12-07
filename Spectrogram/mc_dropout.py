from lasagne.layers import Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
import theano.tensor as T

class MCDropout(Layer):
    """
    Exactly like normal dropout, but is always on regardless of Deterministic
    """
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(MCDropout, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, **kwargs):
        # Using theano constant to prevent upcasting
        one = T.constant(1)

        retain_prob = one - self.p
        if self.rescale:
            input /= retain_prob

        # use nonsymbolic shape for dropout mask if possible
        mask_shape = self.input_shape
        if any(s is None for s in mask_shape):
            mask_shape = input.shape

        # apply dropout, respecting shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            mask_shape = tuple(1 if a in shared_axes else s
                               for a, s in enumerate(mask_shape))
        mask = self._srng.binomial(mask_shape, p=retain_prob,
                                   dtype=input.dtype)
        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in mask_shape)
            mask = T.patternbroadcast(mask, bcast)
        return input * mask

mc_dropout = MCDropout  # shortcut