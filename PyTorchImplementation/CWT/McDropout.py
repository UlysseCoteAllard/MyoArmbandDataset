import torch.nn.functional as F
from torch.nn.modules import Module

class McDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(McDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, training=True)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) \
               + inplace_str + ')'
