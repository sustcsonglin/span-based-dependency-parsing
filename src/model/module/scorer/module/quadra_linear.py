import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pdb
import opt_einsum

## TODO: 完成一个quadra-linear的label的打分函数?
class QlinearScorer(nn.Module):
    """
    Outer product version of trilinear function.
    Trilinear attention layer.
    """

    def __init__(self, input_size_1, input_size_2, input_size_3, input_size_4,  factorize=True,
                 **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        # super(QlinearScorer, self).__init__()
        # self.input_size_1 = input_size_1 + 1
        # self.input_size_2 = input_size_2 + 1
        # self.input_size_3 = input_size_3 + 1
        # self.input_size_4 = input_size_3 + 1
        # self.rank = rank
        # self.init_std = init_std
        # self.factorize = factorize
        # if not factorize:
        #     self.W = Parameter(torch.Tensor(self.input_size_1, self.input_size_2, self.input_size_3, self.input_size_4))
        # else:
        #     self.W_1 = Parameter(torch.Tensor(self.input_size_1, self.rank))
        #     self.W_2 = Parameter(torch.Tensor(self.input_size_2, self.rank))
        #     self.W_3 = Parameter(torch.Tensor(self.input_size_3, self.rank))
        #     self.W_4 = Parameter(torch.Tensor(self.input_size_4, self.rank))
        #     self.W_5 = Parameter(torch.Tensor(self.rank, ))
        # # if self.biaffine:
        # #     self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        # # else:
        # #     self.register_parameter('U', None)
        #
        # self.reset_parameters()

    # def reset_parameters(self):
    #     if not self.factorize:
    #         nn.init.xavier_normal_(self.W)
    #     else:
    #         nn.init.xavier_normal_(self.W_1, gain=self.init_std)
    #         nn.init.xavier_normal_(self.W_2, gain=self.init_std)
    #         nn.init.xavier_normal_(self.W_3, gain=self.init_std)
    #         nn.init.xavier_normal_(self.W_4, gain=self.init_std)

    def forward(self, layer1, layer2, layer3, layer4 = delattr()):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert layer1.size(0) == layer2.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        layer_shape = layer1.size()
        one_shape = list(layer_shape[:2]) + [1]
        ones = torch.ones(one_shape).cuda()
        layer1 = torch.cat([layer1, ones], -1)
        layer2 = torch.cat([layer2, ones], -1)
        layer3 = torch.cat([layer3, ones], -1)
        layer4 = torch.cat([layer4, ones], -1)

        layer = opt_einsum.contract('bnx,xr,bny,yr,bnz,zr, bns,sr->bn', self.W_1, layer1, self.W_2, layer2, self.W_3, layer3, self.W_4, layer4, backend='torch')
        return layer
