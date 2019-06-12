import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models.resnet import ResNet, BasicBlock


class CompositionalEmbedding(nn.Module):
    r"""A simple compositional codeword and codebook that store embeddings.

     Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): size of each embedding vector
        num_codebook (int): size of the codebook of embeddings
        num_codeword (int, optional): size of the codeword of embeddings
        weighted (bool, optional): weighted version of unweighted version
        return_code (bool, optional): return code or not

     Shape:
         - Input: (LongTensor): (N, W), W = number of indices to extract per mini-batch
         - Output: (Tensor): (N, W, embedding_dim)

     Attributes:
         - code (Tensor): the learnable weights of the module of shape
              (num_embeddings, num_codebook, num_codeword)
         - codebook (Tensor): the learnable weights of the module of shape
              (num_codebook, num_codeword, embedding_dim)

     Examples::
         >>> m = CompositionalEmbedding(200, 64, 16, 32, weighted=False)
         >>> a = torch.randperm(128).view(16, -1)
         >>> output = m(a)
         >>> print(output.size())
         torch.Size([16, 8, 64])
     """

    def __init__(self, num_embeddings, embedding_dim, num_codebook, num_codeword=None, num_repeat=10, weighted=True,
                 return_code=False):
        super(CompositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_codebook = num_codebook
        self.num_repeat = num_repeat
        self.weighted = weighted
        self.return_code = return_code

        if num_codeword is None:
            num_codeword = math.ceil(math.pow(num_embeddings, 1 / num_codebook))
        self.num_codeword = num_codeword
        self.code = Parameter(torch.Tensor(num_embeddings, num_codebook, num_codeword))
        self.codebook = Parameter(torch.Tensor(num_codebook, num_codeword, embedding_dim))

        nn.init.normal_(self.code)
        nn.init.normal_(self.codebook)

    def forward(self, input):
        batch_size = input.size(0)
        index = input.view(-1)
        code = self.code.index_select(dim=0, index=index)
        if self.weighted:
            # reweight, do softmax, make sure the sum of weight about each book to 1
            code = F.softmax(code, dim=-1)
            out = (code[:, :, None, :] @ self.codebook[None, :, :, :]).squeeze(dim=-2).sum(dim=1)
        else:
            # because Gumbel SoftMax works in a stochastic manner, needs to run several times to
            # get more accurate embedding
            code = (torch.sum(torch.stack([F.gumbel_softmax(code) for _ in range(self.num_repeat)]), dim=0)).argmax(
                dim=-1)
            out = []
            for index in range(self.num_codebook):
                out.append(self.codebook[index, :, :].index_select(dim=0, index=code[:, index]))
            out = torch.sum(torch.stack(out), dim=0)
            code = F.one_hot(code, num_classes=self.num_codeword)

        out = out.view(batch_size, -1, self.embedding_dim)
        code = code.view(batch_size, -1, self.num_codebook, self.num_codeword)
        if self.return_code:
            return out, code
        else:
            return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_embeddings) + ', ' + str(self.embedding_dim) + ')'


class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_class), []
        for name, module in basic_model.named_children():
            if isinstance(module, nn.Linear) or isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        self.linear = nn.Linear(7 * 7 * 512, num_class)

        # embedding
        self.embedding = CompositionalEmbedding(vocab_size, embedding_size, num_codebook, num_codeword, weighted=False)

        # to Hash
        self.hash_layer = nn.Linear(config.embedding_size, config.hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)  # means和std -> 均值和标准差
        self.hash_layer.bias.data.fill_(0.0)  # 补0

    def forward(self, x):
        # 基础网络(ResNet18...)
        x = self.base(x)
        # x = x.permute(0, 2, 3, 1)
        # x = x.contiguous().view(x.size(0), -1, 32)
        x = x.view(x.size(0), -1)
        feature = x
        classes = self.linear(x)
        # classes = self.softmax(x)
        # 分类
        # classes = x.norm(dim=-1)

        # embedding
        # x = self.embedding(x)

        # to Hash
        # hash = self.hash_layer(x)

        # return classes, hash
        return classes, feature
