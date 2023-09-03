import torch
import torch.nn as nn
from model.mstcn import SingleStageModel
from model.transformer import Transformer


class SpaTemModel(nn.Module):
    def __init__(self, args):
        super(SpaTemModel, self).__init__()
        self.fusion_model = SingleStageModel(num_layers=16,
                                             num_f_maps=32,
                                             dim=2048,
                                             num_classes=256,
                                             causal_conv=True,
                                             is_train=True)
        self.transformer = Transformer(32, 2048, args.out_classes, args.trans_seq, d_model=32)

    def forward(self, x):

        x = self.fusion_model(x)
        x = self.transformer(x)

        return x
