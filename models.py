import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, sizes=[25,12,12], act_funcs=[]):
        super(NeuralNet, self).__init__()
        self.layers= nn.ModuleList()
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1], sizes[i]))
        self.act_funcs = act_funcs
        self.init_weight()

    def forward(self, x):
        for i in range(len(self.layers)):
            x=self.layers[i](x)
            x=self.act_funcs[i](x)
        return x

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight * 10)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(len(self.layers)):
            self.init_layer(self.layers[i])


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch
    c=NeuralNet(sizes=[5,12,12],act_funcs=[F.relu_ for i in range(3)])
    d=c(torch.tensor([i for i in range(5)],dtype=torch.float32))
    print(d.shape)
