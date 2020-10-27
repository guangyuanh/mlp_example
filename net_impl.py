import torch

class OneHiddenNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(OneHiddenNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h      = self.linear1(x).sigmoid()
        y_pred = self.linear2(h)
        return y_pred
