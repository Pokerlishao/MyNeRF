import torch
from torch import nn
import torch.nn.functional as F

class NeRF_MLP(nn.Module):
    def __init__(self, d_input=3, n_layers=8, d_hidden=256, skip=(4,), d_viewdirs = None):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.skip = skip
        self.act = F.relu
        self.d_viewdirs = d_viewdirs
        
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, self.d_hidden)] + 
            [nn.Linear(self.d_hidden, self.d_hidden) if i not in skip 
                else nn.Linear(self.d_hidden + self.d_input, self.d_hidden)
                for i in range(n_layers-1)]
        )
        
        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(self.d_hidden, 1)
            self.rgb_hidden = nn.Linear(self.d_hidden, self.d_hidden)
            self.branch = nn.Linear(self.d_hidden + self.d_viewdirs, self.d_hidden//2)
            self.output = nn.Linear(d_hidden//2, 3)
        else:
            self.output = nn.Linear(d_hidden, 4)
    

    def forward(self, x, viewdirs=None):
        # assert self.d_viewdirs is None and viewdirs is not None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x,x_input], dim=-1)
        
        if self.d_viewdirs is not None:      
            alpha = self.alpha_out(x)
            x = self.rgb_hidden(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.branch(x)
            x = self.act(x)
            x = self.output(x)
            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)
        return x