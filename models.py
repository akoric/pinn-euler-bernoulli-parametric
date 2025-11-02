import torch
import torch.nn as nn
import numpy as np

# neural network for deflection with parametric inputs
class ParametricDeflectionNet(nn.Module):
    def __init__(self, hidden_layers=[64,64,64], activation='tanh'):
        super().__init__()
        sizes = [4] + hidden_layers + [1]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.Tanh() if activation=='tanh' else nn.SiLU())
        self.net = nn.Sequential(*layers)
        # reference values for normalization 
        self.register_buffer('E_ref', torch.tensor(200e9))
        self.register_buffer('I_ref', torch.tensor(1e-6))
        self.register_buffer('q_ref', torch.tensor(1000.0))
        self.apply(self._init)
        
    def _init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def set_normalization_params(self,E,I,q):
        self.E_ref=torch.tensor(E)
        self.I_ref=torch.tensor(I)
        self.q_ref=torch.tensor(q)
        
    def normalize(self,x,E,I,q):
        return x, E/self.E_ref, I/self.I_ref, q/self.q_ref
        
    def forward(self,x,E,I,q):
        x,E,I,q=self.normalize(x,E,I,q)
        inp=torch.cat([x,E,I,q],dim=1)
        return self.net(inp)
        
    def derivatives(self,x,E,I,q):
        x=x.requires_grad_(True)
        psi=self.forward(x,E,I,q)
        # compute derivatives up to 4th order
        psi_x=torch.autograd.grad(psi,x,torch.ones_like(psi),create_graph=True,retain_graph=True)[0]
        psi_xx=torch.autograd.grad(psi_x,x,torch.ones_like(psi_x),create_graph=True,retain_graph=True)[0]
        psi_xxx=torch.autograd.grad(psi_xx,x,torch.ones_like(psi_xx),create_graph=True,retain_graph=True)[0]
        psi_xxxx=torch.autograd.grad(psi_xxx,x,torch.ones_like(psi_xxx),create_graph=True,retain_graph=True)[0]
        return psi,psi_x,psi_xx,psi_xxx,psi_xxxx

class ParametricClampedBeam(nn.Module):
    def __init__(self, hidden_layers=[64,64,64], L=1.0):
        super().__init__()
        self.base=ParametricDeflectionNet(hidden_layers)
        self.L=L
        
    def set_normalization_params(self,E,I,q):
        self.base.set_normalization_params(E,I,q)
        
    def forward(self,x,E,I,q):
        psi,px,pxx,pxxx,pxxxx=self.base.derivatives(x,E,I,q)
        x2=x*x
        # apply boundary conditions through multiplication with x^2
        w=x2*psi
        w_x=2*x*psi + x2*px
        w_xx=2*psi + 4*x*px + x2*pxx
        w_xxx=6*px + 6*x*pxx + x2*pxxx
        w_xxxx=12*pxx + 8*x*pxxx + x2*pxxxx
        
        L=self.L
        # scaling factor
        s=(q*(L**4))/(E*I + torch.finfo(x.dtype).eps)
        w=s*w
        w_x=s*(w_x/L)
        w_xx=s*(w_xx/(L**2))
        w_xxx=s*(w_xxx/(L**3))
        w_xxxx=s*(w_xxxx/(L**4))
        return {'w':w,'w_x':w_x,'w_xx':w_xx,'w_xxx':w_xxx,'w_xxxx':w_xxxx}

# analytical solution for clamped beam deflection
def analytical_solution(x,L,E,I,q):
    if isinstance(x,torch.Tensor):
        x=x.detach().cpu().numpy()
    xi=x
    return (q*L**4)/(24*E*I) * xi**2*(6-4*xi+xi**2)
