import torch
import numpy as np
from models import ParametricClampedBeam, analytical_solution

def load_model(path='parametric_pinn_model.pt'):
    ckpt=torch.load(path,map_location='cpu')
    model=ParametricClampedBeam(ckpt.get('hidden_layers',[64,64,64]), L=ckpt.get('L',1.0))
    model.load_state_dict(ckpt['model_state_dict'])
    model.set_normalization_params(ckpt['E_ref'], ckpt['I_ref'], ckpt['q_ref'])
    model.eval()
    return model, ckpt.get('L',1.0)

def eval_case(model,L,E,I,q,N=200):
    x=torch.linspace(0,1,N).reshape(-1,1).requires_grad_(True)
    E_t=torch.full_like(x,float(E))
    I_t=torch.full_like(x,float(I))
    q_t=torch.full_like(x,float(q))
    out=model(x,E_t,I_t,q_t)
    w=out['w'].detach().numpy().flatten()
    x_np=x.detach().numpy().flatten()
    w_true=analytical_solution(x_np,L,float(E),float(I),float(q))
    # compute relative L2 error
    l2=((w-w_true)**2).mean()**0.5
    rel=l2/((w_true**2).mean()**0.5 + 1e-15)
    return rel

# load model and run tests
model,L=load_model()
tests=[
    {'name':'Training center','E':200e9,'I':1e-6,'q':1000},
    {'name':'E=150','E':150e9,'I':1e-6,'q':1000},
    {'name':'E=250','E':250e9,'I':1e-6,'q':1000},
    {'name':'I=0.75e-6','E':200e9,'I':0.75e-6,'q':1000},
    {'name':'I=1.5e-6','E':200e9,'I':1.5e-6,'q':1000},
    {'name':'q=750','E':200e9,'I':1e-6,'q':750},
    {'name':'q=1250','E':200e9,'I':1e-6,'q':1250},
    {'name':'E=100 (edge)','E':100e9,'I':1e-6,'q':1000},
    {'name':'E=300 (edge)','E':300e9,'I':1e-6,'q':1000},
    {'name':'OUT q=2000','E':200e9,'I':1e-6,'q':2000},
]

for t in tests:
    r=eval_case(model,L,t['E'],t['I'],t['q'])
    print(f"{t['name']}: rel L2={r:.4f}")
