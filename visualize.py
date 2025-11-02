import numpy as np
import torch
import matplotlib.pyplot as plt
from models import ParametricClampedBeam, analytical_solution

def load_model(path='parametric_pinn_model.pt'):
    ckpt=torch.load(path,map_location='cpu')
    model=ParametricClampedBeam(ckpt.get('hidden_layers',[64,64,64]), L=ckpt.get('L',1.0))
    model.load_state_dict(ckpt['model_state_dict'])
    model.set_normalization_params(ckpt['E_ref'], ckpt['I_ref'], ckpt['q_ref'])
    model.eval()
    return model, ckpt.get('L',1.0)

def plot_case(ax, model, L, E, I, q, label):
    x=torch.linspace(0,1,200).reshape(-1,1).requires_grad_(True)
    E_t=torch.full_like(x,float(E))
    I_t=torch.full_like(x,float(I))
    q_t=torch.full_like(x,float(q))
    out=model(x,E_t,I_t,q_t)
    # convert to mm for plotting
    w_p=out['w'].detach().numpy().flatten()*1000
    x_np=x.detach().numpy().flatten()
    w_t=analytical_solution(x_np,L,float(E),float(I),float(q))*1000
    ax.plot(x_np,w_t,'k-',lw=2,label='Analytical')
    ax.plot(x_np,w_p,'r--',lw=2,label='PINN')
    ax.set_title(label)
    ax.set_xlabel('x/L')
    ax.set_ylabel('Deflection (mm)')
    ax.grid(True,alpha=0.3)

# load trained model
model,L=load_model()
cases=[
    ('Center', 200e9, 1e-6, 1000),
    ('E=150', 150e9, 1e-6, 1000),
    ('E=250', 250e9, 1e-6, 1000),
    ('I=1.5e-6', 200e9, 1.5e-6, 1000),
    ('q=1250', 200e9, 1e-6, 1250),
    ('q=2000 (out)', 200e9, 1e-6, 2000),
]

fig,axs=plt.subplots(2,3,figsize=(14,8))
axs=axs.ravel()
for ax,(name,E,I,q) in zip(axs,cases):
    plot_case(ax,model,L,E,I,q,name)
    
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=[0,0,1,0.95])
plt.savefig('visualizations.png', dpi=200)
print('Saved visualizations.png')
