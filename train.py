import time
import torch, torch.optim as optim
import numpy as np
from tqdm import tqdm
from models import ParametricClampedBeam, analytical_solution


def gen_data(n=15000, E_range=(100e9,300e9), I_range=(0.5e-6,2e-6), q_range=(500,1500), L=1.0,
             importance=True, hi_I=(1.3e-6,2e-6), hi_q=(1200.0,1500.0)):
    if not importance:
        E=np.random.uniform(*E_range, n)
        I=np.random.uniform(*I_range, n)
        q=np.random.uniform(*q_range, n)
    else:
        n_uni=n//2; n_hiI=n//4; n_hiq=n - n_uni - n_hiI
        E=np.concatenate([
            np.random.uniform(*E_range,n_uni),
            np.random.uniform(*E_range,n_hiI),
            np.random.uniform(*E_range,n_hiq),
        ])
        I=np.concatenate([
            np.random.uniform(*I_range,n_uni),
            np.random.uniform(*hi_I,n_hiI),
            np.random.uniform(*I_range,n_hiq),
        ])
        q=np.concatenate([
            np.random.uniform(*q_range,n_uni),
            np.random.uniform(*q_range,n_hiI),
            np.random.uniform(*hi_q,n_hiq),
        ])
    x=np.random.beta(0.5,0.5,n)
    w=np.array([analytical_solution(xi,L,Ei,Ii,qi) for xi,Ei,Ii,qi in zip(x,E,I,q)])
    return {
        'x':torch.tensor(x.reshape(-1,1),dtype=torch.float32),
        'E':torch.tensor(E.reshape(-1,1),dtype=torch.float32),
        'I':torch.tensor(I.reshape(-1,1),dtype=torch.float32),
        'q':torch.tensor(q.reshape(-1,1),dtype=torch.float32),
        'w':torch.tensor(w.reshape(-1,1),dtype=torch.float32),
    }

def pde_residual(E,I,q,out):
    eps=torch.finfo(out['w_xxxx'].dtype).eps
    return (E*I*out['w_xxxx'])/(q+eps) - 1.0


def train():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L=1.0
    model=ParametricClampedBeam([64,64,64],L=L).to(device)
    model.set_normalization_params(200e9,1e-6,1000.0)

    data=gen_data(15000)
    for k in data: data[k]=data[k].to(device)

    opt=optim.Adam(model.parameters(), lr=5e-4)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=400)

    pde_w, bc_w, data_w = 10.0, 50.0, 1.0
    start=time.time()
    pbar=tqdm(range(6000), desc='Training')
    for epoch in pbar:
        model.train(); opt.zero_grad()
        idx=torch.randint(0,len(data['x']),(512,),device=device)
        x=data['x'][idx].requires_grad_(True)
        E=data['E'][idx]; I=data['I'][idx]; q=data['q'][idx]; w_true=data['w'][idx]
        out=model(x,E,I,q)
        res=pde_residual(E,I,q,out); loss_pde=(res**2).mean()
        n_bc=64
        x_bc=torch.ones(n_bc,1,device=device,requires_grad=True)
        E_bc=torch.rand(n_bc,1,device=device)*200e9 + 100e9
        I_bc=torch.rand(n_bc,1,device=device)*1.5e-6 + 0.5e-6
        q_bc=torch.rand(n_bc,1,device=device)*1000 + 500
        out_bc=model(x_bc,E_bc,I_bc,q_bc)
        loss_bc=(out_bc['w_xx']**2).mean() + (out_bc['w_xxx']**2).mean()
        loss_data=((out['w']-w_true)**2).mean()
        loss=pde_w*loss_pde + bc_w*loss_bc + data_w*loss_data
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step(loss)
        if epoch%200==0:
            pbar.set_postfix(Loss=f"{loss.item():.2e}", PDE=f"{loss_pde.item():.2e}", BC=f"{loss_bc.item():.2e}", Data=f"{loss_data.item():.2e}")
    print(f"Training took {(time.time()-start):.1f}s")

    torch.save({'model_state_dict':model.state_dict(),'E_ref':200e9,'I_ref':1e-6,'q_ref':1000.0,'L':L,'hidden_layers':[64,64,64]}, 'parametric_pinn_model.pt')
    print('Saved model to parametric_pinn_model.pt')

if __name__=='__main__':
    train()
