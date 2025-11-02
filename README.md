# Parametric PINN for Euler–Bernoulli Beams

Minimal, clean version that generalizes across E, I, q. Left clamp BCs enforced by trial solution; right free end via loss. Physics scaling s=qL^4/(EI) and normalized residual (EI w''''/q - 1).

## Usage
- Train
  - python train.py
- Test
  - python test.py

## Files
- models.py — network and analytical solution
- train.py — training loop with importance sampling
- test.py — 14-case generalization suite
- requirements.txt — minimal deps

## Notes
- Model file saved as parametric_pinn_model.pt
- For better extrapolation on high q, increase sampling and epochs.
