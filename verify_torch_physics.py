import sys
sys.path.insert(0, r'c:\Users\shubh\Downloads\AEROTHON HAL')

import torch
from physics_model import build_physics_model

model = build_physics_model()
inputs = {
    'T4': torch.tensor(1500.0, requires_grad=True),
    'P4': torch.tensor(2.0e6, requires_grad=True),
    'P5': torch.tensor(1.0e5, requires_grad=True),
    'RPM': torch.tensor(12000.0, requires_grad=True),
    'mdot_core': torch.tensor(0.4, requires_grad=True),
    'mdot_cooling': torch.tensor(0.05, requires_grad=True),
    'Tcool': torch.tensor(800.0, requires_grad=True),
    'Pcool': torch.tensor(1.5e6, requires_grad=True),
    'flight_hours': torch.tensor(1.5, requires_grad=True),
}

state, outputs = model.step(inputs)
loss = torch.mean(outputs['predicted_T5'] * outputs['predicted_T5'])
loss.backward()

print('loss=', loss.item())
print('grad_T4=', inputs['T4'].grad.item())
print('grad_P4=', inputs['P4'].grad.item())
