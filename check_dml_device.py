import torch
import torch_directml

for i in range(torch_directml.device_count()):
    dml = torch_directml.device(i)
    print(dml)
    print(torch_directml.device_name(i))
