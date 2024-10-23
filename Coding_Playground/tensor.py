import torch
import numpy as np

def f(x, y):
    return torch.sqrt(x**2 + y**2)
    
x = torch.tensor(3.0)
y = torch.tensor(4.0)

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(data)
print(x_data)

# print(x)
# print(f(x, y))