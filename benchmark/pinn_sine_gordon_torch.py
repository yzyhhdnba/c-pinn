"""
PyTorch PINN solver for Sine-Gordon equation: u_tt - u_xx + sin(u) = 0
Configuration matches C++ version exactly:
  - Network: [2, 50, 50, 50, 1], tanh activation
  - Optimizer: Adam(lr=1e-3)
  - Iterations: 1000, batch_size: 64
  - Domain: [0,1] x [0,1]
  - Xavier uniform init
"""
import time
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

def main():
    print("=== PyTorch Sine-Gordon Solver (Autograd) ===")

    torch.manual_seed(42)
    device = torch.device('cpu')

    layers = [2, 50, 50, 50, 1]
    model = PINN(layers).to(device).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    iterations = 1000
    batch_size = 64

    start_time = time.perf_counter()

    for it in range(iterations):
        optimizer.zero_grad()

        x_t = torch.rand(batch_size, 2, dtype=torch.float64, device=device)
        x_t.requires_grad_(True)

        u = model(x_t)

        # First-order gradients
        grads = torch.autograd.grad(u, x_t, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        # Second-order: u_xx, u_tt
        u_xx_grads = torch.autograd.grad(u_x, x_t, grad_outputs=torch.ones_like(u_x),
                                          create_graph=True)[0]
        u_xx = u_xx_grads[:, 0:1]

        u_tt_grads = torch.autograd.grad(u_t, x_t, grad_outputs=torch.ones_like(u_t),
                                          create_graph=True)[0]
        u_tt = u_tt_grads[:, 1:2]

        # PDE residual: u_tt - u_xx + sin(u) = 0
        residual = u_tt - u_xx + torch.sin(u)
        loss = torch.mean(residual ** 2)

        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"Iter {it} Loss: {loss.item():.6e}")

    elapsed = time.perf_counter() - start_time
    print(f"Training finished. Time: {elapsed:.3f}s")

if __name__ == "__main__":
    main()
