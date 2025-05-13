import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve

# Parameters
Lx = 10.0        # Domain length in x-direction
Ly = 10.0        # Domain length in y-direction
Nx = 100         # Number of spatial points in x-direction
Ny = 100         # Number of spatial points in y-direction
dx = Lx / (Nx - 1)  # Spatial step in x
dy = Ly / (Ny - 1)  # Spatial step in y
D = 0.1          # Diffusion coefficient
r = 1.0          # Growth rate
dt = 0.005       # Time step
T = 5.0          # Total simulation time
Nt = int(T / dt) # Number of time steps

# Spatial grids
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: 2D Gaussian at the center
u0 = np.exp(-10 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
u = u0.flatten()  # Flatten to 1D vector

# Function to construct 1D Laplacian matrix with Neumann BCs
def construct_1d_laplacian(N, d):
    main_diag = -2 * np.ones(N) / d**2
    super_diag = np.ones(N-1) / d**2
    sub_diag = np.ones(N-1) / d**2
    # Neumann BC adjustments
    super_diag[0] = 2 / d**2
    sub_diag[-1] = 2 / d**2
    return diags([sub_diag, main_diag, super_diag], [-1, 0, 1], format="csc")

# Construct 1D Laplacians
Lx = construct_1d_laplacian(Nx, dx)
Ly = construct_1d_laplacian(Ny, dy)

# Construct 2D Laplacian using Kronecker products
Ix = eye(Nx, format="csc")
Iy = eye(Ny, format="csc")
L = kron(Iy, Lx) + kron(Ly, Ix)

# Crank-Nicolson matrix
alpha = D * dt / 2
A = eye(Nx * Ny) - alpha * L
A = A.tocsc()

# Store solutions at certain time steps
snapshot_interval = 1 # Desired number of interval snapshot to store
snapshots = []

# Time stepping
for tt in range(1, Nt+1):
    u_prev = u.copy()
    reaction = r * u_prev * (1 - u_prev)
    b = u_prev + dt * reaction + alpha * L.dot(u_prev)
    u = spsolve(A, b)
    
    if tt % snapshot_interval == 0:
        snapshots.append(u.reshape(Ny, Nx))

# Convert snapshots to numpy array
Qfom = np.array(snapshots)