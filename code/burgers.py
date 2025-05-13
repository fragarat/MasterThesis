import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid and time parameters
M = 9999  # Number of time steps
T = 1.0   # Final time
dt = T / M  # Time step size

#N = 998  # Number of spatial grid points
N = 2**7  # Number of spatial grid points
L = 1.0  # Domain length
dx = L / (N + 1)  # Spatial step size (excluding boundary points)

x = np.linspace(0, L, N+2)  # Includes boundary points
t = np.linspace(0, T, M+1)

# Diffusivity (viscosity)
nu = 0.01

# Initial condition q(x,0) = sin(πx) (excluding boundaries)
q0 = np.sin(2*np.pi * x[1:-1])

# Solution matrix initialization (excluding boundaries)
Q = np.zeros((N, M+1))
Q[:, 0] = q0

# Finite Difference Matrix (Diffusion Term)
off_diag = np.full(N-1, 1 / dx**2)
main_diag = np.full(N, -2 / dx**2)

Tdx = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).tocsc()

# Function to compute next time step using implicit diffusion
def viscous_burgers(qold, nu, dt, dx):
    n = len(qold)
    dx2 = dx**2

    # Lax-Wendroff for nonlinear convection term
    qjpone = np.roll(qold, -1)
    qjmone = np.roll(qold, 1)

    LW = qold - (dt / (2 * dx)) * qold * (qjpone - qjmone) + \
         (dt**2 / (2 * dx2)) * qold * (0.5 * (qjpone - qjmone)**2 + \
         qold * (qjpone - 2 * qold + qjmone))
    
    # Apply Dirichlet BC: q(0,t) = q(1,t) = 0
    LW[0] = 0
    LW[-1] = 0

    # Implicit diffusion term
    A = np.eye(n) - nu * (dt / 2) * Tdx.toarray()
    rhs = LW + nu * (dt / 2) * (Tdx @ qold)

    # Solve linear system
    qnew = spsolve(A, rhs)
    
    # Enforce boundary conditions
    qnew[0] = 0
    qnew[-1] = 0

    return qnew

# Time stepping
qold = q0.copy()
for i in range(M):
    qnew = viscous_burgers(qold, nu, dt, dx)
    Q[:, i+1] = qnew
    qold = qnew

# Extend solution to include boundary points (q=0 at x=0 and x=1)
Qfom = np.zeros((N+2, M+1))
Qfom[1:-1, :] = Q  # Fill interior points