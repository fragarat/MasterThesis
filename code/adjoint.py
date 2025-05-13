import numpy as np
import opinf
from scipy.integrate import cumulative_trapezoid

################################ OpInf ###############################
######################################################################

opinf.utils.mpl_config()

# Parameters
r = 10
k_samples = 1000  # number of samples (snapshot data)
T = 1.0

# Snapshot data Q = [q(t_0) q(t_1) ... q(t_k)], size=(r,k)
Q = np.load("burgers_snapshots.npy")
t = np.load("burgers_time.npy")  # t_0 = 0,...,t_k = 1
x = np.load("burgers_space.npy")

dt = t[1] - t[0]

# Initialize orthonormal basis V
Vr = opinf.basis.PODBasis(cumulative_energy=0.999)
#Vr = opinf.basis.PODBasis(num_vectors=r)
Vr.fit(Q)
print(Vr)

# Compressed state snapshots Q_ by projecting onto the reduced space defined by Vr
Q_ = Vr.compress(Q)

# Estimate time derivatives using 6th–order finite differences.
ddt_estimator = opinf.ddt.UniformFiniteDifferencer(t, "ord6")
Qdot_ = ddt_estimator.estimate(Q_)[1]

# Build the quadratic continuous model and fit it.
model = opinf.models.ContinuousModel(operators=["A", "H"])
model.fit(states=Q_, ddts=Qdot_)
A_opinf = model.operators[0].entries
H_opinf = model.operators[1].entries

# Expanded H: size (r, r^2)
H_opinf = opinf.operators.QuadraticOperator.expand_entries(H_opinf)

# Flatten A and H to build initial guess for theta.
# Start from the OpInf solution.
theta = np.concatenate([A_opinf.ravel(), H_opinf.ravel()])

############################### Adjoint ##############################
######################################################################

# Integral of q(tau) from 0 to t_k for each t_k.
Q_int = np.zeros_like(Q_)
for i in range(r):
    Q_int[i, :] = cumulative_trapezoid(Q_[i, :], t, initial=0)

# Quantity q(tau) ⊗ q(tau) for each tau.
Q2_ = np.zeros((r**2, k_samples))
for k in range(k_samples):
    q_k = Q_[:, k]
    Q2_[:, k] = np.kron(q_k, q_k)

# Integral of q(tau) ⊗ q(tau) from 0 to t_k.
Q2_int = np.zeros_like(Q2_)
for i in range(r**2):
    Q2_int[i, :] = cumulative_trapezoid(Q2_[i, :], t, initial=0)

############################ GD Parameters ###########################
######################################################################

max_iter = 1000
epsilon = 1e-8   # stopping threshold for gradient norm

# Armijo parameters:
eta = 1e-3       # initial learning rate 
alpha = 1e-4
beta = 0.5
d = r**2 + r**3  

############################### GD Loop ##############################
######################################################################

for j in range(max_iter):
    A = theta[:r**2].reshape(r, r)
    H = theta[r**2:].reshape(r, r**2)
    q0 = Q_[:, 0]
    
    # Compute predicted states: tilde_Q = q0 + A @ Q_int + H @ Q2_int.
    tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
    
    # Loss computation: mean squared error.
    loss = np.mean(np.sum((Q_ - tilde_Q)**2, axis=0))
    print(f"Iteration {j}, Loss: {loss:.6f}")
    
    # Solve adjoint ODE backwards using implicit Euler.
    lambda_values = np.zeros((r, k_samples))
    lambda_values[:, -1] = 0.0  # Terminal condition: λ(T)=0
    
    for k in reversed(range(k_samples - 1)):
        q_k = Q_[:, k]
        tilde_q_k = tilde_Q[:, k]
        error_k = q_k - tilde_q_k
        
        # Compute M(q_k)
        H_3d = H.reshape(r, r, r)
        M_k = np.einsum("ijk,j->ki", H_3d, q_k)
        
        # Backward Euler step.
        matrix = np.eye(r) + dt * (A.T + 2 * M_k)
        rhs = lambda_values[:, k+1] - 2 * dt * error_k
        lambda_values[:, k] = np.linalg.solve(matrix, rhs)
    
    # Gradient computation.
    grad_A = np.zeros(r**2)
    grad_H = np.zeros(r**3)
    
    for k in range(k_samples):
        lambda_k = lambda_values[:, k]
        q_k = Q_[:, k]
        error_k = q_k - tilde_Q[:, k]
        
        # Gradient parts for A.
        outer_A = np.outer(lambda_k, q_k).flatten()
        grad_A += outer_A * dt
        outer_g_A = np.outer(error_k, q_k).flatten()
        grad_A -= 2 * outer_g_A * dt
        
        # Gradient parts for H.
        q_outer = np.outer(q_k, q_k).flatten()
        outer_H = np.outer(lambda_k, q_outer).flatten()
        grad_H += outer_H * dt
        outer_g_H = np.outer(error_k, q_outer).flatten()
        grad_H -= 2 * outer_g_H * dt
    
    gradient = np.concatenate([grad_A, grad_H])
    grad_norm = np.linalg.norm(gradient)
    
    if grad_norm < epsilon:
        print("Gradient norm below tolerance; stopping descent.")
        break
    
    # Armijo backtracking line search
    eta_current = eta * 1.05  # Initial trial step size
    ls_success = False
    for _ in range(50):  # Max line search iterations
        theta_new = theta - eta_current * gradient
        A = theta_new[:r**2].reshape(r, r)
        H = theta_new[r**2:].reshape(r, r**2)
        tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
        loss_new = np.mean(np.sum((Q_ - tilde_Q)**2, axis=0))
            
        if loss_new <= loss - alpha * eta_current * (grad_norm ** 2):
            eta = eta_current
            theta = theta_new
            ls_success = True
            break
        else:
            eta_current *= beta
        
    if not ls_success:
        theta = theta - eta * gradient  # Fallback to previous eta
        A = theta[:r**2].reshape(r, r)
        H = theta[r**2:].reshape(r, r**2)
        tilde_Q = q0[:, np.newaxis] + A @ Q_int + H @ Q2_int
        loss_new = np.mean(np.sum((Q_ - tilde_Q)**2, axis=0))
        if loss_new > loss:
            eta *= beta  # Force reduce eta if line search failed


# Detach optimal A and H from theta.
A_opt = theta[:r**2].reshape(r, r)
H_opt = theta[r**2:].reshape(r, r**2)