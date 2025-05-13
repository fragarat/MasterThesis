import numpy as np
import opinf          # Shane's package (pip install opinf).

# Define dimensions and get test vectors.
r = 10
q = np.random.standard_normal(r)
v = np.random.standard_normal(r)
w = np.random.standard_normal(r)

# Get the initial H and check it is Kronecker symmetric.
Hcompressed = np.random.random((r, r*(r+1)//2))
H = opinf.operators.QuadraticOperator.expand_entries(Hcompressed)
assert np.allclose(H @ np.kron(q, v), H @ np.kron(v, q))

# Form Htilde and check it satisfies its property.
Htilde = np.reshape(H.reshape((r, r, r)).T, (r, r**2))
assert np.allclose(w @ H @ np.kron(q, v), v @ Htilde @ np.kron(q, w))

# Form M(q) and check it satisfies its required property.
Mq = np.einsum("ijk,j", Htilde.reshape((r, r, r)), q)
assert np.allclose(Mq @ w, Htilde @ np.kron(q, w))

# Get M(q) directly from H (not Htilde).
Mq2 = np.einsum("ijk,j->ki", H.reshape((r, r, r)), q)
assert np.allclose(Mq, Mq2)