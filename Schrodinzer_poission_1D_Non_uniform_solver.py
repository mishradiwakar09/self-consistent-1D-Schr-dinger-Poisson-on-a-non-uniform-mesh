# sc_sp_1d_nonuniform.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt

# Physical constants
hbar = 1.054571817e-34
m0   = 9.10938356e-31
q    = 1.602176634e-19
eps0 = 8.8541878128e-12

# Problem parameters
L = 10e-9
N = 401
m_eff = 0.067 * m0
eps_r = 12.0
N_electrons = 2
max_iter = 300
tol_eV = 1e-5
mixing = 0.2

# Non-uniform mesh (tanh clustering)
xu = np.linspace(-1,1,N)
cluster_strength = 2.5
x = (L/2) * (1 + np.tanh(cluster_strength * xu) / np.tanh(cluster_strength))
x[0] = 0.0; x[-1] = L
h = np.diff(x)

# integration weights (trapezoidal-like)
w = np.zeros(N)
w[0] = (x[1]-x[0])/2
w[-1] = (x[-1]-x[-2])/2
w[1:-1] = (x[2:] - x[:-2]) / 2

# Build second-derivative sparse matrix on non-uniform grid
rows = []
cols = []
data = []
for i in range(1, N-1):
    h_im = x[i] - x[i-1]
    h_i  = x[i+1] - x[i]
    a = 2.0 / (h_im * (h_im + h_i))
    b = -2.0 / (h_im * h_i)
    c = 2.0 / (h_i * (h_im + h_i))
    rows += [i, i, i]
    cols += [i-1, i, i+1]
    data += [a, b, c]
D2 = sp.csr_matrix((data, (rows, cols)), shape=(N,N))

# Poisson matrix with Dirichlet BCs
rows = []
cols = []
data = []
for i in range(1, N-1):
    h_im = x[i] - x[i-1]
    h_i  = x[i+1] - x[i]
    A_im = 1.0 / h_im
    A_i  = 1.0 / h_i
    center = -(A_im + A_i)
    rows += [i, i, i]
    cols += [i-1, i, i+1]
    data += [A_im, center, A_i]
P = sp.csr_matrix((data, (rows, cols)), shape=(N,N))
# Dirichlet BC rows
P = P.tolil()
P[0,:] = 0; P[:,0] = 0; P[0,0] = 1
P[-1,:] = 0; P[:,-1] = 0; P[-1,-1] = 1
P = P.tocsr()

prefactor = - (hbar**2) / (2.0 * m_eff)

# initial potential (J)
V = np.zeros(N)
V_old = V.copy()

converged = False

for it in range(1, max_iter+1):
    # Hamiltonian (dense small matrix)
    H = prefactor * D2.todense()
    H = np.array(H, dtype=float)
    big = 1e20
    H[0,:] = 0; H[:,0] = 0; H[0,0] = big
    H[-1,:] = 0; H[:,-1] = 0; H[-1,-1] = big
    H += np.diag(V)
    # eigen-decomposition for lowest states
    try:
        evals, evecs = la.eigh(H)
    except Exception:
        evals, evecs = np.linalg.eig(H)
        idx = np.argsort(evals.real)
        evals = evals[idx].real
        evecs = evecs[:, idx].real
    evals_eV = evals / q
    psi = evecs
    # normalize eigenvectors with integration weights
    for n in range(psi.shape[1]):
        norm = np.sqrt(np.sum(np.abs(psi[:,n])**2 * w))
        psi[:,n] = psi[:,n] / norm

    # occupations (T=0, spin degeneracy)
    occ = np.zeros_like(evals)
    nstates_occ = int(np.ceil(N_electrons / 2.0))
    occ[:nstates_occ] = 2.0
    if N_electrons % 2 == 1:
        occ[nstates_occ-1] = 1.0

    # density (electrons per meter)
    density = np.zeros(N)
    for n in range(len(evals)):
        density += occ[n] * (np.abs(psi[:,n])**2)
    rho = - q * density

    # Poisson RHS
    rhs = - rho / (eps0 * eps_r)
    rhs[0] = 0.0; rhs[-1] = 0.0
    phi = spla.spsolve(P, rhs)    # volts
    V_new = q * phi               # Joules

    V = (1.0 - mixing) * V_old + mixing * V_new
    V_old = V.copy()

    delta_eV = np.max(np.abs((V - V_new))) / q
    if it % 10 == 0 or it == 1:
        print(f"Iter {it:3d}: E0 = {evals_eV[0]:10.6f} eV, max dV = {delta_eV:.3e} eV")
    if delta_eV < tol_eV:
        print(f"Converged in {it} iterations. E0 = {evals_eV[0]:.6f} eV")
        converged = True
        break

if not converged:
    print("Warning: did not converge within max_iter")

# plot results
phi_eV = V / q
prob0 = np.abs(psi[:,0])**2
density_nm = density * 1e-9

plt.figure(figsize=(8,5))
plt.plot(x*1e9, phi_eV, label='Potential (eV)')
scale = (np.max(phi_eV) - np.min(phi_eV))
psi_scaled = prob0 * scale + np.min(phi_eV)
plt.plot(x*1e9, psi_scaled, '--', label='|psi_0|^2 (scaled)')
plt.plot(x*1e9, density_nm, ':', label='Electron density (e / nm)')
plt.xlabel('x (nm)')
plt.ylabel('Value')
plt.title('Self-consistent Schrodinger-Poisson (1D, non-uniform mesh) - Python')
plt.legend()
plt.grid(True)
plt.show()

print("Lowest three eigenvalues (eV):", evals_eV[:3])
