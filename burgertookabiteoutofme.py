import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, cumtrapz
from completeSolver import geturec

# x points
nx = 7000
x = np.linspace(0, np.pi, nx+2)[:-1]
dx = x[1] - x[0]

# prescribe u dt / dx = 0.01
evolution_time = 1.

# baseline viscosity
nu = 1e-5

n_save_t = 20
u_record = np.load('./urec%f.npy'%nu)
#u_record = geturec(nu=nu, x=x, evolution_time=evolution_time)
#np.save('urec%f'%nu, u_record)
dt = geturec(nu=nu, x=x, evolution_time=evolution_time, returndt=True)
nt = int(evolution_time / dt)
divider = int(nt / float(n_save_t))


# SVD the covariance matrix
psi, D, phi = np.linalg.svd(u_record) # P D Q is the approximation

S = np.zeros((nx+1, u_record.shape[1]))
mm = min(nx+1, u_record.shape[1])
S[:mm, :mm] = np.diag(D)
assert(np.allclose(u_record, np.dot(psi, np.dot(S, phi)))) # check that a = P D Q
Q = np.dot(S, phi.T)
phi = phi.T


# choose # of modes to keep
MODES = 2

def genterms(MODES):
    TERM1 = np.zeros((MODES, MODES))
    TERM2 = np.zeros((MODES, MODES, MODES))
    for ii in range(MODES):
        for kk in range(MODES):
            TERM1[ii, kk] = simps(psi[:, kk] *  np.gradient(np.gradient(psi[:, ii], dx), dx), dx=dx)
            for jj in range(MODES):
                TERM2[ii, jj, kk] = simps(psi[:, kk] * psi[:, ii] * np.gradient(psi[:, jj], dx))
    return TERM1, TERM2
TERM1, TERM2 = genterms(MODES)

def dqdt(a, kk, nu=nu):
    # calculate first term
    t1 = 0
    for ii in range(a.shape[0]):
        t1 += nu * a[ii] * TERM1[ii, kk]
    t2 = 0
    for ii in range(a.shape[0]):
        for jj in range(a.shape[0]):
            t2 += a[jj] * a[ii] * TERM2[ii, jj, kk]
    return t1 - t2

# record weights associated with first time :)
a0 = Q[:MODES, 0].copy()

def evolve(a,nu=nu, dt=dt):
    n_records = 0
    u_red_record = np.zeros((nx+1, int(nt / divider) + 1))
    for _ in range(nt):
        for ii in range(MODES):
            na = a.copy()
            a[ii] = na[ii] + dqdt(na, ii, nu) * dt
            #del na
        if _ % (divider) == 0:
            u_red_record[:, n_records] = np.dot(a, psi[:, :MODES].T).copy()
            n_records += 1
    return a, u_red_record

def ROM_go(PREDICT):
    a, u_surr = evolve(a0, nu=PREDICT)
    u = u_surr[:, -1]
    gradu = u.copy()
    gradu[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    gradu[0] = (u[1] - u[-1]) / (2 * dx)
    gradu[-1] = (u[0] - u[-2]) / (2 * dx)
    return np.max(np.abs(gradu))
