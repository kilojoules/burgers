# Load modules
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.integrate import simps, cumtrapz
from completeSolver import geturec

# Let's evolve burgers' equation in time

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

# Calculate the weight functions
#Q = np.dot(S[:, :MODES], phi[:, :MODES].T)

# Generate expensive terms
@nb.jit
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

print("GENERATED!")

@nb.jit
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

@nb.jit
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

a, u_red_record_g6 = evolve(a0)
print('evolved!')

#MODES = 3
ubar_red = np.dot(Q[:MODES, :].T, psi[:, :MODES].T).T

@nb.jit
def maxgrad(u): return np.max(np.abs(u[:, -1]))

def shockloc(u_record, x=x): 
    a = u_record[:, -1] - 1. - 1e-4
    return x[np.where(np.sign(a[:-1]) != np.sign(a[1:]))[0] + 1][-1]
    #return x[1:][np.diff(np.sign(u_record[:, -1] - np.mean(u_record[:, -1]))) == -2]
#def shockloc(u_record, x=x): return x[1:][np.diff(np.sign(u_record[:, -1] - np.mean(u_record[:, -1]))) == -2]

END = -1
PLOT = 1
if PLOT:

    # Generate flow field using forward method
    plt.plot(x, u_red_record_g6[:, -1], c='k', label="ROM Evolved", lw=3)
    plt.plot(x, u_record[:, 0], ls='--', c='fuchsia', label="True Initial", lw=3)
    plt.plot(x, u_record[:, -1], c='purple', label="True Evolved", ls='--', lw=3)
    plt.legend().get_frame().set_alpha(0.5)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
    #hey
    #plt.savefig('./compared.pdf')
    #savefig('./compared.pdf')
    
    # predict a new viscosity
    PREDICT = 1e-3
    a, u_surr = evolve(a0, nu=PREDICT)
    plt.plot(x, u_record[:, -1], c='gray', label="Base Truth", lw=3)
    plt.plot(x, u_red_record_g6[:, END], label="Base ROM", lw=3)
    plt.plot(x, geturec(nu=PREDICT, x=x, evolution_time=evolution_time)[:, -1], c='k', label="Viscous Truth", lw=3)
    plt.plot(x, u_surr[:, -1], c='purple', label="Viscous ROM", lw=3)
    plt.legend(ncol=2)
    plt.ylabel("u")
    plt.xlabel("x")
    plt.show()
    
    plt.plot(cumtrapz(D) / max(cumtrapz(D)))
    plt.ylabel("Cumulative Normalied Eigen Value")
    plt.xlabel("Mode")
    plt.xlim(0, 100)
    plt.show()
    
hey
nus = np.array([1, 5e-1, 2e-1, 1e-1, 1e-2, 1e-3, 0])
plt.plot(nus, np.array([maxgrad(geturec(nu)) for nu in nus]), c='r', marker='x')
plt.plot(nus, np.array([maxgrad(evolve(a0, nu)[1]) for nu in nus]), c='b', marker='x')
plt.ylabel(r"max($\nabla |u|$)")
plt.xlabel(r"$\nu$")
plt.show()


ns = np.array([500, 1000, 1500, 2000, 2500, 5000, 10000, 20000, 30000, 40000, 60000, 80000])
dxs = []
grads = []
for n in ns:
   x = np.linspace(0, 4 * np.pi, n)
   dxs.append(x[1] - x[0])
   grads.append(shockloc(geturec(x=x), x=x))
plt.plot(dxs, grads, marker='x')
plt.ylabel("Location of the Shock")
plt.xlabel("dx")
plt.show()
