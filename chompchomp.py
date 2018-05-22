# Load modules
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.integrate import simps, cumtrapz
from completeSolver import geturec

def d1(u, dx):
    gradu = u.copy()
    gradu[2:-2] = ((1./12.) * u[:-4] + (-2./3.) * u[1:-3] + (2./3.) * u[3:-1] + (-1./12.) * u[4:]) / dx
    gradu[0] = (u[1] - u[0]) / dx
    gradu[1] = (u[2] - u[0]) / 2 / dx
    gradu[-2] = (u[-1] - u[-3]) / 2 / dx
    gradu[-1] = (u[-1] - u[-2]) / dx
    return gradu
def d2(u, dx):
    dgradu = u.copy()
    dgradu[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    dgradu[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dx ** 2
    dgradu[0] = (u[0] - 2 * u[1] + u[2]) / dx ** 2
    return dgradu

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
dt = geturec(nu=nu, x=x, evolution_time=evolution_time, returndt=True)
nt = int(evolution_time / dt)
divider = int(nt / float(n_save_t))
u_record = geturec(nu=nu, x=x, evolution_time=evolution_time, n_save_t=100)
np.save('urec%f'%nu, u_record)
#u_record = np.load('./urec%f.npy'%nu)
ubar = u_record.mean(1)
u_record -= ubar.reshape(-1, 1)
print('flow fielded')

# SVD the covariance matrix
psi, D, phi = np.linalg.svd(u_record, full_matrices=False)
for ii in range(psi.shape[1]): 
     psi[:, ii] /= np.sqrt(dx)
     phi[ii, :] *= np.sqrt(dx)
assert(np.allclose(u_record, np.dot(psi * D, phi)))
#Q = np.dot(S.T, psi.T)
#phi = phi.T


# choose # of modes to keep
MODES = 10

# Calculate the weight functions
#Q = np.dot(S[:, :MODES], phi[:, :MODES].T)

# Generate expensive terms
#@nb.jit
EDDYVISC = .001
def genterms(MODES):
    bk1 = np.zeros(MODES)
    bk2 = np.zeros(MODES)
    bk3 = np.zeros(MODES)
    lik1 = np.zeros((MODES, MODES))
    lik2 = np.zeros((MODES, MODES))
    lik3 = np.zeros((MODES, MODES))
    nijk = np.zeros((MODES, MODES, MODES))
    for kk in range(MODES):
        bk1[kk] = simps(psi[:, kk] * (-1 * ubar * d1(ubar, dx)), dx=dx)
        bk2[kk] = simps(psi[:, kk] * d2(ubar, dx), dx=dx)
        bk3[kk] = simps(psi[:, kk] * EDDYVISC * d2(ubar, dx), dx=dx)
        for ii in range(MODES):
            lik1[ii, kk] = simps(psi[:, kk] * - 1 * ubar * d1(psi[:, kk], dx) - psi[:, kk] * d1(ubar, dx), dx=dx)
            lik2[ii, kk] = simps(psi[:, kk] * d2(psi[:, ii], dx), dx=dx)
            lik3[ii, kk] = simps(psi[:, kk] * EDDYVISC * d2(psi[:, ii], dx), dx=dx)
            for jj in range(MODES):
                nijk[ii, jj, kk] = simps(psi[:, kk] * psi[:, ii] * d1(psi[:, jj], dx), dx=dx)
    return bk1, bk2, bk3, lik1, lik2, lik3, nijk
bk1, bk2, bk3, lik1, lik2, lik3, nijk = genterms(MODES)
TERM1A = lik1 + lik3
TERM1B = lik2
TERM2 = nijk

print("GENERATED!")

def dqdt(a, nu):
    # calculate first term
    TERM0 = bk1 + nu * bk2 + bk3
    t1 = 0
    for ii in range(a.shape[0]):
        t1 += a[ii] * (TERM1A[ii, :] + nu * TERM1B[ii,:])
    t2 = 0
    for ii in range(a.shape[0]):
        for jj in range(a.shape[0]):
            t2 += a[jj] * a[ii] * (TERM2[ii, jj, :])
    return TERM0 + t1 - t2

# record weights associated with first time :)
a0 = phi[:MODES, 0] * D[:MODES]

#@nb.jit
def evolve(a,nu=nu, dt=dt):
    n_records = 0
    a_record = np.zeros((MODES, int(nt / divider) + 3))
    a_record[:, 0] = a.copy()
    for _ in range(nt):
        #for ii in range(MODES):
        na = a.copy()
        #ahalf = na + dqdt(na, nu) * dt/2
        #a = na + (dqdt(ahalf, nu) + dqdt(na, nu)) * dt /2
        a = na + dqdt(na, nu) * dt
            #del na
        if _ % (divider) == 0:
            a_record[:, n_records] = a.copy()
            n_records += 1
    a_record[:, -1] = a.copy()
    return a_record

a = evolve(a0)
print('evolved!')

#MODES = 3
ubar_red = np.dot(psi[:, :MODES] * D[:MODES], phi[:MODES, :])

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
    plt.plot(x, np.dot(psi[:, :MODES], a0), c='k', label="ROM Initial", lw=3)
    plt.plot(x, np.dot(psi[:, :MODES], a[:, -1]), c='k', label="ROM Evolved", lw=3)
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
