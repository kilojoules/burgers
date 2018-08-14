from completeSolver import geturec
import numpy as np
def mgrad(u, dx):
    gradu = u.copy()
    gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
    gradu[-1] = (u[0] - u[-2]) / 2 / dx
    gradu[0] = (u[1] - u[-1]) / 2 / dx
    return np.max(np.abs(gradu))
x1 = np.linspace(0, np.pi, 5002)[:-1]
dx1 = x1[1] - x1[0]
x2 = np.linspace(0, np.pi, 50002)[:-1]
dx2 = x1[1] - x1[0]
ET = 1.
np.random.seed(299)
#nus = np.random.uniform(0.00001, .00000005, 2)
nus = np.random.uniform(1e-4, 1e-5, 500)
logg = open('sampsagainagain.dat', 'w')
logg.write('nu,nx5000_4,nx50000_2,nx50000_4\n')
for nu in nus:
    print(nu)
    u1 = geturec(x=x1, nu=nu, evolution_time=ET)
    qoi1 = mgrad(u1[:, -1], dx=dx1)
    u2 = geturec(x=x2, nu=nu, evolution_time=ET, convstrategy='2c', diffstrategy='3c', timestrategy='fe')
    qoi2 = mgrad(u2[:, -1], dx=dx2)
    print('ok...')
    u3 = geturec(x=x2, nu=nu, evolution_time=ET)
    qoi3 = mgrad(u3[:, -1], dx=dx2)
    print(nu, qoi1, qoi2, qoi3)
    logg.write(','.join([str(s) for s in [nu, qoi1, qoi2, qoi3]])+'\n')
logg.close()
