from completeSolver import geturec
import matplotlib.pyplot as plt
import numpy as np
#nus = [.5, .1, .05, .01, .005, 0.001, .0005, .0001, .00005, .00001, .000005, .000001]
nus = [.5, .1, .05, .01, .005, 0.001, .0005, .00005, .00001]
#nus = [.5, .1, .05, .01, .005]
ET = 1.
nxs = []
for nu in nus:
    nx = 25
    oldqoi = None
    f, ax = plt.subplots(3)
    l, l2 = [], []
    qois = []
    while True:
        print(nu, nx)
        x = np.linspace(0, np.pi, nx+2)[:-1]
        dx = x[1] - x[0]
        u = geturec(nu=nu, x=x, evolution_time=ET, n_save_t=1)[:, -1]
        l.append(ax[0].plot(x, u, label=nx, lw=.5)[0])
        gradu = u.copy()
        gradu[1:-1] = (u[2:] - u[:-2]) / ( 2 * dx)
        gradu[-1] = (u[0] - u[-2]) / ( 2 * dx)
        gradu[0] = (u[1] - u[-1]) / ( 2 * dx)
        l2.append(ax[1].plot(x, gradu, label=nx, lw=.5)[0])
        qoi = np.max(np.abs(gradu))
        #qoi = x[np.argmax(gradu)]
        qois.append(qoi)
        if oldqoi is not None:
            print(qoi, oldqoi, np.max(gradu),np.abs(qoi - oldqoi)/qoi)
            if np.abs(qoi - oldqoi)/qoi < 1e-2:
            #if np.sqrt(np.sum((u[0::2] - oldu)**2)/oldu.size) < 1e-2:
               break
        oldqoi = qoi
        nx *=2
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(l)))
    for ii, line in enumerate(l):
        line.set_color(colors[ii])
    for ii, line in enumerate(l2):
        line.set_color(colors[ii])
    ax[0].legend()
    ax[1].legend()
    ax[2].plot(qois)
    plt.savefig('%f.pdf'%nu)
    plt.clf()
    nxs.append(nx)
print(nus, nxs)
# for tol of 5e-2,
# [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005] 
# [50,  50,  100,  400,  800,   6400,  12800]

# for tol of 1e-2,
# [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005] 
# [50,  200, 200,  1600, 1600,  12800, 25600]

# for 5e-3,
# [0.5, 0.1, 0.05, 0.01, 0.005] 
# [100, 200, 200,  1600, 3200]

# for 1e-3,
# [0.5, 0.1, 0.05, 0.01, 0.005] 
# [200, 400, 800,  3200, 6400]
