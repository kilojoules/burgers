from completeSolver import geturec
import matplotlib.pyplot as plt
import numpy as np
#nus = [0.001, .0005, .0001, .00005, .00001, .000005, .000001,]
nus = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
nus.reverse()
#nus = [.5, .1, .05, .01, .005, 0.001, .0005]
#nus = [.5, .1, .05, .01, .005]
LW = .8
ET = 1.
nxs = []
ETs = [.96]
logg = open('logged.dat', 'w')
logg.write('nu,TOL,ET,nx')
for TOL in [1e-2]:
 for ET in ETs:
  for nu in nus:
    nx = 25
    oldqoi = None
    f, ax = plt.subplots(3)
    plt.subplots_adjust(bottom=-.8)
    l, l2 = [], []
    qois = []
    nxs = [nx]
    while True:
        print(nu, nx)
        x = np.linspace(0, np.pi, nx+2)[:-1]
        dx = x[1] - x[0]
        u = geturec(nu=nu, x=x, evolution_time=ET, timestrategy='fe', diffstrategy='3c', convstrategy='2c', n_save_t=1)[:, -1]
        l.append(ax[0].plot(x, u, label=nx, lw=LW)[0])
        gradu = u.copy()
        gradu[1:-1] = (u[2:] - u[:-2]) / ( 2 * dx)
        gradu[-1] = (u[0] - u[-2]) / ( 2 * dx)
        gradu[0] = (u[1] - u[-1]) / ( 2 * dx)
        l2.append(ax[1].plot(x, np.abs(gradu), label=nx, lw=LW)[0])
        qoi = np.max(np.abs(gradu))
        #qoi = x[np.argmax(gradu)]
        qois.append(qoi)
        if oldqoi is not None:
            print(qoi, oldqoi, np.max(gradu), np.abs(qoi - oldqoi)/qoi < 1e-3, (qoi - oldqoi)/qoi)
            if np.abs(qoi - oldqoi)/qoi < TOL:
            #if np.sqrt(np.sum((u[0::2] - oldu)**2)/oldu.size) < 1e-2:
               break
        oldqoi = qoi
        nx *=2
        if nx > 10000: nx = -1 ; break
        nxs.append(nx)
    logg.write(','.join([str(s) for s in [nu,TOL,ET,nx]]) + '\n')
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(l)))
    for ii, line in enumerate(l):
        line.set_color(colors[ii])
    for ii, line in enumerate(l2):
        line.set_color(colors[ii])
    l = ax[0].legend(ncol=3, bbox_to_anchor=[.8,1.5])
    l.set_title('Space Resolution')
    ax[0].set_ylabel('U')
    #ax[0].set_yscale('log')
    ax[1].set_ylabel(r'$|\nabla U|$')
    #ax[1].set_yscale('log')
    ax[2].set_ylabel(r'$\max(|\nabla U|)$')
    #ax[1].legend()
    ax[2].plot(nxs, qois, marker='x')
    ax[2].set_xticklabels([25 * 2 ** jj for jj in range(len(colors))])
    ax[2].set_xscale('log')
    ax[2].set_xlabel('nx')
    ax[1].set_xlabel('x')
    ax[0].set_xlabel('x')
    plt.savefig('nu%E_tol%f_ET%f.pdf'%(nu, TOL, ET), bbox_inches='tight')
    plt.clf()
    nxs.append(nx)
logg.close()
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
