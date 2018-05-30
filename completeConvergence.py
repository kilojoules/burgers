from completeSolver import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize as mini
nu = 0.7

# Spatial Convergence - second order - periodic
if True:
    nu = 0.1
    ET=.004
    nxs = [2 ** _ for _ in range(4, 8)]
    nxs.reverse()
    nxs = np.array(nxs)  * 2 ** 1
    nx = nxs[0] * 4
    print(nxs, nx)
    x = np.linspace(0, np.pi, nx+1)[:-1]
    BC='periodic'
    TS='fe'
    cs = '2c'
    ds = '3c'
    truex = x.copy()
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    print(truedt)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)[:-1]
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.sqrt((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    plt.plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon$')
    plt.savefig('2space%s.pdf'%BC)
    plt.clf(); plt.cla() ; plt.close()

    u = u.copy()[:, -1]
    dx = x[1] - x[0]
    dgradu = u.copy()
    dgradu[0] = (u [1] - 2 * u[0] + u[-1]) / dx ** 2
    dgradu[-1] = (u [0] - 2 * u[-1] + u[-2]) / dx ** 2
    dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    gradu = u.copy()
    gradu[0] = (u[1] - u[-1]) / 2 / dx
    gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
    gradu[-1] = (u[0] - u[-2]) / 2 / dx
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(x, u, label='u')
    ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
    ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
    #plt.plot(1 / np.abs(dgradu[:, -1] - gradu[:, -1]), label='du/dt')
    ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
    #plt.plot(np.abs(dgradu[:, -1]**4), label=r'$\frac{\partial^2u}{\partial x^2}^4$')
    #plt.plot(np.abs(gradu[:, -1]**4), label=r'$\frac{\partial u}{\partial x}^4$')
    #plt.plot((np.abs(gradu[:, -1]) + np.abs(dgradu[:, -1]))**4, label=r'$(\frac{\partial u}{\partial x} + \frac{\partial^2u}{\partial x^2})^4$')
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)[:-1]
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
#ax[a.set_yscale('log')
    plt.savefig('error/space_2_periodic.pdf', bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()
    plt.cla()
    plt.close()


    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    ax[0].plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    ax[0].plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    ax[0].plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_xlabel(r'$\Delta x$')
    ax[0].set_ylabel(r'$\epsilon$')
    #plt.plot(np.abs(dgradu[:, -1]**4), label=r'$\frac{\partial^2u}{\partial x^2}^4$')
    #plt.plot(np.abs(gradu[:, -1]**4), label=r'$\frac{\partial u}{\partial x}^4$')
    #plt.plot((np.abs(gradu[:, -1]) + np.abs(dgradu[:, -1]))**4, label=r'$(\frac{\partial u}{\partial x} + \frac{\partial^2u}{\partial x^2})^4$')
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)[:-1]
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[0].set_ylabel(r'$|\epsilon|$')
    ax[1].set_ylabel(r'$\epsilon$')
    ax[1].set_xlabel(r'x')
#ax[a.set_yscale('log')
    plt.savefig('convplots/space_2_periodic.pdf', bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()
    plt.cla()
    plt.close()





# Spatial Convergence - fourth order - periodic
if True:
    nu = 0.01
    ET=.0002
    nxs = [2 ** _ for _ in range(4, 9)]
    nxs.reverse()
    nxs = np.array(nxs)  * 2 ** 1
    nx = nxs[0] * 4
    print(nx)
    x = np.linspace(0, np.pi, nx+1)[:-1]
    BC='periodic'
    TS='rk4'
    cs = '4c'
    ds = '5c'
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    print(truedt)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        x = np.linspace(0, np.pi, nx+1)[:-1]
        print(nx)
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.sqrt((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    plt.plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    plt.legend()
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon_x$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('space4%s.pdf'%BC)
    plt.clf() ; plt.cla() ; plt.close()

    u = u.copy()[:, -1]
    dx = x[1] - x[0]
    dgradu = u.copy()
    dgradu[0] = (u [1] - 2 * u[0] + u[-1]) / dx ** 2
    dgradu[-1] = (u [0] - 2 * u[-1] + u[-2]) / dx ** 2
    dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    gradu = u.copy()
    gradu[0] = (u[1] - u[-1]) / 2 / dx
    gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
    gradu[-1] = (u[0] - u[-2]) / 2 / dx
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(x, u, label='u')
    ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
    ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
    ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel(r'$|\epsilon|$')
    ax[1].set_xlabel('x')
    ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)[:-1]
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('error/space_4_periodic.pdf', bbox_inches='tight')
    plt.clf() ; plt.cla() ; plt.close()


    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    ax[0].plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    ax[0].plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    ax[0].plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    ax[0].legend()
    ax[0].set_xlabel(r'$\Delta x$')
    ax[0].set_ylabel(r'$|\epsilon|$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)[:-1]
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_yscale('log')
    ax[1].set_xlabel('x')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('convplots/space_4_periodic.pdf', bbox_inches='tight')
    plt.clf() ; plt.cla() ; plt.close()




# Spatial Convergence - second order - dirchlet
if True:
    nu = 0.1
    ET=.004
    nxs = [2 ** _ for _ in range(6, 10)]
    nxs.reverse()
    nxs = np.array(nxs)  * 2 ** 1
    nx = nxs[0] * 4
    print(nxs, nx)
    x = np.linspace(0, np.pi, nx+1)
    BC='dirchlet'
    TS='rk4'
    cs = '2c'
    ds = '3c'
    f, ax = plt.subplots()
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.sqrt((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    plt.plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon$')
    plt.savefig('2space%s.pdf'%BC)
    plt.clf(); plt.cla() ; plt.close()

    u = u.copy()[:, -1]
    dx = x[1] - x[0]
    dgradu = u.copy()
    dgradu[0] = (u [2] - 2 * u[1] + u[0]) / dx ** 2
    dgradu[-1] = (u [-3] - 2 * u[-2] + u[-1]) / dx ** 2
    dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    gradu = u.copy()
    gradu[0] = (u[1] - u[0]) / dx
    gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
    gradu[-1] = (u[-1] - u[-2]) / dx
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(x, u, label='u')
    ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
    ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
    ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('error/space_2_%s.pdf'%BC, bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()


    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(dxs, y_2, marker='*', label='convergence', markersize=10)
    ax[0].plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='--')
    ax[0].plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls='-.')
    ax[0].plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls=':')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_xlabel(r'$\Delta x$')
    ax[0].set_ylabel(r'$|\epsilon|$')
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
    ax[1].set_xlabel('x')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('convplots/space_2_%s.pdf'%BC, bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()


# Spatial Convergence - fourth order - dirchlet
if True:
    nu = 0.001
    ET = .001
    nxs = [2 ** _ for _ in range(4, 8)]
    nxs.reverse()
    nxs = np.array(nxs) 
    nx = nxs[0] * 2 ** 4
    print(nxs, nx)
    x = np.linspace(0, np.pi, nx+1)
    f, ax = plt.subplots()
    BC='dirchlet'
    TS='rk4'
    cs = '4c'
    ds = '5c'
    print('finding truth with ', x[1] - x[0])
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    print('dt is ', truedt)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=20, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.sqrt((((u[:, -1] - trueu[0::2 **( ii + 4 ), -1])) ** 2)/nx))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 5 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 4 ), -1])**2) / nx))
        print(dxs[-1], y_2[-1])
    dxs = np.array(dxs)
    
    def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
    a = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
    b = mini(fitness, 4).x
    def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
    c = mini(fitness, 4).x
    
    plt.plot(dxs, y_2, marker='*', label='convergence', markersize=10, ls='--')
    plt.plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='-.')
    plt.plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls=':')
    plt.plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$')
    plt.legend()
    #plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\epsilon$')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('space4%s.pdf'%BC)
    plt.clf(); plt.cla() ; plt.close()

    dx = x[1] - x[0]
    u = u.copy()[:, -1]
    dgradu = u.copy()
    dgradu[0] = (u [2] - 2 * u[1] + u[0]) / dx ** 2
    dgradu[-1] = (u [-3] - 2 * u[-2] + u[-1]) / dx ** 2
    dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    gradu = u.copy()
    gradu[0] = (u[1] - u[0]) / dx
    gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
    gradu[-1] = (u[-1] - u[-2]) / dx
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(x, u, label='u')
    ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
    ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
    ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
       print('***',errs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('error/space_4_%s.pdf'%BC, bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()

    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-0.4)
    ax[0].plot(dxs, y_2, marker='*', label='convergence', markersize=10, ls='--')
    ax[0].plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='-.')
    ax[0].plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls=':')
    ax[0].plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$')
    ax[0].legend()
    ax[0].set_ylabel(r'$|\epsilon|$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\Delta x$')
    ax[1].set_xlabel('x')
    for ii, nx in enumerate(nxs):
       x = np.linspace(0, np.pi, nx+1)
       ax[1].plot(x, errs[ii], label=r'$\Delta x=%f$'%dxs[ii])
       print('***',errs[ii])
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_ylabel(r'$\epsilon$')
    plt.savefig('convplots/space_4_%s.pdf'%BC, bbox_inches='tight')
    plt.clf(); plt.cla() ; plt.close()



# Time comvergence second order - periodic
if True:
     nu = 1e-3
     ET = 1e-2
     TS = 'rk2'
     BC = 'periodic'
     x = np.linspace(0,np.pi, 2002)[:-1]
     dts = np.linspace(10, 20, 5) * 1e-6
     maxdt = geturec(x, nu=nu, evolution_time=ET, timestrategy=TS, returndt=True)
     print(dts, dts[0] / 10)
     f, ax = plt.subplots()
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=nu, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS)[:, -1]
     y = []
     errs = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=nu, n_save_t=1, timestrategy=TS)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        errs.append(np.sqrt((u-trueu)**2))
        print('-->', dt, y[-1])
     errs = np.array(errs) 
     for ii in range(dts.size):
        plt.plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     plt.legend()
     plt.yscale('log')
     plt.xlabel('x')
     plt.ylabel(r'$|u^{\Delta t} - u^{True}|$')
     plt.savefig('errPlotJustRight.pdf')
     plt.clf(); plt.cla() ; plt.close()
     dx = x[1] - x[0]
     dgradu = u.copy()                                                   
     dgradu[0] = (u [1] - 2 * u[0] + u[-1]) / dx ** 2
     dgradu[-1] = (u [0] - 2 * u[-1] + u[-2]) / dx ** 2
     dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
     gradu = u.copy() 
     gradu[0] = (u[1] - u[-1]) / 2 / dx 
     gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
     gradu[-1] = (u[0] - u[-2]) / 2 / dx
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(x, u, label='u')
     ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
     ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
     ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
     ax[0].set_xlabel('x')
     ax[1].set_xlabel('x')
     ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('error/time_2_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()


     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     z = mini(lambda x: 1e15 * (x*dts[-1] - y[-1]) ** 2, [1]).x
     a = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t^3$')[0]
     l.set_dashes([1, 1])
     c = mini(lambda x: 1e50 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
     plt.plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$')
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     plt.plot(dts, y, c='r', marker='*', label='convergnce', markersize=10)
     plt.legend()
     plt.yscale('log')
     plt.xscale('log')
     plt.ylabel(r'$\epsilon$')
     plt.xlabel(r'$\Delta t$')
     plt.savefig('time2periodic.pdf')
     plt.clf(); plt.cla() ; plt.close()

     dx = x[1] - x[0]
     dgradu = u.copy()
     dgradu[0] = (u [1] - 2 * u[0] + u[-1]) / dx ** 2
     dgradu[-1] = (u [0] - 2 * u[-1] + u[-2]) / dx ** 2
     dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
     gradu = u.copy()
     gradu[0] = (u[1] - u[-1]) / 2 / dx
     gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
     gradu[-1] = (u[0] - u[-2]) / 2 / dx
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(x, u, label='u')
     ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
     ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
     ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
     ax[0].set_xlabel('x')
     ax[1].set_xlabel('x')
     ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('error/time_2_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()

     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$')
     ax[0].plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     ax[0].plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     ax[0].plot(dts, y, c='r', marker='*', label='convergnce', markersize=10)
     ax[0].legend()
     ax[0].set_yscale('log')
     ax[0].set_xscale('log')
     ax[0].set_ylabel(r'$|\epsilon|$')
     ax[0].set_xlabel(r'$\Delta t$')
     ax[1].set_xlabel('x')
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('convplots/time_2_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()



# Time comvergence fourth order - periodic
if True:
     print('hey')
     nu = 1e-3
     ET = 1.
     TS = 'rk4'
     BC = 'periodic'
     x = np.linspace(0,np.pi, 32)[:-1]
     dts = np.linspace(1, 2, 5) * 1e-3
     maxdt = geturec(x, nu=nu, evolution_time=ET, timestrategy=TS, returndt=True, BCs=BC)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=nu, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
     y = []
     errs = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=nu, n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        errs.append(np.sqrt((u-trueu)**2))
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     #z = mini(lambda x: 1e24 * (np.exp(x)*dts[-1] - y[-1]) ** 2, [1]).x
     #plt.plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$')
     a = mini(lambda x: 1e25 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     b = mini(lambda x: 1e25 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t^3$')[0]
     l.set_dashes([1, 1])
     c = mini(lambda x: 1e24 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     plt.plot(dts, y, c='r', marker='*', label='convergence', markersize=10)
     plt.legend()
     plt.yscale('log')
     plt.xscale('log')
     plt.ylabel(r'$\epsilon_t$')
     plt.xlabel(r'$\Delta t$')
     plt.savefig('time4%s.pdf'%BC)
     plt.clf(); plt.cla() ; plt.close()
     dx = x[1] - x[0]
     dgradu = u.copy()
     dgradu[0] = (u [1] - 2 * u[0] + u[-1]) / dx ** 2
     dgradu[-1] = (u [0] - 2 * u[-1] + u[-2]) / dx ** 2
     dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
     gradu = u.copy()
     gradu[0] = (u[1] - u[-1]) / 2 / dx
     gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
     gradu[-1] = (u[0] - u[-2]) / 2 / dx
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(x, u, label='u')
     ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
     ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
     ax[0].plot(x, nu * dgradu - u * gradu, label='du/dt', ls='--')
     ax[0].set_xlabel('x')
     ax[1].set_xlabel('x')
     ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon_t$')
     plt.savefig('error/time_4_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()


     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     ax[0].plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     ax[0].plot(dts, y, c='r', marker='*', label='convergence', markersize=10)
     ax[0].legend()
     ax[0].set_yscale('log')
     ax[0].set_xscale('log')
     ax[0].set_ylabel(r'$|\epsilon|$')
     ax[0].set_xlabel(r'$\Delta t$')
     ax[1].set_xlabel('x')
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('convplots/time_4_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()



# Time comvergence fourth order, dirchlet
if True:
     nu = 0.1
     ET = 5e-1
     TS = 'rk4'
     f, ax = plt.subplots()
     x = np.linspace(0,np.pi, 31)
     dts = np.linspace(1, 2, 5) * 1e-3
     BC='dirchlet'
     maxdt = geturec(x, nu=nu, evolution_time=ET, timestrategy=TS, returndt=True)
     print(dts, dts[0]/10)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=nu, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
     y = []
     errs = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        print(dt)
        u = geturec(x, evolution_time=ET, dt=dt, nu=nu, n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        errs.append(np.sqrt((u-trueu)**2))
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     #z = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] - y[-1]) ** 2, [1]).x
     #plt.plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$')
     a = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     #b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     #l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t$')[0]
     #l.set_dashes([1, 1])
     c = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     plt.plot(dts, y, c='r', marker='*', markersize=10, label='convergence')
     plt.legend()
     plt.yscale('log')
     plt.xscale('log')
     plt.ylabel(r'$\epsilon$')
     plt.xlabel(r'$\Delta t$')
     plt.savefig('time4%s.pdf'%BC)
     plt.clf(); plt.cla() ; plt.close()

     dx = x[1] - x[0]
     dgradu = u.copy()
     dgradu[0] = (u [2] - 2 * u[1] + u[0]) / dx ** 2
     dgradu[-1] = (u [-3] - 2 * u[-2] + u[-1]) / dx ** 2
     dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
     gradu = u.copy()
     gradu[0] = (u[1] - u[0]) / dx
     gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
     gradu[-1] = (u[-1] - u[-2]) / dx
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(x, u, label='u')
     ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
     ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
     ax[0].plot(x, nu * dgradu - gradu, label='du/dt', ls='--')
     ax[0].set_xlabel('x')
     ax[1].set_xlabel('x')
     ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('error/time_4_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()

     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     ax[0].plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     ax[0].plot(dts, y, c='r', marker='*', markersize=10, label='convergence')
     ax[0].legend()
     ax[0].set_yscale('log')
     ax[0].set_xscale('log')
     ax[0].set_ylabel(r'$|\epsilon|$')
     ax[0].set_xlabel(r'$\Delta t$')
     ax[1].set_xlabel('x')
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('convplots/time_4_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()


# second order time dirchlet
if True:
     nu = 1e-3
     ET = 2e-2
     TS = 'rk2'
     BC = 'dirchlet'
     x = np.linspace(0,np.pi, 2001)
     dts = np.linspace(1, 2, 5) * 1e-6
     print(dts, dts[0]/10)
     maxdt = geturec(x, nu=nu, evolution_time=ET, timestrategy=TS, returndt=True)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=nu, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
     y = []
     errs = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=nu, n_save_t=1, timestrategy=TS, BCs=BC)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        errs.append(np.sqrt((u-trueu)**2))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        print('-->', dt, y[-1])
     #plt.plot(dts, dts, ls='--', marker='x')
     #plt.plot(dts, 1e12 * dts ** 4, ls='--', marker='^')
     z = mini(lambda x: 1e20 * (x*dts[-1] - y[-1]) ** 2, [1]).x
     a = mini(lambda x: 1e22 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
     b = mini(lambda x: 1e23 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
     c = mini(lambda x: 1e50 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
     plt.plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$', ls='-.')
     plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t^3$')[0]
     l.set_dashes([1, 1])
     plt.plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     plt.plot(dts, y, c='r', marker='*', label='convergence')
     plt.legend()
     plt.yscale('log')
     plt.xscale('log')
     plt.ylabel(r'$\epsilon$')
     plt.xlabel(r'$t$')
     plt.savefig('time2%s.pdf'%BC)
     plt.clf(); plt.cla() ; plt.close()

     dx = x[1] - x[0]
     dgradu = u.copy()
     dgradu[0] = (u [2] - 2 * u[1] + u[0]) / dx ** 2
     dgradu[-1] = (u [-3] - 2 * u[-2] + u[-1]) / dx ** 2
     dgradu[1:-1] = (u [:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
     gradu = u.copy()
     gradu[0] = (u[1] - u[0]) / dx
     gradu[1:-1] = (u[2:] - u[:-2]) / 2 / dx
     gradu[-1] = (u[-1] - u[-2]) / dx
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(x, u, label='u')
     ax[0].plot(x, dgradu, label=r'$\frac{\partial^2u}{\partial x^2}$')
     ax[0].plot(x, gradu, label=r'$\frac{\partial u}{\partial x}$')
     ax[0].plot(x, nu * dgradu - gradu, label='du/dt', ls='--')
     ax[0].set_xlabel('x')
     ax[1].set_xlabel('x')
     ax[0].legend(ncol=4, bbox_to_anchor=[1, 1.5])
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('error/time_2_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()
  
     f, ax = plt.subplots(2)
     plt.subplots_adjust(bottom=-0.4)
     ax[0].plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$', ls='-.')
     ax[0].plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
     ax[0].plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
     ax[0].plot(dts, y, c='r', marker='*', label='convergence')
     ax[0].legend()
     ax[0].set_yscale('log')
     ax[0].set_xscale('log')
     ax[0].set_ylabel(r'$|\epsilon|$')
     ax[0].set_xlabel(r'$\Delta t$')
     ax[1].set_xlabel('x')
     for ii in range(dts.size):
        ax[1].plot(x, errs[ii], label=r'$\Delta t=%f$'%dts[ii])
     ax[1].set_yscale('log')
     ax[1].legend()
     ax[1].set_ylabel(r'$\epsilon$')
     plt.savefig('convplots/time_2_%s.pdf'%BC, bbox_inches='tight')
     plt.clf(); plt.cla() ; plt.close()




## BAD ERROR PLOTS ##
# too much convergence
if True:
     nu = 1e-3
     ET = 2e-7
     TS = 'rk2'
     x = np.linspace(0,np.pi, 2002)[:-1]
     dts = np.linspace(1, 20, 5) * 1e-9
     maxdt = geturec(x, nu=nu, evolution_time=ET, timestrategy=TS, returndt=True)
     print(dts, dts[0] / 10)
     if np.max(dts) > maxdt: raise(Exception("Bad time")) ; quit()
     trueu = geturec(x, nu=nu, evolution_time=ET, dt=dts[0] / 10., n_save_t=1, timestrategy=TS)[:, -1]
     y = []
     errs = []
     #dts = np.array([2e-6, 1e-6, 8e-7, 5e-7, 2e-7, 1e-7])  * 1e2
     for dt in dts:
        u = geturec(x, evolution_time=ET, dt=dt, nu=nu, n_save_t=1, timestrategy=TS)[:, -1]
        #y.append(np.max(np.abs(u - trueu)))
        y.append(np.sqrt(np.sum((u - trueu) ** 2)))
        errs.append(np.abs(u-trueu))
        print('-->', dt, y[-1])
     errs = np.array(errs)
     for ii in range(dts.size):
        plt.plot(x, errs[ii], label=r'$\Delta t=%.3E$'%dts[ii], lw=0.1)
     plt.legend()
     plt.yscale('log')
     plt.xlabel('x')
     plt.ylim(1e-18, 1e-12)
     plt.ylabel(r'$|u^{\Delta t} - u^{True}|$')
     plt.savefig('errPlotTooMuch.pdf')
     plt.clf(); plt.cla() ; plt.close()

     if True:
         z = mini(lambda x: 1e15 * (x*dts[-1] - y[-1]) ** 2, [1]).x
         plt.plot(dts, dts * z, c='k', lw=3, label=r'$\Delta t$')
         a = mini(lambda x: 1e18 * (np.exp(x)*dts[-1] ** 2 -y[-1])**2, [20]).x
         plt.plot(dts, dts ** 2 * np.exp(a), c='k', lw=3, ls='--', label=r'$\Delta t^2$')
         b = mini(lambda x: 1e20 * (np.exp(x)*dts[-1] ** 3 -y[-1]) ** 2, [50]).x
         l = plt.plot(dts, dts **3 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t^3$')[0]
         l.set_dashes([1, 1])
         c = mini(lambda x: 1e50 * (np.exp(x)*dts[-1] ** 4 -y[-1]) ** 2, [70]).x
         f, ax = plt.subplots(2)
         plt.subplots_adjust(bottom=-0.4)
         ax[0].plot(dts, dts ** 1 * np.exp(z), c='k', lw=3, ls=':', label=r'$\Delta t^4$')
         ax[0].plot(dts, dts ** 2 * np.exp(b), c='k', lw=3, ls='--', label=r'$\Delta t^4$')
         ax[0].plot(dts, dts ** 4 * np.exp(c), c='k', lw=3, ls='-.', label=r'$\Delta t^4$')
         ax[0].plot(dts, y, c='r', marker='*', label='convergence')
         ax[0].legend()
         ax[0].set_yscale('log')
         ax[0].set_xscale('log')
         ax[0].set_ylabel(r'$|\epsilon|$')
         ax[0].set_xlabel(r'$\Delta t$')
         for ii in range(len(dts)):
            ax[1].plot(x, errs[ii], label=r'$\Delta t = dxs[ii]$')
         ax[1].legend()
         ax[1].set_yscale('log')
         ax[1].set_xlabel('x')
         ax[1].set_ylabel(r'$\epsilon$')
         plt.savefig('toomuch2.pdf', bbox_inches='tight')
         plt.clf(); plt.cla() ; plt.close()
    

# not enough convergence
if True:
    nu = 0.001
    ET=4
    nxs = [2 ** _ for _ in range(3, 7)]
    nxs.reverse()
    nxs = np.array(nxs) 
    nx = nxs[0] * 4
    x = np.linspace(0, np.pi, nx+1)[:-1]
    BC='periodic'
    TS='fe'
    cs = '2c'
    ds = '3c'
    trueu = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds)
    truedt = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, convstrategy=cs, diffstrategy=ds, returndt=True)
    print("hey nxs and truedt are ", nxs, truedt)
    y_2 = []
    dxs = []
    errs = []
    for ii, nx in enumerate(nxs):
        print(nx)
        x = np.linspace(0, np.pi, nx+1)[:-1]
        dxs.append(x[1] - x[0])
        u = geturec(x=x, nu=nu, evolution_time=ET, n_save_t=1, timestrategy=TS, BCs=BC, dt=truedt, convstrategy=cs, diffstrategy=ds)
        errs.append(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1]))
        #y_2.append(np.max(np.abs(u[:, -1] - trueu[0::2 **( ii + 2 ), -1])))
        #y_2.append(np.sum(np.abs((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])) / nx))
        y_2.append(np.sqrt(np.sum((u[:, -1] - trueu[0::2 **( ii + 2 ), -1])**2) / nx))
        plt.plot(x, errs[-1], label=r'$\Delta x=%.3E$'%dxs[ii])
    dxs = np.array(dxs)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel(r'$|u^{\Delta x}(x) - u^{True}(x)|$')
    plt.savefig('errPlotTooLittle.pdf')
    plt.clf(); plt.cla() ; plt.close()
    
    if True:
        def fitness(a): return 1e25 * np.sum((np.exp(a) * dxs[0] - y_2[0]) ** 2)
        a = mini(fitness, 4).x
        def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 2 - y_2[0]) ** 2)
        b = mini(fitness, 4).x
        def fitness(a): return 1e29 * np.sum((np.exp(a) * dxs[0] ** 4 - y_2[0]) ** 2)
        c = mini(fitness, 4).x
        
        f, ax = plt.subplots(2)
        plt.subplots_adjust(bottom=-.4)
        ax[0].plot(dxs, y_2, marker='*', label='convergence', markersize=10)
        ax[0].plot(dxs, np.exp(c) * dxs ** 4, c='k', label=r'$\Delta X^4$', ls='-.')
        ax[0].plot(dxs, np.exp(b) * dxs ** 2, c='k', label=r'$\Delta X^2$', ls=':')
        ax[0].plot(dxs, np.exp(a) * dxs, c='k', label=r'$\Delta X$', ls='--')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[0].set_xlabel(r'$\Delta x$')
        ax[0].set_ylabel(r'$|\epsilon|$')

        for ii, nx in enumerate(nxs):
           print(nx)
           x = np.linspace(0, np.pi, nx+1)[:-1]
           ax[1].plot(x, errs[ii], label=r'$\Delta x = dxs[ii]$')
        ax[1].legend()
        ax[1].set_yscale('log')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel(r'$\epsilon$')

        plt.savefig('toolittle2.pdf', bbox_inches='tight')
