from completeSolver import geturec
import time
import numpy as np
x = np.linspace(0, np.pi, 7000)
def d1(arr, dnu):
     outv = np.zeros(arr.size)
     outv[0] = (arr[1] - arr[0]) / dnu
     outv[1] = (arr[2] - arr[0]) / (2 * dnu)
     outv[2:-2] = (-1 * arr[4:] + 8 * arr[3:-1] - 8 * arr[1:-3] + arr[:-4]) / (12 * dnu)
     outv[-1] = (arr[-1] - arr[-2]) / dnu
     outv[-2] = (arr[-1] - arr[-3]) / (2 * dnu)
     return outv
for VISC in np.random.uniform(1e-3, 1e-5, 3000000):
#for VISC in [0.00015716675169]:
#VISC = 0.000920109564951
    x = np.linspace(0, np.pi, 7002)[:-1]
    dx = x[1] - x[0]
    u0 = np.sin(x) ** 2
    tic = time.time()
    u = geturec(x, nu=VISC, u0=u0, evolution_time=0.96)[:, -1]
    toc = time.time()
    t1 = tic - toc
    qoi1 = np.max(np.abs(np.gradient(u, dx)))
    #qoi1 = np.max(np.abs(d1(u, dx)))
    x = np.linspace(0, np.pi, 7002)[:-1]
    dx = x[1] - x[0]
    u0 = np.sin(x) ** 2
    tic = time.time()
    u = geturec(x, nu=VISC, evolution_time=0.96, convstrategy='2c', diffstrategy='3c', timestrategy='fe')[:, -1]
    toc = time.time()
    t2 = tic - toc
    qoir = np.max(np.abs(np.gradient(u, dx)))
    x = np.linspace(0, np.pi, 402)[:-1]
    u0 = np.sin(x) ** 2
    dx = x[1] - x[0]
    tic = time.time()
    u = geturec(x, nu=VISC, u0=u0, evolution_time=0.96)[:, -1]
    toc = time.time()
    t3 = tic - toc
    qoiLF = np.max(np.abs(np.gradient(u, dx)))
    #qoiLF = np.max(np.abs(d1(u, dx)))
    print(','.join([str(s) for s in [VISC, qoi1, qoir, qoiLF, t1, t2, t3]]) + '\n')
