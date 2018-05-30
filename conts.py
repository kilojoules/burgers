import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
dat = pd.read_csv('./timedsamples.dat')
dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
dat = dat.convert_objects(convert_numeric=True)
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = pd.read_csv('./mysampledsamples.dat')
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = dat[dat.visc > 1e-6]
print(dat.shape)

LW = .1
ALPHA = 0.5
TEST = 52
BS = int(dat.shape[0] / ((TEST-2) * 6))
LF = 52
lf_tests = np.linspace(TEST, (TEST-2)*6, LF).astype(int)
dats = np.array_split(dat, BS)
x = np.array([int(s) for s in np.arange(10, TEST, 2)])
X = np.zeros(LF * len(x))
Y = np.zeros(LF * len(x))
Z = np.zeros(LF * len(x))
epsh = np.zeros((x.size, BS))
epscv = np.zeros((x.size, BS, LF))
for _ in range(BS):
    dat = dats[_]
    def mch(x):
         return (np.mean(dat.hf[TEST:]) - np.mean(dat.hf.values[:x])) ** 2, np.var(dat.hf.values[:x]) / x, np.mean(dat.hf.values[:x]), np.mean(dat.hf[TEST:])
    mch = np.vectorize(mch)
    
    def cv2(x, n):
        alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
        if n < x: raise IOError(":(")
        m = (np.mean(dat.hf[TEST:]) - (
            np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf[:n]) - np.mean(dat.lf.values[:x])))) ** 2
        std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf))) * (np.var(dat.lf) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf) * np.var(dat.hf[:x])))
        return m, std, alpha, np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x]))
    #cv2 = np.vectorize(cv2, )
            
    print (_, dat.shape)
    ev_mch = mch(x)
    epsh[:, _] = np.sqrt(ev_mch[0])
    for kk, sn in enumerate(lf_tests):
      for ii in range(len(x)):
         ev_cv2 = cv2(x[ii], sn)
         epscv[ii, _, kk] = np.sqrt(ev_cv2[0])
         Z[kk*ii] += np.sqrt(ev_cv2[0])
         Y[kk*ii] = sn
         X[kk*ii] = x[ii]
         if len(ev_cv2[0][np.isnan(ev_cv2[0])]) > 0: hey
Z /= BS
#Z = Z.reshape(X.size, Y.size)

xi = np.linspace(X.min(),X.max(),1000)
yi = np.linspace(Y.min(),Y.max(),1000)

# Z is a matrix of x-y values
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')

np.save('zi', zi)
np.save('xi', xi)
np.save('yi', yi)
plt.contourf(xi, yi, np.log(zi), 1000, vmax=0)
c = plt.colorbar()
c.set_label('log(error, CV)')
plt.savefig('contours.png')
plt.show()
