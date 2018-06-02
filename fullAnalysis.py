'''
This will:
 - plot error vs N form VC and HF
 - plot E(error) vs E(time) and N
        (line and violin plots)
 - plot contours of P(E(f(N))) for different N
        (line and violin plots)
 - plot contours of P(F(N)) for different N
        (line and violin plots)
 - plot D(N) for PDF convergence
'''
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
#d1 = pd.read_csv('./d1.dat')
#d2 = pd.read_csv('./collected.csv')
myhfdat = pd.read_csv('./collected.csv')
#d3 = pd.read_csv('./timedsamples.dat')
#dat = pd.concat([d1, d2, d3])
myhfdat.rename(columns={'nx7000':'hf', 'nx400':'lf2', 'nx7000_red':'lf1', 'nu':'visc', 'tnx7000':'thf', 'tnx400': 'tlf2'}, inplace=1)
myhfdat = shuffle(myhfdat)
TRUTH = np.mean(myhfdat.hf)
mylfdat = pd.read_csv('./lfcollected.csv')
mylfdat.rename(columns={'nx7000':'hf', 'nx7000_red':'lf1', 'nx400':'lf2', 'nu':'visc', 'tnx7000':'thf', 'tnx7000_red': 'tlf1', 'tnx400': 'tlf2'}, inplace=1)
#LF2M = np.mean(lfdat.lf2)
mylfdat = shuffle(mylfdat)
def cv(x, LF, hfdat, lfdat):
   #alpha = np.corrcoef(hfdat.hf, hfdat.lf2)[0][1] * np.sqrt(np.var(hfdat.hf) / np.var(hfdat.lf2))
   #cv = np.mean(hfdat.hf) + alpha * (np.mean(lfdat.lf2) - np.mean(hfdat.lf2))
   lowfis = pd.concat([hfdat, lfdat])
   alpha = np.corrcoef(hfdat.hf[:x], hfdat.lf2[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf2[:x]))
   cv = np.mean(hfdat.hf[:x]) + alpha * (np.mean(lowfis.lf2[:LF]) - np.mean(hfdat.lf2[:x]))
   return cv
cvv = np.vectorize(cv, excluded=[2,3])

def cv2(x, LF1, LF2, hfdat, lfdat):
   lowfis = pd.concat([hfdat, lfdat])
   alpha1 = np.corrcoef(hfdat.hf[:x], hfdat.lf1[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf1[:x]))
   alpha2 = np.corrcoef(hfdat.hf[:x], hfdat.lf2[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf2[:x]))
   cv = np.mean(hfdat.hf[:x]) + alpha1 * (np.mean(lowfis.lf1[:LF1]) - np.mean(hfdat.lf1[:x])) + alpha2 * (np.mean(lowfis.lf2[:LF2]) - np.mean(lowfis.lf2[:LF1]))
   return cv
cv2v = np.vectorize(cv2, excluded=[3,4])

def lf2(N_LF2, lfdat): return np.mean(lfdat.lf2[:N_LF2])
lf2v = np.vectorize(lf2, excluded=[1])

def hf(N_HF, hfdat): return np.mean(hfdat.hf[:N_HF])
hfv = np.vectorize(hf, excluded=[1])

x = np.arange(2, 10)
#plt.plot(x, (TRUTH - cv2v(x, x * 910, x * 5700, dat, lfdat))**2, label='CV')
#plt.plot(x, (TRUTH - lf2v(x, lfdat))**2, label='LF')
LFMAX = 100
dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
lfdats = np.array_split(mylfdat, mylfdat.shape[0] / LFMAX / x[-1])
BS = min(len(dats), len(lfdats))
#BS = 20
hfest = np.zeros((x.size, BS))
cvest = np.zeros((x.size, BS))
cv2est = np.zeros((x.size, BS))
for _ in range(BS):
   print(_,'/',BS)
   dat = dats[_]
   lfdat = lfdats[_]
   hfest[:, _] = hfv(x, dat)
   cvest[:, _] = cvv(x, LFMAX * x, dat, lfdat)
   cv2est[:, _] = cv2v(x, LFMAX * x, LFMAX * x, dat, lfdat)
#plt.plot(x, (TRUTH - np.mean(hfest, 1))**2, label='HF', marker='x')
#plt.plot(x, (TRUTH - np.mean(cvest, 1))**2, label='CV', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - hfest), 1), label='HF', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - cvest), 1), label='CV', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - cv2est), 1), label='CV2', marker='x')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.savefig('graphed.pdf')
plt.show()
