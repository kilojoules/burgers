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
dat = pd.read_csv('./collected.csv')
#d3 = pd.read_csv('./timedsamples.dat')
#dat = pd.concat([d1, d2, d3])
dat.rename(columns={'nx7000':'hf', 'nx400':'lf2', 'nx7000_red':'lf1', 'nu':'visc', 'tnx7000':'thf', 'tnx400': 'tlf2'}, inplace=1)
#dat = shuffle(dat)
TRUTH = np.mean(dat.hf)
lfdat = pd.read_csv('./lfcollected.csv')
lfdat.rename(columns={'nx7000':'hf', 'nx7000_red':'lf1', 'nx400':'lf2', 'nu':'visc', 'tnx7000':'thf', 'tnx7000_red': 'tlf1', 'tnx400': 'tlf2'}, inplace=1)
LF2M = np.mean(lfdat.lf2)
#lfdat = shuffle(lfdat)
def cv(N_HF, N_LF2, hfdat, lfdat):
   lowfis = np.concatenate([hfdat.lf2[:N_HF], lfdat.lf2[:N_LF2]])
   alpha_2 = np.corrcoef(hfdat.hf.values[:N_HF], hfdat.lf2.values[:N_HF])[0][1] * np.sqrt(
                         np.var(hfdat.hf.values[:N_HF], ddof=1) / np.var(hfdat.lf2.values[:N_HF], ddof=1))
   #print(alpha_2)
   cv = (np.mean(hfdat.hf.values[:N_HF]) +
         alpha_2 * (np.mean(lowfis) - np.mean(hfdat.lf2[:N_HF])))
         #alpha_2 * (np.mean(lowfis) - np.mean(hfdat.lf2[:N_HF])))
   return cv
cvv = np.vectorize(cv, excluded=[2,3])

def cv2(N_HF, N_LF1, N_LF2, hfdat, lfdat):
   alpha_1 = np.corrcoef(hfdat.hf.values[:N_HF], lfdat.lf1.values[:N_HF])[0][1] * np.sqrt(
                         np.var(hfdat.hf.values[:N_HF]) / np.var(lfdat.lf1.values[:N_LF1]))
   alpha_2 = np.corrcoef(hfdat.hf.values[:N_HF], lfdat.lf2.values[:N_HF])[0][1] * np.sqrt(
                         np.var(hfdat.hf.values[:N_HF]) / np.var(lfdat.lf2.values[:N_LF2]))
   cv = (np.mean(hfdat.hf.values[:N_HF]) +
         alpha_1 * np.mean(lfdat.lf1.values[:N_HF] - np.mean(lfdat.lf1.values[:N_LF1])) +
         alpha_2 * np.mean(lfdat.lf2.values[:N_LF1] - np.mean(lfdat.lf2.values[:N_LF2])))
   return cv
cv2v = np.vectorize(cv2, excluded=[3,4])

def lf2(N_LF2, lfdat): return np.mean(lfdat.lf2[:N_LF2])
lf2v = np.vectorize(lf2, excluded=[1])

def hf(N_HF, hfdat): return np.mean(hfdat.hf[:N_HF])
hfv = np.vectorize(hf, excluded=[1])

x = np.arange(10, 200)
#plt.plot(x, (TRUTH - cv2v(x, x * 910, x * 5700, dat, lfdat))**2, label='CV')
#plt.plot(x, (TRUTH - lf2v(x, lfdat))**2, label='LF')
LFMAX = 100
dats = np.array_split(dat, dat.shape[0] / x[-1])
lfdats = np.array_split(lfdat, lfdat.shape[0] / LFMAX / x[-1])
BS = min(len(dats), len(lfdats))
#BS = 20
hfest = np.zeros((x.size, BS))
cvest = np.zeros((x.size, BS))
for _ in range(BS):
   print(_,'/',BS)
   dat = dats[_]
   lfdat = lfdats[_]
   hfest[:, _] = hfv(x, dat)
   cvest[:, _] = cvv(x, LFMAX*x, dat, lfdat)
#plt.plot(x, (TRUTH - np.mean(hfest, 1))**2, label='HF', marker='x')
#plt.plot(x, (TRUTH - np.mean(cvest, 1))**2, label='CV', marker='x')
plt.plot(x, np.abs(TRUTH - np.mean(hfest, 1)), label='HF', marker='x')
plt.plot(x, np.abs(TRUTH - np.mean(cvest, 1)), label='CV', marker='x')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.show()
