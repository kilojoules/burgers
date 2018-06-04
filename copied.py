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
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
#d1 = pd.read_csv('./d1.dat')
#d2 = pd.read_csv('./collected.csv')
myhfdat = pd.read_csv('./collected.csv')
#d3 = pd.read_csv('./timedsamples.dat')
#dat = pd.concat([d1, d2, d3])
myhfdat.rename(columns={'nx7000':'hf', 'nx400':'lf2', 'nx7000_red':'lf1', 'tnx7000_red':'tlf1', 'nu':'visc', 'tnx7000':'thf', 'tnx400': 'tlf2'}, inplace=1)
myhfdat = shuffle(myhfdat)
TRUTH = np.mean(myhfdat.hf)
mylfdat = pd.read_csv('./lfcollected.csv')
mylfdat.rename(columns={'nx7000':'hf', 'nx7000_red':'lf1', 'nx400':'lf2', 'nu':'visc', 'tnx7000':'thf', 'tnx7000_red': 'tlf1', 'tnx400': 'tlf2'}, inplace=1)
#LF2M = np.mean(lfdat.lf2)
omega_1 = np.max(myhfdat.thf)
omega_2 = np.mean(mylfdat.tlf1)
omega_3 = np.mean(mylfdat.tlf2)
rho_1_2 = np.corrcoef(myhfdat.hf, myhfdat.lf1)[0][1]
rho_1_3 = np.corrcoef(myhfdat.hf, myhfdat.lf2)[0][1]

r_cv = np.sqrt(omega_1 * rho_1_3**2 / (omega_3 * (1 - rho_1_3**2)))
#r_cv = 500
r_cv2_1 = np.sqrt(omega_1 * (rho_1_2**2 - rho_1_3**2) / (omega_2 * (1 - rho_1_2**2)))
r_cv2_2 = np.sqrt(omega_1 * rho_1_3**2 / (omega_3 * (1 - rho_1_2**2)))
mylfdat = shuffle(mylfdat)
def cv(x, LF, hfdat, lfdat):
   #alpha = np.corrcoef(hfdat.hf, hfdat.lf2)[0][1] * np.sqrt(np.var(hfdat.hf) / np.var(hfdat.lf2))
   #cv = np.mean(hfdat.hf) + alpha * (np.mean(lfdat.lf2) - np.mean(hfdat.lf2))
   lowfis = pd.concat([hfdat, lfdat])
   alpha = np.corrcoef(hfdat.hf[:x], hfdat.lf2[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf2[:x]))
   cv = np.mean(hfdat.hf[:x]) + alpha * (np.mean(lowfis.lf2[:LF]) - np.mean(hfdat.lf2[:x]))
   return cv
cvv = np.vectorize(cv, excluded=[2,3])
def cvtime(x, LF, hfdat, lfdat):
    return(np.sum([hfdat.thf[:x].sum(),  hfdat.tlf2[:x].sum(), lfdat.tlf2.values[:LF].sum()]))
cvtimev = np.vectorize(cvtime, excluded=[2, 3])

def cv2(x, LF1, LF2, hfdat, lfdat):
   lowfis = pd.concat([hfdat, lfdat])
   alpha1 = np.corrcoef(hfdat.hf[:x], hfdat.lf1[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf1[:x]))
   alpha2 = np.corrcoef(hfdat.hf[:x], hfdat.lf2[:x])[0][1] * np.sqrt(np.var(hfdat.hf[:x]) / np.var(hfdat.lf2[:x]))
   cv = np.mean(hfdat.hf[:x]) + alpha1 * (np.mean(lowfis.lf1[:LF1]) - np.mean(hfdat.lf1[:x])) + alpha2 * (np.mean(lowfis.lf2[:LF2]) - np.mean(lowfis.lf2[:LF1]))
   return cv
cv2v = np.vectorize(cv2, excluded=[3, 4])

def cv2time(x, LF1, LF2, hfdat, lfdat):
   return np.sum([hfdat.thf[:x].sum(), hfdat.tlf1[:x].sum(), hfdat.tlf2[:x].sum(), lfdat.tlf1[:LF1].sum(), lfdat.tlf2[:LF2].sum()])
cv2timev = np.vectorize(cv2time, excluded=[3, 4])

def lf(LF, lfdat): return np.mean(lfdat.lf1[:LF])
lfv = np.vectorize(lf, excluded=[1])

def lftime(LF, lfdat): return np.sum(lfdat.tlf1[:LF])
lftimev = np.vectorize(lftime, excluded=[1])

def hf(N_HF, hfdat): return np.mean(hfdat.hf[:N_HF])
hfv = np.vectorize(hf, excluded=[1])

def hftime(HF, hfdat): return np.sum(hfdat.thf[:HF])
hftimev = np.vectorize(hftime, excluded=[1])

#x = np.arange(2, 7)
#xcv2 = x.copy()
#LFMAX = max([r_cv, r_cv2_1, r_cv2_2]) 
#dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
#lfdats = np.array_split(mylfdat, mylfdat.shape[0] / LFMAX / x[-1])
#BS = min(len(dats), len(lfdats))
#cv2t = np.zeros((x.size, BS))
#cv2est = np.zeros((x.size, BS))
#for _ in range(BS):
#   print(_,'/',BS, 'cv2')
#   dat = dats[_]
#   lfdat = lfdats[_]
#   cv2est[:, _] = cv2v(x, (r_cv2_1 * x).astype(int), LFMAX.astype(int) * x, dat, lfdat)
#   cv2t[:, _] = cv2timev(x, (r_cv2_1 * x / x[-1]).astype(int), LFMAX.astype(int) * x, dat, lfdat)
x = np.arange(2, 50)
xcv = x.copy()
LFMAX = max([r_cv, r_cv2_1, r_cv2_2]) 
dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
LFMAX = r_cv
dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
lfdats = np.array_split(mylfdat, mylfdat.shape[0] / LFMAX / x[-1])
BS = min(len(dats), len(lfdats))
cvest = np.zeros((x.size, BS))
cvt = np.zeros((x.size, BS))
for _ in range(BS):
   print(_,'/',BS, 'cv')
   dat = dats[_]
   lfdat = lfdats[_]
   cvt[:, _] = cvtimev(x, (r_cv * x / x[-1]).astype(int), dat, lfdat)
   cvest[:, _] = cvv(x, (r_cv * x).astype(int), dat, lfdat)
x = np.arange(2, 1000)
dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
lfdats = np.array_split(mylfdat, mylfdat.shape[0] / x[-1])
BS = min(len(dats), len(lfdats))
hfest = np.zeros((x.size, BS))
hft = np.zeros((x.size, BS))
lfest = np.zeros((x.size, BS))
lft = np.zeros((x.size, BS))
for _ in range(BS):
   print(_,'/',BS, 'hflf')
   dat = dats[_]
   lfdat = lfdats[_]
   hft[:, _] = hftimev(x, dat)
   lft[:, _] = lftimev(x, lfdat)
   hfest[:, _] = hfv(x, dat)
   lfest[:, _] = lfv(x, lfdat)

#plt.plot(x, (TRUTH - np.mean(hfest, 1))**2, label='HF', marker='x')
#plt.plot(x, (TRUTH - np.mean(cvest, 1))**2, label='CV', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - hfest), 1), label='HF', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - lfest), 1), label='LF', marker='x')
plt.plot(xcv, np.mean(np.abs(TRUTH - cvest), 1), label='CV', marker='x')
#plt.plot(xcv2, np.mean(np.abs(TRUTH - cv2est), 1), label='CV2', marker='x')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.savefig('convPlot.pdf')
plt.clf()

plt.plot(hft.mean(1), np.mean(np.abs(TRUTH - hfest), 1), label='HF', marker='x')
plt.plot(lft.mean(1), np.mean(np.abs(TRUTH - lfest), 1), label='LF', marker='x')
plt.plot(cvt.mean(1), np.mean(np.abs(TRUTH - cvest), 1), label='CV', marker='x')
#plt.plot(cv2t.mean(1), np.mean(np.abs(TRUTH - cv2est), 1), label='CV2', marker='x')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('cpu time (sec)')
plt.ylabel(r'$|\mathbb{E}[f_0(\nu)] - Q|$')
plt.legend()
plt.savefig('timePlot.pdf')
plt.clf()

BINS = 300

SAMPS = [50]
while SAMPS[-1] * 2 < hfest.shape[0]:
   SAMPS.append(SAMPS[-1] * 2)
lastdist = np.histogram(hfest[:int(SAMPS[0] / 2), -1], BINS, normed=True)[0]
hfDs = []
for n in SAMPS:
   dist = np.histogram(hfest[:n, -1], BINS, normed=True)[0]
   hfDs.append(entropy(dist, lastdist))
   lastdist = dist.copy()
   

SAMPS = [50]
while SAMPS[-1] * 2 < lfest.shape[0]:
   SAMPS.append(SAMPS[-1] * 2)
lastdist = np.histogram(lfest[:int(SAMPS[0] / 2), -1], BINS, normed=True)[0]
lfDs = []
for n in SAMPS:
   dist = np.histogram(lfest[:n, -1], BINS, normed=True)[0]
   lfDs.append(entropy(dist, lastdist))
   lastdist = dist.copy()
   

SAMPS = [50]
while SAMPS[-1] * 2 < lfest.shape[0]:
   SAMPS.append(SAMPS[-1] * 2)
lastdist = np.histogram(cvest[:int(SAMPS[0] / 2), -1], BINS, normed=True)[0]
cvDs = []
for n in SAMPS:
   dist = np.histogram(cvest[:n, -1], BINS, normed=True)[0]
   cvDs.append(entropy(dist, lastdist))
   lastdist = dist.copy()
   

SAMPS = [50]
while SAMPS[-1] * 2 < lfest.shape[0]:
   SAMPS.append(SAMPS[-1] * 2)
lastdist = np.histogram(cv2est[:int(SAMPS[0] / 2), -1], BINS, normed=True)[0]
cv2Ds = []
for n in SAMPS:
   dist = np.histogram(cv2est[:n, -1], BINS, normed=True)[0]
   cv2Ds.append(entropy(dist, lastdist))
   lastdist = dist.copy()
   



