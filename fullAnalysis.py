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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import moment
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.integrate import simps
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity
mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'
#d1 = pd.read_csv('./d1.dat')
#d2 = pd.read_csv('./collected.csv')
#myhfdat = pd.read_csv('./collected.csv')
#d3 = pd.read_csv('./timedsamples.dat')
#myhfdat = pd.concat([d1, d2, d3])
myhfdat =  pd.read_csv('./collected.csv')
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
sf = open('datstats.dat', 'w')
sf.write(' '.join([str(s) for s in ['omega_0: ', omega_1, 'omega_1: ', omega_2, 'omega_2: ', omega_3, '\n']]))
sf.write(' '.join([str(s) for s in ['rho_1_2: ', rho_1_2, 'rho_1_3: ', rho_1_3, '\n']]))
sf.write(' '.join([str(s) for s in ['Var(f_0):', np.var(myhfdat.hf), 'var(f_1): ', np.var(mylfdat.lf1), 'var(f_2): ', np.var(mylfdat.lf2)]]))
sf.close()
f, ax = plt.subplots(1, 3)
plt.subplots_adjust(wspace=.8, right=2)
ax[0].set_yscale('log')
ax[0].hist(myhfdat.thf, 50, normed=True)
ax[0].set_xlabel(r'$\omega_{0}$ (seconds)')
ax[1].set_yscale('log')
ax[1].hist(mylfdat.tlf1, 50, normed=True)
ax[1].set_xlabel(r'$\omega_{{\mathrm{Reduced}}}$ (seconds)')
ax[2].set_yscale('log')
ax[2].hist(mylfdat.tlf2, 50, normed=True)
ax[2].set_xlabel(r'$\omega_{1}$ (seconds)')
for ii in range(3): ax[ii].set_ylabel("Probability Density")
plt.savefig('TimeDists.pdf', bbox_inches='tight')
plt.clf()
plt.close()
plt.cla()
plt.show()

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

if True:
    x = np.arange(2, 11)
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
       dat = dats[_]
       lfdat = lfdats[_]
       cvt[:, _] = cvtimev(x, (r_cv * x / x[-1]).astype(int), dat, lfdat)
       cvest[:, _] = cvv(x, (r_cv * x).astype(int), dat, lfdat)
    x = np.arange(2, 100)
    dats = np.array_split(myhfdat, myhfdat.shape[0] / x[-1])
    lfdats = np.array_split(mylfdat, mylfdat.shape[0] / x[-1])
    BS = len(dats)
    hfest = np.zeros((x.size, BS))
    hft = np.zeros((x.size, BS))
    lfest = np.zeros((x.size, BS))
    lft = np.zeros((x.size, BS))
    for _ in range(BS):
       dat = dats[_]
       hft[:, _] = hftimev(x, dat)
       hfest[:, _] = hfv(x, dat)
    #BS = len(lfdats)
    #for _ in range(BS):
       lfdat = lfdats[_]
       lft[:, _] = lftimev(x, lfdat)
       lfest[:, _] = lfv(x, lfdat)

np.save('lfest', lfest)
np.save('cvest', lfest)
np.save('hfest', lfest)
np.save('lft', lft)
np.save('hft', hft)
np.save('cvt', cvt)

plt.plot(x, np.mean(np.abs(TRUTH - hfest), 1), label=r'$Q_{0}$', marker='x')
plt.plot(x, np.mean(np.abs(TRUTH - lfest), 1), label=r'$Q_{\mathrm{Reduced}}$', marker='x')
plt.plot(xcv, np.mean(np.abs(TRUTH - cvest), 1), label=r'$Q_{\mathrm{CV}}$', marker='x')
#plt.plot(xcv2, np.mean(np.abs(TRUTH - cv2est), 1), label='CV2', marker='x')
plt.legend(ncol=3, bbox_to_anchor=[1.1, 1.3], prop={'size':16})
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$|\mathbb{E}[f_0(\nu)] - Q|$')
plt.xlabel('High-Fidelity Samples')
plt.savefig('convPlot.pdf', bbox_inches='tight')
plt.clf()


plt.plot(hft.mean(1), np.mean(np.abs(TRUTH - hfest), 1), label=r'$Q_{0}$', marker='x')
plt.plot(lft.mean(1), np.mean(np.abs(TRUTH - lfest), 1), label=r'$Q_{\mathrm{Reduced}}$', marker='x')
plt.plot(cvt.mean(1), np.mean(np.abs(TRUTH - cvest), 1), label=r'$Q_{\mathrm{CV}}$', marker='x')
#plt.plot(cv2t.mean(1), np.mean(np.abs(TRUTH - cv2est), 1), label='CV2', marker='x')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('cpu time (seconds)')
plt.ylabel(r'$|\mathbb{E}[f_0(\nu)] - Q|$')
plt.legend(ncol=3, bbox_to_anchor=[1.1, 1.3], prop={'size':16})
plt.savefig('timePlot.pdf', bbox_inches='tight')
plt.clf()


f, ax = plt.subplots(1, 3)
plt.subplots_adjust(right=2)
SAMPS = [50]
while SAMPS[-1] * 2 < hfest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
hfDs = []
xx = np.linspace(10, 25, 200)
lastpdf = None
BW = np.std(hfest[-1, :], ddof=2) * 1.06 * hfest.shape[1] ** (-1./5)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(hfest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff / simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: hfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[0].plot(xx, pdff, label=n, color=colors[ii])
hfsamps = np.array(SAMPS).copy()

ax[0].set_xlabel(r'$Q_{0}$')
ax[0].hist(hfest[0, :n], 50, normed=True)


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(12, 20, 200)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
lfDs = []
BW = np.std(lfest[-1, :], ddof=2) * 1.06 * lfest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   #dist = np.histogram(cvest[-1, :n], BINS, normed=True)[0]
   #cvDs.append(entropy(dist, lastdist))
   #lastdist = dist.copy()
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(lfest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: lfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[1].plot(xx, pdff, label=n, color=colors[ii])
ax[1].legend(title="Groups of Samples")
ax[1].hist(lfest[0, :], 50, normed=True)
ax[1].set_xlabel(r'$Q_{\mathrm{Reduced}}$')


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(12, 18, 200)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
cvDs = []
BW = np.std(cvest[-1, :], ddof=2) * 1.06 * cvest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(cvest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: cvDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[2].plot(xx, pdff, label=n, color=colors[ii])
ax[2].hist(cvest[0, :], 50, normed=True)
ax[2].set_xlabel(r'$Q_{\mathrm{CV}}$')
for ii in [1]: ax[ii].legend(ncol=4, bbox_to_anchor=[1.7, 1.6], title="Groups of Samples")
plt.savefig('./CVHists.pdf', bbox_inches='tight')
plt.clf()
plt.close()
plt.cla()

plt.show()

plt.plot(hfsamps[1:], hfDs, label=r'$Q_{0}$')
plt.plot(SAMPS[1:], lfDs, label=r'$Q_{\mathrm{Reduced}}$')
plt.plot(SAMPS[1:], cvDs, label=r'$Q_{\mathrm{CV}}$')
plt.ylabel('D')
plt.xlabel('Number of Sets of Samples')
plt.yscale('log')
plt.xscale('log')
plt.yscale('log')
plt.legend(ncol=3, bbox_to_anchor=[1.2, 1.4])
plt.savefig('./DPlot.pdf', bbox_inches='tight')
plt.clf()


f, ax = plt.subplots(1, 3)
plt.subplots_adjust(right=2, wspace=.5)

SAMPS = [400]
while SAMPS[-1] * 2 < hfest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
hfDs = []
xx = np.linspace(-10, 5, 200)
lastpdf = None
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
BW = np.std(hfest[-1, :], ddof=2) * 1.06 * hfest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - hfest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff[pdff==0] = 1e-100
   pdff /= simps(pdff, xx)
   if lastpdf is not None: hfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[0].plot(xx, pdff, label=n, color=colors[ii])
hfsamps = np.array(SAMPS).copy()
ax[0].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_0$')
ax[0].hist(TRUTH - hfest[0, :n], 50, normed=True)

SAMPS = [400]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
cvDs1 = []
cvDs2 = []
cvDs3 = []
cvDs4 = []
#xx = np.linspace(np.min(cvest), np.max(cvest), 200)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
lfDs = []
BW = np.std(TRUTH - lfest[-1, :], ddof=2) * 1.06 * lfest.shape[1] ** (-1./5)
xx = np.linspace(-10, 5, 100)
for ii, n in enumerate(SAMPS):
   #dist = np.histogram(cvest[-1, :n], BINS, normed=True)[0]
   #cvDs.append(entropy(dist, lastdist))
   #lastdist = dist.copy()
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - lfest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff[pdff==0] = 1e-100
   pdff /= simps(pdff, xx)
   if lastpdf is not None: lfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[1].plot(xx, pdff, label=n, color=colors[ii])
#ax[1].legend()
ax[1].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_{Reduced}$')
ax[1].hist(TRUTH - lfest[0, :], 50, normed=True)


xx = np.linspace(-4, 5, 200)
SAMPS = [400]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
cvDs = []
BW = np.std(TRUTH - cvest[-1, :], ddof=2) * 1.06 * cvest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - cvest[0,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: cvDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[2].plot(xx, pdff, label=n, color=colors[ii])
ax[2].hist(TRUTH - cvest[0, :], 50, normed=True)
ax[2].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_{CV}$')
#ax[1].legend(ncol=5, bbox_to_anchor=[2.4, 1.2], title="Groups of Samples")
ax[1].legend(ncol=3, bbox_to_anchor=[1.7, 1.6], title="Groups of Samples")
for ii in range(3): ax[ii].set_ylabel('Probability Density')
plt.savefig('./ErrorHists.pdf', bbox_inches='tight')
plt.clf()
plt.close()
plt.cla()
plt.show()


f, ax = plt.subplots()
ax.plot(hfsamps[1:], hfDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{0}$')
ax.plot(SAMPS[1:], lfDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{Reduced}}$')
ax.plot(SAMPS[1:], cvDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{CV}}$')
ax.set_ylabel('D')
ax.set_xlabel('Number of Sets of Samples')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(ncol=1, bbox_to_anchor=[1., 1.6])
plt.savefig('./EroorDPlot.pdf', bbox_inches='tight')
plt.clf()
plt.show()



viohf, viorf, viocv, viox, vioxcv = [], [], [], [], []
for ii in [np.where(x==2)[0][0], np.where(x==5)[0][0], np.where(x==xcv[-1])[0][0]]:
  for jj in range(hfest.shape[1]):
    viohf.append(hfest[ii, jj])
    viorf.append(lfest[ii, jj])
    viox.append(x[ii])
for ii in [np.where(xcv==2)[0][0], np.where(xcv==5)[0][0], np.where(xcv==xcv[-1])[0][0]]:
  for jj in range(cvest.shape[1]):
    viocv.append(cvest[ii, jj])
    vioxcv.append(xcv[ii])

newd1 = pd.DataFrame()
newd2 = pd.DataFrame()
newd2['xcv'] = vioxcv
newd1['x'] = viox
newd1['y'] = viohf
newd2['y2'] = viocv
newd1['ylf1'] = viorf
f, ax = plt.subplots(3)
plt.subplots_adjust(bottom=-1., hspace=.5)
sns.violinplot(x='x', y='y', data=newd1, ax=ax[0], color='white')
sns.violinplot(x='x', y='ylf1', data=newd1, ax=ax[1], color='white')
sns.violinplot(x='xcv', y='y2', data=newd2, ax=ax[2], color='white')
for ii in range(3):
    ax[ii].set_xlabel('High Fidelity Samples')
    ax[ii].set_ylim(newd1.y.min(), newd1.y.max())
ax[0].set_ylabel(r'$Q_{0}$')
ax[1].set_ylabel(r'$Q_{\mathrm{Reduced}}$')
ax[2].set_ylabel(r'$Q_{\mathrm{CV}}$')
plt.savefig('./expectedviolins.pdf', bbox_inches='tight')
plt.clf()
plt.show()

MOST = np.where(x==xcv[-1])[0][0]



newd1 = pd.DataFrame()
newd2 = pd.DataFrame()
newd2['xcv'] = vioxcv
newd1['x'] = viox
newd1['y'] = TRUTH - viohf
newd2['y2'] = TRUTH - viocv
newd1['ylf1'] = TRUTH - viorf
f, ax = plt.subplots(3)
plt.subplots_adjust(bottom=-1., hspace=.5)
sns.violinplot(x='x', y='y', data=newd1, ax=ax[0], color='white')
sns.violinplot(x='x', y='ylf1', data=newd1, ax=ax[1], color='white')
sns.violinplot(x='xcv', y='y2', data=newd2, ax=ax[2], color='white')
for ii in range(3):
    ax[ii].set_xlabel('High Fidelity Samples')
    ax[ii].set_ylim(newd1.y.min(), newd1.y.max())
ax[0].set_ylabel(r'$\mathbb{E}[f_0(\nu)] - Q_{0}$')
ax[1].set_ylabel(r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{Reduced}}$')
ax[2].set_ylabel(r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{CV}}$')
plt.savefig('./errorviolins.pdf', bbox_inches='tight')
plt.clf()
plt.show()




f, ax = plt.subplots(1, 3)
plt.subplots_adjust(right=2)
SAMPS = [50]
while SAMPS[-1] * 2 < hfest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
hfDs = []
xx = np.linspace(-5, 5, 100)
lastpdf = None
BW = np.std(TRUTH - hfest[-1, :], ddof=1) * 1.06 * hfest.shape[1] ** (-1./5)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - hfest[MOST,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff / simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: hfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[0].plot(xx, pdff, label=n, color=colors[ii])
hfsamps = np.array(SAMPS).copy()

ax[0].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_{0}$')
ax[0].hist(TRUTH - hfest[MOST, :n], 50, normed=True)


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(-5, 5, 200)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
lfDs = []
BW = np.std(lfest[-1, :], ddof=2) * 1.06 * lfest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   #dist = np.histogram(cvest[-1, :n], BINS, normed=True)[0]
   #cvDs.append(entropy(dist, lastdist))
   #lastdist = dist.copy()
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - lfest[MOST,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: lfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[1].plot(xx, pdff, label=n, color=colors[ii])
ax[1].legend()
ax[1].hist(TRUTH - lfest[MOST, :], 50, normed=True)

ax[1].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{Reduced}}$')


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(-5, 5, 200)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
cvDs = []
BW = np.std(TRUTH - cvest[-1, :], ddof=2) * 1.06 * cvest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(TRUTH - cvest[-1,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: cvDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[2].plot(xx, pdff, label=n, color=colors[ii])
ax[2].hist(TRUTH - cvest[-1, :], 50, normed=True)
ax[2].set_xlabel(r'$\mathbb{E}[f_0(\nu)] - Q_{CV}$')
ax[1].legend(ncol=3, bbox_to_anchor=[1.7, 1.6], title="Groups of Samples")
plt.savefig('./CVHistsErrMost.pdf', bbox_inches='tight')
plt.clf()
plt.close()
plt.cla()

plt.show()

plt.plot(hfsamps[1:], hfDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{0}$')
plt.plot(SAMPS[1:], lfDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{Reduced}}$')
plt.plot(SAMPS[1:], cvDs, label=r'$\mathbb{E}[f_0(\nu)] - Q_{\mathrm{CV}}$')
plt.ylabel('D')
plt.xlabel('Number of Sets of Samples')
plt.yscale('log')
plt.xscale('log')
plt.yscale('log')
plt.legend(ncol=1, bbox_to_anchor=[.9, 1.55555])
plt.savefig('./MostDErrorPlot.pdf', bbox_inches='tight')
plt.clf()
plt.show()


f, ax = plt.subplots(1, 3)
plt.subplots_adjust(right=2)
SAMPS = [50]
while SAMPS[-1] * 2 < hfest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
hfDs = []
xx = np.linspace(12, 22, 200)
lastpdf = None
BW = np.std(hfest[-1, :], ddof=1) * 1.06 * hfest.shape[1] ** (-1./5)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(hfest[MOST,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff / simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: hfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[0].plot(xx, pdff, label=n, color=colors[ii])
hfsamps = np.array(SAMPS).copy()


ax[0].set_xlabel(r'$Q_{0}$')
ax[0].hist(hfest[MOST, :n], 50, normed=True)


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(12, 22, 100)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
lfDs = []
BW = np.std(lfest[-1, :], ddof=2) * 1.06 * lfest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   #dist = np.histogram(cvest[-1, :n], BINS, normed=True)[0]
   #cvDs.append(entropy(dist, lastdist))
   #lastdist = dist.copy()
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(lfest[MOST,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: lfDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[1].plot(xx, pdff, label=n, color=colors[ii])
ax[1].legend()
ax[1].hist(lfest[MOST, :], 50, normed=True)

ax[1].set_xlabel(r'$Q_{mathrm{Reduced}}$')


SAMPS = [50]
while SAMPS[-1] * 2 < cvest.shape[1]:
   SAMPS.append(SAMPS[-1] * 2)
xx = np.linspace(14, 18, 100)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
lastpdf = None
cvDs = []
BW = np.std(cvest[-1, :], ddof=2) * 1.06 * cvest.shape[1] ** (-1./5)
for ii, n in enumerate(SAMPS):
   kde2 = KernelDensity(kernel='gaussian', bandwidth=BW)
   kde2.fit(cvest[-1,:n].reshape(-1, 1))
   pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
   pdff /= simps(pdff, xx)
   pdff[pdff==0] = 1e-100
   if lastpdf is not None: cvDs.append(entropy(pdff, lastpdf))
   lastpdf = pdff.copy()
   ax[2].plot(xx, pdff, label=n, color=colors[ii])
ax[2].hist(cvest[-1, :], 50, normed=True)
ax[2].set_xlabel(r'$Q_{CV}$')
ax[1].legend(ncol=3, bbox_to_anchor=[1.7, 1.6], title="Groups of Samples")
plt.savefig('./CVHistsMost.pdf', bbox_inches='tight')
plt.clf()
plt.close()
plt.cla()
plt.show()

plt.plot(hfsamps[1:], hfDs, label=r'$Q_{0}$')
plt.plot(SAMPS[1:], lfDs, label=r'$Q_{\mathrm{Reduced}}$')
plt.plot(SAMPS[1:], cvDs, label=r'$Q_{\mathrm{CV}}$')
plt.ylabel('D')
plt.xlabel('Number of Sets of Samples')
plt.yscale('log')
plt.xscale('log')
plt.yscale('log')
plt.legend(ncol=3, bbox_to_anchor=[1.3, 1.3])
plt.savefig('./MostDPlot.pdf', bbox_inches='tight')
plt.clf()
plt.show()

