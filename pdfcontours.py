import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
dat = pd.read_csv('./timedsamples.dat')
dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
dat = dat.convert_objects(convert_numeric=True)
f, ax = plt.subplots(2)
plt.subplots_adjust(bottom=-.7)
TEST = 102
bigBS = int(dat.shape[0] / (TEST * 6))
dats = np.array_split(dat, bigBS)
samplesx = [2]
while samplesx[-1]*2 < bigBS:
    samplesx.append(samplesx[-1] * 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(samplesx)))
print(samplesx)
for ll, BS in enumerate(samplesx):
    LW = .1
    ALPHA = 0.5
    x = np.array([int(s) for s in np.arange(10, TEST, 2)])
    epsh = np.zeros((x.size, BS))
    epscv = np.zeros((x.size, BS))
    for _ in range(BS):
        dat = dats[_]
        def mch(x):
             return (np.mean(dat.hf[TEST:]) - np.mean(dat.hf.values[:x])) ** 2, np.var(dat.hf.values[:x]) / x, np.mean(dat.hf.values[:x]), np.mean(dat.hf[TEST:])
        mch = np.vectorize(mch)
        
        def cv2(x):
            alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
            m = (np.mean(dat.hf[TEST:]) - (
                np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x])))) ** 2
            std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf))) * (np.var(dat.lf) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf) * np.var(dat.hf[:x])))
            return m, std, alpha, np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x]))
        cv2 = np.vectorize(cv2)
                
        print (_, dat.shape)
        ev_mch = mch(x)
        ev_cv2 = cv2(x)
        epsh[:, _] = np.sqrt(ev_mch[0])
        epscv[:, _] = np.sqrt(ev_cv2[0])
        if len(ev_cv2[0][np.isnan(ev_cv2[0])]) > 0: hey
    
    xs, ys = [], []
    y2s = []
    for ii in [np.where(x==TEST-2)[0][0]]:
        thesex, thesey, vesey = [], [], []
        for jj in range(epsh.shape[1]):
            ys.append(epsh[ii, jj])
            y2s.append(epscv[ii, jj])
            xs.append(x[ii])
            thesey.append(y2s[-1])
            vesey.append(ys[-1])
        np.save('thesey%i'%BS, thesey)
        thesey = np.array(thesey)
        vesey = np.array(vesey)
        xx = np.linspace(min(vesey), max(vesey), 100)
        print('---->', min(thesey), max(thesey))
        kde2 = KernelDensity(kernel='gaussian', bandwidth=np.mean(vesey))
        kde2.fit(vesey.reshape(-1, 1))
        ax[0].plot(xx, np.exp(kde2.score_samples(xx.reshape(-1, 1))), label=BS, c=colors[ll])
        xx = np.linspace(min(thesey), max(thesey), 100)
        kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(thesey))
        kde.fit(thesey.reshape(-1, 1))
        ax[1].plot(xx, np.exp(kde.score_samples(xx.reshape(-1, 1))), label=BS, c=colors[ll])
ax[0].set_xlabel('Error, HF') 
ax[1].set_xlabel('Error, CV') 
l = ax[0].legend(ncol=3)
l.set_title('Sample Sets')
ax[0].set_ylabel('probability density') 
ax[1].set_ylabel('probability density') 
f.suptitle('%i HF samples, %i LF samples, %i test HF samples'%(TEST-2, dat.shape[0], dat.shape[0]-TEST+2))
f.savefig('sample_contours.pdf', bbox_inches='tight') ; plt.clf()
plt.clf()
plt.close()




dat = pd.read_csv('./timedsamples.dat')
dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
dat = dat.convert_objects(convert_numeric=True)
f, ax = plt.subplots(2, sharex=True)
#plt.subplots_adjust(bottom=-2, hspace=1)
SAMPS = [25]
while SAMPS[-1] < 1000: SAMPS.append(SAMPS[-1] * 2)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
for ll, n in enumerate(SAMPS):
    HF = dat.hf[:n]
    LF = dat.lf[:n]
    xx = np.linspace(min(HF), max(HF), 100)
    kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(HF))
    kde.fit(HF.reshape(-1, 1))
    ax[0].plot(xx, np.exp(kde.score_samples(xx.reshape(-1, 1))), c=colors[ll], label=n)
    ax[0].legend(ncol=3)
    xx = np.linspace(min(LF), max(LF), 100)
    kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(LF))
    kde.fit(LF.reshape(-1, 1))
    ax[1].plot(xx, np.exp(kde.score_samples(xx.reshape(-1, 1))), c=colors[ll])
ax[1].set_xlabel(r'$\max(\nabla u)_{400 Cells}$')
ax[0].set_xlabel(r'$\max(\nabla u)_{7000 Cells}$')
ax[0].set_ylabel('probability density') 
ax[1].set_ylabel('probability density') 
plt.savefig('./rawPDFs.pdf', bbox_inches='tight')
plt.clf()
