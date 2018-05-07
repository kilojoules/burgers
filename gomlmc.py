import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dat = pd.read_csv('./correctedData.dat', sep=', ')
dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = pd.read_csv('./mysampledsamples.dat')
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = dat[dat.visc > 1e-6]
print(dat.shape)

LW = .1
ALPHA = 0.5
f, ax = plt.subplots(3)
TEST = 200
BS = 800
x = np.array([int(s) for s in np.arange(10, TEST, 2)])
epsh = np.zeros((x.size, BS))
epscv = np.zeros((x.size, BS))
for _ in range(BS):
    dat = dat.sample(frac=1).reset_index(drop=True)
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
            
    print (_)
    ev_mch = mch(x)
    ev_cv2 = cv2(x)
    epsh[:, _] = np.sqrt(ev_mch[0])
    epscv[:, _] = np.sqrt(ev_cv2[0])
    ax[0].plot(x, ev_mch[-2], c='b', lw=LW)
    ax[0].plot(x, ev_cv2[-1], c='g', lw=LW)
    #ax[0].plot(x, cv(x)[-1], c='g')
    if _ == 0:
       ax[1].plot(x, np.sqrt(ev_mch[0]), c='b', label=r'$\epsilon_{HF}$', lw=LW, alpha=ALPHA)
       ax[1].plot(x, np.sqrt(ev_cv2[0]), c='g', label=r'$\epsilon_{\alpha^*}$', lw=LW, alpha=ALPHA)
    else:
       ax[1].plot(x, np.sqrt(ev_mch[0]), c='b', lw=LW, alpha=ALPHA)
       ax[1].plot(x, np.sqrt(ev_cv2[0]), c='g', lw=LW, alpha=ALPHA)
    #ax.plot(x, mch(x)[1], c='b', label=r'$\sigma^2_{HF}$', ls='--')
    #ax.plot(x, cv2(x)[1], c='g', label=r'$\sigma^2_{\alpha^*}$', ls='--')
    #if np.gradient( cv2(x)[0])[-1] > 0 : break    
ax[0].set_ylabel('Expected Value')
ax[1].set_yscale('log')
#ax[1].set_xlabel("High Fidelity Samples")
ax[1].set_xscale('log')
ax[1].set_ylabel(r'$\epsilon$')
ax[1].legend(bbox_to_anchor=[1., 0.8])
#ax[2].plot(x, np.percentile(epsh, 10, 1), c='b', label='10th percentile', ls='--')
#ax[2].plot(x, np.percentile(epsh, 90, 1), c='b', label='90th percentile', ls=':')
ax[2].fill_between( x, np.percentile(epsh, 20, 1), np.percentile(epsh, 80, 1), facecolor='lightblue', alpha=0.7)
ax[2].fill_between( x, np.percentile(epscv, 20, 1), np.percentile(epscv, 80, 1), facecolor='lightgreen', alpha=0.7)
#ax[2].legend(bbox_to_anchor=[1., 0.8])
ax[2].plot(x, epscv.mean(1), c='g', label='mean')
ax[2].plot(x, epsh.mean(1), c='b', label='mean')
#ax[2].plot(x, np.percentile(epscv, 10, 1), c='g', label='10th percentile', ls='--')
#ax[2].plot(x, np.percentile(epscv, 90, 1), c='g', label='90th percentile', ls=':')
ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[0].set_xscale('log')
ax[2].set_ylabel(r'$\epsilon$')
ax[2].set_xlabel('High Fidelity Samples')
plt.savefig('mlmc.pdf', bbox_inches='tight')
#plt.show()
