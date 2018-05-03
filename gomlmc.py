import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dat = pd.read_csv('./mysampledsamples.dat')
dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)

LW = .1
f, ax = plt.subplots()
for _ in range(300):
    dat = dat.sample(frac=1).reset_index(drop=True)
    def mch(x):
         return (np.mean(dat.hf) - np.mean(dat.hf.values[:x])) ** 2, np.var(dat.hf.values[:x]) / x
    mch = np.vectorize(mch)
    
    def cv2(x):
        alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
        m = (np.mean(dat.hf) - (
            np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x])))) ** 2
        std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf))) * (np.var(dat.lf) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf) * np.var(dat.hf[:x])))
        return m, std, alpha
    cv2 = np.vectorize(cv2)
            
    x = [int(s) for s in np.arange(10, 200, 2)]
    print (x)
    if _ == 0:
       ax.plot(x, mch(x)[0], c='b', label=r'$\epsilon^2_{HF}$', lw=LW)
       ax.plot(x, cv2(x)[0], c='g', label=r'$\epsilon^2_{\alpha^*}$', lw=LW)
    else:
       ax.plot(x, mch(x)[0], c='b', lw=LW)
       ax.plot(x, cv2(x)[0], c='g', lw=LW)
    #ax.plot(x, mch(x)[1], c='b', label=r'$\sigma^2_{HF}$', ls='--')
    #ax.plot(x, cv2(x)[1], c='g', label=r'$\sigma^2_{\alpha^*}$', ls='--')
    #if np.gradient( cv2(x)[0])[-1] > 0 : break    
ax.set_yscale('log')
ax.set_xlabel("High Fidelity Samples")
ax.legend(ncol=2, prop={'size':16}, bbox_to_anchor=[1,1.5])
plt.savefig('mlmc.pdf', bbox_inches='tight')
plt.show()
