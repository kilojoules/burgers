import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.integrate import simps

# error pdfs #
if False:
    dat = pd.read_csv('./timedsamples.dat')
    dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
    dat = dat.convert_objects(convert_numeric=True)
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-.7)
    TEST = 50
    bigBS = int(dat.shape[0] / (TEST * 6))
    dats = np.array_split(dat, bigBS)
    samplesx = [2]
    while samplesx[-1]*2 < bigBS:
        samplesx.append(samplesx[-1] * 2)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(samplesx)))
    print(samplesx)
    for ll, BS in enumerate(samplesx):
        LW = .1
        ALPHA = 0.5
        x = np.array([int(s) for s in np.arange(10, TEST+1, 2)])
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
                    
            ev_mch = mch(x)
            ev_cv2 = cv2(x)
            epsh[:, _] = np.sqrt(ev_mch[0])
            epscv[:, _] = np.sqrt(ev_cv2[0])
            if len(ev_cv2[0][np.isnan(ev_cv2[0])]) > 0: hey
        
        xs, ys = [], []
        y2s = []
        for ii in [np.where(x==TEST)[0][0]]:
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
            xx = np.linspace(0, max(vesey), 100)
            #xx = np.linspace(min(vesey), max(vesey), 100)
            print('---->', min(thesey), max(thesey))
            kde2 = KernelDensity(kernel='gaussian', bandwidth=np.mean(vesey))
            kde2.fit(vesey.reshape(-1, 1))
            pdff = np.exp(kde2.score_samples(xx.reshape(-1, 1)))
            ax[0].plot(xx, pdff / np.sum(pdff), label=BS, c=colors[ll])
            xx = np.linspace(0, 2 * max(thesey), 1000)
            #xx = np.linspace(min(thesey), max(thesey), 100)
            kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(thesey))
            kde.fit(thesey.reshape(-1, 1))
            pdff = np.exp(kde.score_samples(xx.reshape(-1, 1)))
            ax[1].plot(xx, pdff, label=BS, c=colors[ll])
    ax[0].set_xlabel('Error, HF') 
    ax[1].set_xlabel('Error, CV') 
    l = ax[0].legend(ncol=3)
    l.set_title('Sample Sets')
    ax[0].set_ylabel('probability density') 
    ax[1].set_ylabel('probability density') 
    f.suptitle('%i HF samples, %i LF samples, %i test HF samples'%(TEST, dat.shape[0], dat.shape[0]-TEST))
    f.savefig('sampleContours.pdf', bbox_inches='tight') ; plt.clf()
    plt.clf()
    plt.close()
    
if True:    
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
        xx = np.linspace(10, 26, 1000)
        #xx = np.linspace(min(HF), max(HF), 100)
        kde = KernelDensity(kernel='gaussian')
        kde.fit(HF.reshape(-1, 1))
        myfit = np.exp(kde.score_samples(xx.reshape(-1, 1)))
        myfit /= simps(myfit, xx)
        ax[0].plot(xx, np.exp(kde.score_samples(xx.reshape(-1, 1))), c=colors[ll], label=n)
        ax[0].legend(ncol=3)
        #xx = np.linspace(min(LF), max(LF), 100)
        kde = KernelDensity(kernel='gaussian')
        kde.fit(LF.reshape(-1, 1))
        pdff = np.exp(kde.score_samples(xx.reshape(-1, 1)))
        pdff /= simps(pdff, xx)
        ax[1].plot(xx, pdff, c=colors[ll])
    ax[0].hist(HF, 30, alpha=0.3, normed=True)
    ax[1].hist(LF, 30, alpha=0.3, normed=True)
    ax[1].set_xlabel(r'$\max(\nabla u)_{400 Cells}$')
    ax[0].set_xlabel(r'$\max(\nabla u)_{7000 Cells}$')
    ax[0].set_ylabel('probability density') 
    ax[1].set_ylabel('probability density') 
    plt.savefig('./rawPDFs.pdf', bbox_inches='tight')
    plt.clf()
    
    
# expected values #
if False:
    
    dat = pd.read_csv('./timedsamples.dat')
    dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc'}, inplace=1)
    dat = dat.convert_objects(convert_numeric=True)
    TEST = 50
    bigBS = int(dat.shape[0] / (TEST * 6))
    dats = np.array_split(dat, bigBS)
    f, ax = plt.subplots(2, sharex=True)
    SAMPS = [50]
    while SAMPS[-1] * 2 < len(dats): SAMPS.append(SAMPS[-1]*2)
    #plt.subplots_adjust(bottom=-2, hspace=1)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(SAMPS)))
    for ll, n in enumerate(SAMPS):
        exps = []
        CC = []
        for ii in range(n):
            exps.append(dats[ii].hf[:TEST].mean())
            dat = dats[ii]
            def cv2(x):
                alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
                m = (np.mean(dat.hf[TEST:]) - (
                    np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x])))) ** 2
                std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf))) * (np.var(dat.lf) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf) * np.var(dat.hf[:x])))
                return m, std, alpha, np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x]))
            CC.append(cv2(50)[-1])
        xx = np.linspace(15, 17, 1000)
        #xx = np.linspace(.9 * np.min(exps), 1.1 * np.max(exps), 1000)
        kde = KernelDensity(kernel='gaussian', bandwidth=1e-1)
        kde.fit(np.array(exps).reshape(-1, 1))
        myfit = np.exp(kde.score_samples(xx.reshape(-1, 1)))
        myfit /= simps(myfit, xx)
        ax[0].plot(xx, myfit, c=colors[ll], label=n, zorder=3)
        #xx = np.linspace(min(LF), max(LF), 100)
        kde = KernelDensity(kernel='gaussian', bandwidth=5e-2)
        kde.fit(np.array(CC).reshape(-1, 1))
        pdff = np.exp(kde.score_samples(xx.reshape(-1, 1)))
        ax[1].plot(xx, pdff / simps(pdff, xx), c=colors[ll], zorder=4)
    leg = ax[0].legend(ncol=2, bbox_to_anchor=[1, 1])
    leg.set_title("Number of sample sets")
    ax[0].hist(exps, 30, normed=True, alpha=0.8, zorder=1)
    ax[1].hist(CC, 30, normed=True, alpha=0.8, zorder=2)
    ax[0].set_xlabel(r'$Q_{HF}$')
    ax[1].set_xlabel(r'$Q_{CV}$')
    ax[0].set_ylabel('probability density') 
    ax[1].set_ylabel('probability density') 
    plt.savefig('./qPDFs.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()



    exps = []
    CC = []
    NSAMPS = []
    for kk in [10, 20, 50]:
      for ii in range(n):
        exps.append(dats[ii].hf[:kk].mean())
        dat = dats[ii]
        def cv2(x):
            alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
            m = (np.mean(dat.hf[TEST:]) - ( np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x])))) ** 2
            std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf))) * (np.var(dat.lf) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf) * np.var(dat.hf[:x])))
            return m, std, alpha, np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf) - np.mean(dat.lf.values[:x]))
        CC.append(cv2(kk)[-1])
        NSAMPS.append(kk)


    vdat = pd.DataFrame()
    vdat['NSAMPS'] = NSAMPS
    vdat['HF'] = exps
    vdat['CV'] = CC
    f, ax = plt.subplots(1, 2, sharey=True)
    plt.subplots_adjust(right=1.5)
    sns.violinplot(x='NSAMPS', y='HF', data=vdat, ax=ax[0], color='white')
    sns.violinplot(x='NSAMPS', y='CV', data=vdat, ax=ax[1], color='white')
    for ii in range(2): ax[ii].set_xlabel("Number of High Fidelity Samples")
    ax[0].set_ylabel(r'$Q_{HF}$')
    ax[1].set_ylabel(r'$Q_{CV}$')
    plt.suptitle('%i sets, %i test HF samples, %i LF samples'%(bigBS, dat.shape[0] - TEST, dat.shape[0]), y=1.)
    plt.tight_layout()
    plt.savefig('expectedViolins.pdf', bbox_inches='tight')
