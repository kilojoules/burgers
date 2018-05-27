import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
d1 = pd.read_csv('./d1.dat')
d2 = pd.read_csv('./collected.csv')
d3 = pd.read_csv('./timedsamples.dat')
dat = pd.concat([d1, d2, d3])
dat.rename(columns={'nx7000':'hf', 'nx400':'lf', 'nu':'visc', 'tnx7000':'t1', 'tnx400': 't2'}, inplace=1)
TRUTH = np.mean(dat.nx7000)
#dat = dat.convert_objects(convert_numeric=True)
#dat = dat[:100000]
#dat.reset_index(inplace=True)
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = pd.read_csv('./mysampledsamples.dat')
#dat.rename(columns={'nx1000':'lf', 'nx5000':'hf', 'nu':'visc'}, inplace=1)
#dat = dat[dat.visc > 1e-6]
print(dat.shape)

LW = .1
ALPHA = 0.5
f, ax = plt.subplots(4)
plt.subplots_adjust(bottom=-.7, hspace=.4)
TEST = 50
BS = int(dat.shape[0] / ((TEST) * 6))
LF = 200
dats = np.array_split(dat, BS)
x = np.array([int(s) for s in np.arange(2, TEST+1, 2)])
epsh = np.zeros((x.size, BS))
epscv = np.zeros((x.size, BS))
timecv = np.zeros((x.size, BS))
timeh = np.zeros((x.size, BS))
for _ in range(BS):
    dat = dats[_]
    def mch(x):
         timed = np.sum(dat.t1[:x]) / 3600
         return (np.mean(dat.hf[TEST:]) - np.mean(dat.hf.values[:x])) ** 2, np.var(dat.hf.values[:x]) / x, timed, np.mean(dat.hf.values[:x]), np.mean(dat.hf[TEST:])
    mch = np.vectorize(mch)
    
    def cv2(x):
        alpha = np.corrcoef(dat.hf.values[:x], dat.lf.values[:x])[0][1] * np.sqrt(np.var(dat.hf.values[:x]) / np.var(dat.lf.values[:x]))
        m = (np.mean(dat.hf[TEST:]) - (
            np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf[:LF]) - np.mean(dat.lf.values[:x])))) ** 2
        std = np.var(dat.hf.values[:x]) / x + (1. / float(x) - 1. /float(len(dat.lf[:LF]))) * (np.var(dat.lf[:LF]) * alpha ** 2 - 2 * alpha * np.corrcoef(dat.lf.values[:x], dat.hf.values[:x])[0][1] * np.sqrt(np.var(dat.lf[:LF]) * np.var(dat.hf[:x])))
        timed = (np.sum(dat.t1[:x]) + np.sum(dat.t2[:LF])) / 3600
        return m, std, alpha, timed, np.mean(dat.hf.values[:x]) + alpha * (np.mean(dat.lf[:LF]) - np.mean(dat.lf.values[:x]))
    cv2 = np.vectorize(cv2)
            
    print (_, dat.shape)
    ev_mch = mch(x)
    ev_cv2 = cv2(x)
    epsh[:, _] = np.sqrt(ev_mch[0])
    epscv[:, _] = np.sqrt(ev_cv2[0])
    timecv[:, _] = ev_cv2[3]
    timeh[:, _] = ev_mch[2]
    if len(ev_cv2[0][np.isnan(ev_cv2[0])]) > 0: hey
    ax[0].plot(x, ev_mch[-2], c='b', lw=LW)
    ax[0].plot(x, ev_cv2[-1], c='g', lw=LW)
    #ax[0].plot(x, cv(x)[-1], c='g')
    if _ == 0:
       #ax[1].plot(x, np.sqrt(ev_mch[0]), c='b', label=r'$\epsilon_{HF}$', lw=LW, alpha=ALPHA)
       #ax[1].plot(x, np.sqrt(ev_cv2[0]), c='g', label=r'$\epsilon_{\alpha^*}$', lw=LW, alpha=ALPHA)
       ax[1].plot(x, np.sqrt(ev_mch[0]), c='b', label='HF', lw=LW, alpha=ALPHA)
       ax[1].plot(x, np.sqrt(ev_cv2[0]), c='g', label='CV', lw=LW, alpha=ALPHA)
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
leg = ax[1].legend(bbox_to_anchor=[.6, 2.65], ncol=2)
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
    legobj.set_alpha(1)
#ax[2].plot(x, np.percentile(epsh, 10, 1), c='b', label='10th percentile', ls='--')
#ax[2].plot(x, np.percentile(epsh, 90, 1), c='b', label='90th percentile', ls=':')
ax[3].fill_between( x, np.percentile(timeh, 20, 1), np.percentile(timeh, 80, 1), facecolor='lightblue', alpha=0.7)
ax[3].fill_between( x, np.percentile(timecv, 20, 1), np.percentile(timecv, 80, 1), facecolor='lightgreen', alpha=0.7)
ax[3].plot(x, timecv.mean(1), c='g', label='mean')
ax[3].plot(x, timeh.mean(1), c='b', label='mean')
ax[3].minorticks_on()
ax[3].grid(which='both', axis='y', ls='--', color='gainsboro')
ax[3].grid(which='major', axis='y', ls='-', color='darkgrey')
#ax[3].grid(color='k', which='minor', axis='y', ls='--')
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
ax[3].set_xscale('log')
#ax[3].set_yscale('log')
ax[2].set_ylabel(r'$\epsilon$')
ax[3].set_ylabel('CPU Time (Hours)')
ax[3].set_xlabel('High Fidelity Samples')
f.suptitle('%i sets, %i test points, %i LF samples'%(BS, dat.shape[0]-TEST, dat.shape[0]))
plt.savefig('mlmcresults%i.pdf'%LF, bbox_inches='tight')
plt.clf()
plt.close()

np.save('timecv_%s'%LF, timecv.mean(1))
np.save('errorcv_%s'%LF, epscv.mean(1))
#timecv2k = np.load('timecv_2000.npy')
#epscv2k = np.load('errorcv_2000.npy')
timecv200 = np.load('timecv_200.npy')
epscv200 = np.load('errorcv_200.npy')
plt.plot(timecv.mean(1), epscv.mean(1), label='CV, %i LF samples'%LF, marker='x')
plt.plot(timecv200, epscv200, label='CV, 200 LF samples', marker='x')
#plt.plot(timecv2k, epscv2k, label='CV, 2000 samples')
plt.plot(timeh.mean(1), epsh.mean(1), label='HF')
plt.xlabel('cpu time (hours)')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('timeContours%i.pdf'%LF, bbox_inches='tight')


xs, ys = [], []
y2s = []
for ii in [0, np.where(x==20)[0][0], np.where(x==TEST)[0][0]]:
#for ii in [0, np.where(x==60)[0][0], np.where(x==TEST-2)[0][0]]:
    thesex, thesey, vesey = [], [], []
    for jj in range(epsh.shape[1]):
        ys.append(epsh[ii, jj])
        y2s.append(epscv[ii, jj])
        xs.append(x[ii])
        thesey.append(y2s[-1])
        vesey.append(ys[-1])
    thesey = np.array(thesey)
    vesey = np.array(vesey)
    f, ax = plt.subplots(2)
    plt.subplots_adjust(bottom=-.7, hspace=.4)
    ax[0].hist(vesey, 40, normed=True) 
    ax[1].hist(thesey, 40, normed=True) 
    xx = np.linspace(min(vesey), max(vesey), 100)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=np.mean(vesey))
    kde2.fit(vesey.reshape(-1, 1))
    ax[0].plot(xx, np.exp(kde2.score_samples(xx.reshape(-1, 1))))
    xx = np.linspace(min(thesey), max(thesey), 100)
    kde = KernelDensity(kernel='gaussian', bandwidth=np.mean(thesey))
    kde.fit(thesey.reshape(-1, 1))
    ax[1].plot(xx, np.exp(kde.score_samples(xx.reshape(-1, 1))))
    ax[0].set_xlabel('Error, HF') 
    ax[1].set_xlabel('Error, CV') 
    ax[0].set_ylabel('probability density') 
    ax[1].set_ylabel('probability density') 
    #ax[0].set_label('%i sets, %i LF samples, %i HF samples'%(BS, TEST, 10*TEST))
    f.suptitle('%i sets, %i LF samples, %i test HF samples'%(BS, x[ii], dat.shape[0] - TEST))
    f.savefig('hist_%i_%i.pdf'%(x[ii], LF), bbox_inches='tight') ; plt.clf()

newd = pd.DataFrame() 
newd['x'] = xs
newd['y'] = ys
newd['y2'] = y2s
f, ax = plt.subplots(2)
plt.subplots_adjust(right=1.5)
#opts = {'inner':'stripplot'}
#opts = {'scale': 'count', 'inner':'quart', 'cut':0}
opts = {}
#ax[0].set_yscale('log')
#ax[1].set_yscale('log')
sns.violinplot(x='x', y='y', data=newd, ax=ax[0], **opts, color='white')
#sns.stripplot(x='x', y='y', data=newd, ax=ax[0], color='k', size=1)
sns.violinplot(x='x', y='y2', data=newd, ax=ax[1], **opts, color='white')
#for ii in [0, 1]: ax[ii].set_ylim(min(y2s), max(ys))
for ii in range(2): ax[ii].set_xlabel('High Fidelity Samples')
ax[0].set_ylabel(r'$\epsilon_{HF}$')
ax[1].set_ylabel(r'$\epsilon_{CV}$')
f.suptitle('%i sets, %i test HF samples, %i LF samples'%(BS, dat.shape[0] - TEST, dat.shape[0]), y=1.)
plt.tight_layout()
plt.savefig('./violins%s.pdf'%LF, bbox_inches='tight')
plt.clf()

f, ax = plt.subplots(1, 2, sharey=True)
plt.subplots_adjust(hspace=2)
#plt.subplots_adjust(bottom=-.7)
bees = len(newd[newd.x==x[-1]]) * 5
percs = np.zeros((bees, 100))
vpercs = np.zeros((bees, 100))
for ii in range(bees):
    msk = np.random.rand(len(newd)) < 0.8
    subset = newd[msk]
    subset = subset[subset.x==subset.x.iloc[-1]]
    for jj in range(percs.shape[1]):
        percs[ii, jj] = np.percentile(subset.y2, jj)
        vpercs[ii, jj] = np.percentile(subset.y, jj)

ax[0].fill_between(np.linspace(0, 100, 100), np.percentile(vpercs, 10, 0), np.percentile(vpercs, 90, 0), facecolor='gray')
ax[1].fill_between(np.linspace(0, 100, 100), np.percentile(percs, 10, 0), np.percentile(percs, 90, 0), facecolor='gray')
ax[1].plot(np.linspace(0, 100, 100), np.mean(percs, 0))
ax[0].plot(np.linspace(0, 100, 100), np.mean(vpercs, 0))
ax[0].set_xlabel('Percentile')
ax[1].set_xlabel('Percentile')
ax[0].set_ylabel('Error, HF')
ax[1].set_ylabel('Error, CV')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.suptitle('%i sets, %i LF samples, %i test HF samples'%(BS, x[-1], dat.shape[0]-x[-1]))
plt.savefig('%i_sets_%i_LF_samples_%i_test_samples.pdf'%(BS, x[-1], dat.shape[0]-x[-1]), bbox_inches='tight')
plt.clf()
