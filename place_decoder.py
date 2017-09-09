import numpy as np
import decoder2 as dec

from joblib import Parallel, delayed
from joblib import Memory
memory = Memory(cachedir='/tmp/tmp_day_loader_joblib', verbose=0)

def locs_to_cats(rs, divs, eps=1e-5):
    rs = rs.dot([[1,-1],[1,1]])/np.sqrt(2)
    rs -= rs.min(0)
    rs /= rs.max(0)
    rs = np.int64(np.trunc(rs*divs-eps))
    return rs[:,0] + rs[:,1]*divs

def one_hot_mats(cats, divs):
    return np.eye(divs**2)[cats].reshape([len(cats), divs, divs])

def evaluate(transients, rs, divs, train_frac, n_shufs, n_batch, lookback, ranges=None):
    n_cells, _ = transients.shape
    cats = locs_to_cats(rs, divs)
    pats, failed_cells = process_transients(transients, cats, n_shufs, n_batch, lookback)
    print '%d out of %d had insignificant mutual information' % (failed_cells, n_cells)
    err, err_by_loc, inferences, test_cats, times = dec.evaluate(cats, divs**2, pats, train_frac, ranges)
    err_by_loc = err_by_loc.reshape([divs, divs])
    return err, err_by_loc, one_hot_mats(inferences, divs), one_hot_mats(test_cats, divs), times


def include_history(transients, lookback=8):
    accum = transients
    for i in xrange(1, lookback+1):
        accum = np.vstack((accum, np.roll(transients, i)))
    return accum

def process_transients(transients, cats, n_shufs, n_batch, lookback):
    valid_place_cell = detect_valid_place_cells(cats, transients, n_shufs, n_batch)
    return include_history(transients[valid_place_cell], lookback).T==1, np.sum(~valid_place_cell)

@memory.cache
def predictor(transients, rs, divs, train_frac, n_shufs, n_batch, lookback, ranges, future=0):
    n_cells, _ = transients.shape
    cats = locs_to_cats(rs, divs)
    n_cats = divs**2
    #pats, failed_cells = process_transients(transients, cats, n_shufs, n_batch, lookback)
    #print '%d out of %d had insignificant mutual information' % (failed_cells, n_cells)
    #future = np.array(future)
    #return np.array([ dec.evaluate(np.roll(cats, -f), n_cats, pats, train_frac, ranges)[0]
    #       for f in future.flat ]).reshape(future.shape)
    future = np.array(future)
    #errs = []
    #for f in future.flat:
    #    rolled_cats = np.roll(cats, -f)
    #    pats, failed_cells = process_transients(transients, rolled_cats, n_shufs, n_batch, lookback)
    #    print '->%d<-%d: %d out of %d had insignificant mutual information' % (f, lookback, failed_cells, n_cells)
    #    errs.append(memory.cache(dec.evaluate)(rolled_cats, n_cats, pats, train_frac, ranges)[0])
    errs = Parallel(n_jobs=3, verbose=1)( delayed(single_predict)(cats, n_cats, transients, train_frac, n_shufs, n_batch, lookback, ranges, n_cells, f) for f in future.flat )
    return np.array(errs).reshape(future.shape)

def single_predict(cats, n_cats, transients, train_frac, n_shufs, n_batch, lookback, ranges, n_cells, f):
    rolled_cats = np.roll(cats, -f)
    pats, failed_cells = process_transients(transients, rolled_cats, n_shufs, n_batch, lookback)
    print '->%d<-%d: %d out of %d had insignificant mutual information' % (f, lookback, failed_cells, n_cells)
    return memory.cache(dec.evaluate)(rolled_cats, n_cats, pats, train_frac, ranges)[0]

@memory.cache
def self_predictor(rs, divs, train_frac, n_shufs, n_batch, lookback, ranges, future=0):
    cats = locs_to_cats(rs, divs)
    n_cats = divs**2
    n_cells = n_cats
    transients = np.eye(n_cats)[:,cats]
    #pats, failed_cells = process_transients(transients, cats, n_shufs, n_batch, lookback)
    #print '%d out of %d had insignificant mutual information' % (failed_cells, n_cats)
    #future = np.array(future)
    #return np.array([ dec.evaluate(np.roll(cats, -f), n_cats, pats, train_frac, ranges)[0]
    #       for f in future.flat ]).reshape(future.shape)
    future = np.array(future)
    #errs = []
    #for f in future.flat:
    #    rolled_cats = np.roll(cats, -f)
    #    pats, failed_cells = process_transients(transients, rolled_cats, n_shufs, n_batch, lookback)
    #    print '*>%d,<*%d: %d out of %d had insignificant mutual information' % (f, lookback, failed_cells, n_cells)
    #    errs.append(memory.cache(dec.evaluate)(rolled_cats, n_cats, pats, train_frac, ranges)[0])
    errs = Parallel(n_jobs=3, verbose=1)( delayed(single_predict)(cats, n_cats, transients, train_frac, n_shufs, n_batch, lookback, ranges, n_cells, f) for f in future.flat )
    return np.array(errs).reshape(future.shape)

#variable language:
# x : variable of type x
# xs: list of x typed variables
# Nx: the number of possible unique x types
#lxs: length of list of x typed variables
# nx: histogram over the x type
#nx#: histogram entry from the x type of value #
#a_?: variable labeled as belonging to logical category of ?
def muti_bin2(css, Nc, bs, nc):
    ncs_1 = np.apply_along_axis(lambda x: np.bincount(x, minlength=Nc), 1, css[:, bs==1])
    ncs_0 = nc - ncs_1
    N = len(bs)
    nb1 = bs.sum()
    nb0 = N - nb1
    return 1./N * (np.nansum(ncs_1 * np.log(1.*N*ncs_1/nc/nb1),1) +\
                   np.nansum(ncs_0 * np.log(1.*N*ncs_0/nc/nb0),1))

def runbatch(clean_cats, n_shufs, uniq_cats, nc, transients, num_cells):
    permcats = np.vstack([np.random.permutation(clean_cats) for _ in xrange(n_shufs)])
    return np.vstack([muti_bin2(permcats, len(uniq_cats), tr, nc) for tr in transients[:num_cells]])


@memory.cache
def detect_valid_place_cells(cats, transients, n_shufs, n_batch, num_cells=None):
    if num_cells is None:
        num_cells = len(transients)
    uniq_cats, clean_cats = np.unique(cats, return_inverse=True)
    nc = np.bincount(clean_cats, minlength=len(uniq_cats))
    mutis = np.apply_along_axis(lambda c: muti_bin2(clean_cats[None,:], len(uniq_cats), c, nc)[0], 1, transients[:num_cells])
    batches = Parallel(n_jobs=1, verbose=1)(delayed(runbatch)(clean_cats, n_shufs, uniq_cats, nc, transients, num_cells) for k in xrange(n_batch))
    shuf_mutis = np.hstack(batches)
    qs = []
    for i in xrange(len(mutis)):
        muti = mutis[i]
        shufs = shuf_mutis[i]
        p = (shufs >= muti).sum()*1.0/len(shufs)
        q = p * num_cells
        qs.append(q)
    return np.array(qs) < 0.05
