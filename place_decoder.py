import numpy as np
import decoder2 as dec
def locs_to_cats(rs, divs, eps=1e-5):
    rs = rs.dot([[1,-1],[1,1]])/np.sqrt(2)
    rs -= rs.min(0)
    rs /= rs.max(0)
    rs = np.int64(np.trunc(rs*divs-eps))
    return rs[:,0] + rs[:,1]*divs
def one_hot_mats(cats, divs):
    return np.eye(divs**2)[cats].reshape([len(cats), divs, divs])
def evaluate(transients, rs, divs, train_frac, n_shufs, n_batch, lookback, ranges=None):
    cats = locs_to_cats(rs, divs)
    valid_place_cell = detect_valid_place_cells(cats, transients, n_shufs, n_batch)
    transients = transients[valid_place_cell,:]
    print '%d out of %d cells were place cells' % (valid_place_cell.sum(), len(valid_place_cell))
    transients = include_history(transients, lookback)
    pats = transients.T==1
    err, err_by_loc, inferences, test_cats, times = dec.evaluate(cats, divs**2, pats, train_frac, ranges)
    err_by_loc = err_by_loc.reshape([divs, divs])
    return err, err_by_loc, one_hot_mats(inferences, divs), one_hot_mats(test_cats, divs), times


def include_history(transients, lookback=8):
    accum = transients
    for i in xrange(1, lookback+1):
        accum = np.vstack((accum, np.roll(transients, i)))
    return accum


#variable language:
# x : variable of type x
# xs: list of x typed variables
# Nx: the number of possible unique x types
#lxs: length of list of x typed variables
# nx: histogram over the x type
#nx#: histogram entry from the x type of value #
#a_?: variable labeled as belonging to logical category of ?

def mutual_informations(cats, transients):
    """
    Calculate mutual information between categories and the transients in each cell, returns
    the mutual information for each cell
    """
    uniq_cats, clean_cats = np.unique(cats, return_inverse=True)
    Nc = len(uniq_cats)
    nc = np.bincount(clean_cats, minlength=Nc)
    return np.apply_along_axis(lambda c: muti_bin(clean_cats, Nc, c, nc, c.sum()), 1, transients)

def muti_bin(cs, Nc, bs, nc, nb1):
    """
    AUXILLARY FUNCTION
Calculate mutual information between a signal of categories and a signal of binary values.

Inputs:
    cs: list of category values, these should already be cleaned values (at least represented once)
    Nc: number of possible category values
    bs: list of binary values
    nc: histogram of the inputted 'cs' variable. Only works if that is true
    nb1: number of 1's in the binary input

Output:
    the number representing the mutual information between 'cs' and 'bs' in nats
    """
    nc_1 = np.bincount(cs[bs==1], minlength=Nc)
    nc_0 = nc - nc_1

    lbs = len(bs)
    nb0 = lbs - nb1

    mask_1 = nc_1 != 0
    mask_0 = nc_0 != 0

    nc_1_m1 = nc_1[mask_1]
    nc_0_m0 = nc_0[mask_0]

    return 1./lbs * (np.sum(nc_1_m1 * np.log(1.*lbs*nc_1_m1/nc[mask_1]/nb1)) +\
                   np.sum(nc_0_m0 * np.log(1.*lbs*nc_0_m0/nc[mask_0]/nb0)))

def muti_bin2(css, Nc, bs, nc):
    ncs_1 = np.apply_along_axis(lambda x: np.bincount(x, minlength=Nc), 1, css[:, bs==1])
    ncs_0 = nc - ncs_1
    N = len(bs)
    nb1 = bs.sum()
    nb0 = N - nb1
    return 1./N * (np.nansum(ncs_1 * np.log(1.*N*ncs_1/nc/nb1),1) +\
                   np.nansum(ncs_0 * np.log(1.*N*ncs_0/nc/nb0),1))

def detect_valid_place_cells(cats, transients, n_shufs, n_batch, num_cells=None):
    if num_cells is None:
        num_cells = len(transients)
    uniq_cats, clean_cats = np.unique(cats, return_inverse=True)
    nc = np.bincount(clean_cats, minlength=len(uniq_cats))
    mutis = np.apply_along_axis(lambda c: muti_bin2(clean_cats[None,:], len(uniq_cats), c, nc)[0], 1, transients[:num_cells])
    batches = []
    from tqdm import tqdm
    for k in tqdm(xrange(n_batch), desc='batch processing mutual information'):
        permcats = np.vstack([np.random.permutation(clean_cats) for _ in xrange(n_shufs)])
        batches.append(np.vstack([muti_bin2(permcats, len(uniq_cats), tr, nc) for tr in transients[:num_cells]]))
        #print '%d/%d' % (k+1, n_batch),
    #print
    shuf_mutis = np.hstack(batches)
    qs = []
    for i in xrange(len(mutis)):
        muti = mutis[i]
        shufs = shuf_mutis[i]
        p = (shufs >= muti).sum()*1.0/len(shufs)
        q = p * num_cells
        qs.append(q)
    return np.array(qs) < 0.05
