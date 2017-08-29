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

def evaluate(transients, rs, divs, train_frac, ranges=None):
    cats = locs_to_cats(rs, divs)
    pats = transients.T==1
    err, err_by_loc, inferences, test_cats, times = dec.evaluate(cats, divs**2, pats, train_frac, ranges)
    err_by_loc = err_by_loc.reshape([divs, divs])
    return err, err_by_loc, one_hot_mats(inferences, divs), one_hot_mats(test_cats, divs), times
