import numpy as np
import numpy.ma as ma
def evaluate(cats, n_cats, pats, train_frac, ranges=None):
    test_time = int(train_frac * len(cats))
    train_cats, test_cats = cats[:test_time], cats[test_time:]
    train_pats, test_pats = pats[:test_time], pats[test_time:]
    inferences = infer2(train_cats, test_cats, train_pats, test_pats)
    ranges = [(0          ,len(cats)-test_time)] if ranges is None else\
             [(s-test_time,        e-test_time)  for (s,e) in ranges if s>=test_time]
    mistakes, count = 0, 0
    mistakes_by_loc, count_by_loc = 0, 0
    for (s,e) in ranges:
        inf = inferences[s:e]
        act =  test_cats[s:e]
        mistakes += np.sum(inf != act)
        count += len(act)
        one_hot = np.eye(n_cats)[act]
        mistakes_by_loc += np.sum(one_hot * (inf != act)[:,np.newaxis],0)
        count_by_loc    += np.sum(one_hot                             ,0)
    return (mistakes*1.0/count), (ma.array(mistakes_by_loc)*1.0/count_by_loc).filled(-0.01),\
            inferences, test_cats, np.arange(test_time, len(cats))
def infer(train_cats, test_cats, train_pats, test_pats):
    cat_labels = np.array(sorted(list(set(train_cats))))
    cat_freqs  = np.array([np.sum(           train_cats==c ,0) for c in cat_labels])
    ev_x_cat   = np.array([np.sum(train_pats[train_cats==c],0) for c in cat_labels]).T
    lc   = np.log((cat_freqs            + 1.0)/(len(train_cats) + cat_labels.size))
    legc = np.log((            ev_x_cat + 1.0)/(cat_freqs       +               2))
    lngc = np.log((cat_freqs - ev_x_cat + 1.0)/(cat_freqs       +               2))
    logprobs = np.apply_along_axis(lambda p: np.sum(legc[p],0),1, test_pats) +\
               np.apply_along_axis(lambda p: np.sum(lngc[p],0),1,~test_pats) + lc
    return cat_labels[np.argmax(logprobs, 1)] #inferences

def encode(train_cats, train_pats):
    cat_labels = np.array(sorted(list(set(train_cats))))
    cat_freqs  = np.array([np.sum(           train_cats==c ,0) for c in cat_labels])
    ev_x_cats  = np.array([np.sum(train_pats[train_cats==c],0) for c in cat_labels]).T
    pc   = (cat_freqs + 1.0)/(len(train_cats) + cat_labels.size)
    pegc = (ev_x_cats + 1.0)/(cat_freqs       +               2)
    return cat_labels, pc, pegc

def decode(cat_labels, pc, pegc, test_cats, test_pats):
    lc   = np.log(pc)
    legc = np.log(pegc)
    lngc = np.log(1-pegc)
    logprobs = np.apply_along_axis(lambda p: np.sum(legc[p],0) + np.sum(lngc[~p],0), 1, test_pats) + lc
    return cat_labels[np.argmax(logprobs, 1)] #inferences

def infer2(train_cats, test_cats, train_pats, test_pats):
    cat_labels, pc, pegc = encode(train_cats, train_pats)
    return decode(cat_labels, pc, pegc, test_cats, test_pats)
