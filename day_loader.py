#!/usr/bin/env python
#base datatypes and job management
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory

#packaged modules
import place_decoder
import transient_detection as tdet
import visual_analyzer

#for file loaders/writers
import yaml
import os
import re
from glob import glob
import h5py

memory = Memory(cachedir='/tmp/tmp_day_loader_joblib', verbose=0)

def load_from_dir(dirname):
    files = [os.path.join(root,f) for root, dirs, fs in os.walk(dirname) for f in fs]
    def ff(exp):
        matches = [f for f in files if re.search(exp, f)]
        if len(matches) != 1:
            raise ValueError('%s must have 1 %s matching file' % (dirname, exp))
        return matches[0]
    traces_fname = ff(r'rec_.*\.mat$')
    real_cells_fname = ff(r'class_.*\.txt$')
    daycode_re = r'c(\d+)m(\d+)d(\d+)_ti2'
    xy_pos_fname = ff(daycode_re + r'\.xy')
    props_fname = ff(daycode_re + r'\.txt')

    matfile = h5py.File(traces_fname, 'r')
    traces = np.array(matfile.get('traces'))
    real_cells = np.array(['not' not in line
        for line in open(real_cells_fname).read().split('\n') if 'cell' in line])
    real_traces = traces[real_cells, :]
    xy_pos = np.loadtxt(xy_pos_fname)
    props = pd.read_csv(props_fname, sep=' ',\
            names=['initial', 'goal', 'final', 'time', 'begin',\
            'open', 'close', 'end'])
    no_probes = props[(props.initial=='east') | (props.initial=='west')]
    t_ranges = zip(no_probes.open, no_probes.close)

    daycode = re.search(daycode_re, xy_pos_fname)
    meta = dict(cohort=daycode.group(1), mouse=daycode.group(2), day=daycode.group(3))
    part1, part2 = no_probes.goal[:50], no_probes.goal[-50:]
    find_type = lambda p: 'allo' if all(p == 'north') or all(p == 'south') else 'ego'
    type1, type2 = find_type(part1), find_type(part2)
    if type1 == type2:
        meta['day_type'] = type1
    else:
        meta['day_type'] = "%s_to_%s" % (type1, type2)
    return real_traces, xy_pos, t_ranges, meta

def process_dir_proc(dirname, proc, opt):
    print "Loading files..."
    traces, xy, t_ranges, meta = load_from_dir(dirname)
    if not os.path.isfile(os.path.join(dirname,'meta.yaml')):
        yaml.dump(meta, open(os.path.join(dirname, 'meta.yaml'),'w'))
    print meta
    print "Detecting transients..."
    transients = memory.cache(tdet.detect)(traces)
    if opt.send_original_traces:
        opt.original_traces = traces
    return proc(transients, xy, t_ranges, dirname, opt, label=meta['day_type'])

def l1_err(inf_mats, act_mats, t_ranges, times):
    dists = 0
    count = 0
    untimes = {t:i for i,t in enumerate(times)}
    #for im, am in zip(inf_mats, act_mats):
    for (s,e) in t_ranges:
        for t in xrange(s,e):
            if t in untimes:
                im = inf_mats[untimes[t]]
                am = act_mats[untimes[t]]
                xi, yi = im.nonzero()
                xa, ya = am.nonzero()
                xi, yi, xa, ya = map(lambda x: x[0], [xi, yi, xa, ya])
                dists += abs(xi-xa) + abs(yi-ya)
                count += 1
    return 1.0*dists/count

def my_eval_proc(transients, xy, t_ranges, dirname, opt, label='unlabeled'):
    print "Evaluating decoder..."
    err, errmat, inference_mats, actual_mats, times, encoded_data =\
        memory.cache(place_decoder.evaluate)(transients, xy, opt.DIVS, opt.TRAIN_FRACTION, opt.N_SHUFS, opt.N_BATCH, opt.LOOKBACK, t_ranges)
    print 'avg err rate is %f%%' % (100*err)
    errmat[np.isnan(errmat)] = -0.01
    print np.int64(np.round(errmat*100))
#calculate l1 error in terms of divisions of distance
    avg_dist_err = l1_err(inference_mats, actual_mats, t_ranges, times)
    print "Average number of squares of distance away (L1):", avg_dist_err
    print 'calculated from %d trials (nonprobe)' % len(t_ranges)
    visual_analyzer.viz_encoding(encoded_data, dirname, label)
    print 'made visualization for encoded data'
    if opt.MAKE_MOVIE:
        print "making movie..."
        visual_analyzer.make_movie(inference_mats, actual_mats, times, opt.DIVS,\
                                                 dirname+'/decoded.mp4')
        print 'made movie'
    print

def future_prediction(transients, xy, t_ranges, dirname, opt, label='unlabeled'):
    print 'Running future prediction...'
    future = np.arange(opt.FUTURE)
    plt.figure()
    for lb in xrange(opt.LOOKBACK):
        frac = lb*1./opt.LOOKBACK
        self_errs = place_decoder.self_predictor(xy, opt.DIVS, opt.TRAIN_FRACTION, opt.N_SHUFS, opt.N_BATCH, lb, t_ranges, future)
        errs = place_decoder.predictor(transients, xy, opt.DIVS, opt.TRAIN_FRACTION, opt.N_SHUFS, opt.N_BATCH, lb, t_ranges, future)
        plt.plot(future, self_errs, '-o', color=(1-frac,0,frac), label=('$%d \leftarrow_s$' % lb))
        plt.plot(future, errs, '-x', color=(0,1-frac,frac), label=('$%d \leftarrow$' % lb))
    plt.ylim([0,1])
    plt.xlabel('frames ahead to predict')
    plt.ylabel('prediction error')
    plt.title('Future prediction / self prediction errors in session %s' % label)
    plt.legend(loc='best', ncol=opt.LOOKBACK*2/3)
    plt.savefig(dirname+'/future_self_pred.png')

def process_subdirs(dirname, proc_to_run, opt):
    res = []
    for d in glob(os.path.join(dirname, '*')):
        if os.path.isdir(d):
            print 'Now entering into directory "%s" :::::' % d
            res.append(process_dir_proc(d, proc_to_run, opt))
    return res

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--movie", action="store_true", dest="MAKE_MOVIE", default=False,
            help="Generate movies using FFMPEG [default: False]")
    parser.add_option("-d", "--directory", dest="directory",
            default="", help="Open all directories under DIR [default .]", metavar="DIR", type="string")
    parser.add_option("-D", "--divisions", dest="DIVS", default=11,
            help="Turn space into a DIVS x DIVS grid [default 11]", metavar="DIVS", type="int")
    parser.add_option("-f", "--train-fraction", dest="TRAIN_FRACTION", default=0.5,
            help="Use FRAC fraction of the data for the training set [default 0.5]", metavar="FRAC", type="float")
    parser.add_option("-S", "--number-of-shuffles", dest="n_shuf_batch", default=(500+20j),
            help="USE N_SHUFS+N_BATCH*j shuffles per batch for calculating mutual information p-values [default 500 shuffles 20 batches]", metavar="N_SHUFS", type="complex")
    parser.add_option("--future", dest="FUTURE", default=-1,
            help="Run a future decoding up to FRAMES frames ahead [default: -1 (off)]", metavar='FRAMES', type="int")
    (options, args) = parser.parse_args()
    options.N_SHUFS = int(options.n_shuf_batch.real)
    options.N_BATCH = int(options.n_shuf_batch.imag)
    if options.FUTURE >= 0:
        proc_to_run = future_prediction
    else:
        proc_to_run = my_eval_proc
    options.LOOKBACK = 8
    if options.directory == '' and len(args) > 0:
        options.directory = args[0]
    else:
        options.directory = '.'
    process_subdirs(options.directory, proc_to_run, options)
