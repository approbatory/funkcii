import h5py
import numpy as np
from glob import glob
import place_decoder
import transient_detection as tdet
import pandas as pd
import visual_analyzer
from joblib import Memory

DIVS = 11
TRAIN_FRACTION = 0.5
MAKE_MOVIE = False
N_SHUFS = 5000
N_BATCH = 20
LOOKBACK = 8

memory = Memory(cachedir='/tmp/tmp_day_loader_joblib', verbose=0)

def get_by_ext(dirname, ext):
    fs = list(glob(dirname + '/*.' + ext))
    if len(fs) != 1:
        raise ValueError('%s must have 1 .%s file' % (dirname, ext))
    return fs[0]

def load_from_dir(dirname):
    matfile = h5py.File(get_by_ext(dirname+'/cm01', 'mat'),'r')
    traces = np.array(matfile.get('traces'))
    print 'loaded traces', traces.shape

    real_cells = np.array(['not' not in line
        for line in open(get_by_ext(dirname+'/cm01', 'txt')).read().split('\n') if 'cell' in line])
    real_traces = traces[real_cells, :]
    print 'confirmed_cell_traces', real_traces.shape

    xy_pos = np.loadtxt(get_by_ext(dirname+'/_data', 'xy'))
    print 'xy positions per frame', xy_pos.shape

    props = pd.read_csv(get_by_ext(dirname+'/_data', 'txt'), sep=' ',\
            names=['initial', 'goal', 'final', 'time', 'begin',\
            'open', 'close', 'end'])
    print 'props loaded, len=%d' % len(props)
    no_probes = props[(props.initial=='east') | (props.initial=='west')]
    t_ranges = zip(no_probes.open, no_probes.close)

    return real_traces, xy_pos, t_ranges

def process_dir_proc(dirname, proc):
    print "Loading files..."
    traces, xy, t_ranges = load_from_dir(dirname)
    print "Detecting transients..."
    transients = memory.cache(tdet.detect)(traces)
    proc(transients, xy, t_ranges)

def my_eval_proc(transients, xy, t_ranges):
    print "Evaluating decoder..."
    err, errmat, inference_mats, actual_mats, times =\
        memory.cache(place_decoder.evaluate)(transients, xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, LOOKBACK, t_ranges)
    #np.random.shuffle(transients.T)
    #base_err = place_decoder.evaluate(transients, xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, LOOKBACK, t_ranges)[0]
    print 'avg err rate is %f%%' % (100*err)
    #print 'baseerr rate is %f%%' % (100*base_err)
    errmat[np.isnan(errmat)] = -0.01
    print np.int64(np.round(errmat*100))
    print 'calculated from %d trials (nonprobe)' % len(t_ranges)
    if MAKE_MOVIE:
        print "making movie..."
        visual_analyzer.make_movie(inference_mats, actual_mats, times, DIVS,\
                                                 dirname+'/decoded.mp4')
        print 'made movie'
    print

def sanity_checker(transients, xy, t_ranges):
    print 'Running sanity check...'
    err = place_decoder.self_predictor(xy, DIVS, TRAIN_FRACTION)
    print 'The error was %f%%, it should be 0%%' % (100*err)
######TODO: Augment this to do future prediction sanity checks, as well as full future prediction


def process_subdirs(dirname):
    for d in glob(dirname + '/*'):
        print 'Now entering into directory "%s" :::::' % d
        process_dir_proc(d, PROC_TO_RUN)


PROC_TO_RUN = my_eval_proc
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--movie", action="store_true", dest="movie", default=False,
            help="Generate movies using FFMPEG [default: False]")
    parser.add_option("-d", "--directory", dest="directory",
            default=".", help="Open all directories under DIR [default .]", metavar="DIR", type="string")
    parser.add_option("-D", "--divisions", dest="divs", default=11,
            help="Turn space into a DIVS x DIVS grid [default 11]", metavar="DIVS", type="int")
    parser.add_option("-f", "--train-fraction", dest="train_fraction", default=0.5,
            help="Use FRAC fraction of the data for the training set [default 0.5]", metavar="FRAC", type="float")
    parser.add_option("-S", "--number-of-shuffles", dest="n_shuf_batch", default=(5000+20j),
            help="USE N_SHUFS+N_BATCH*j shuffles per batch for calculating mutual information p-values [default 5000 shuffles 20 batches]", metavar="N_SHUFS", type="complex")
    parser.add_option("--sanity-check", action="store_true", dest="sanity_check", default=False,
            help="Run a sanity check instead of decoding [default: False]")
    (options, args) = parser.parse_args()
    MAKE_MOVIE = options.movie
    DIVS = options.divs
    TRAIN_FRACTION = options.train_fraction
    N_SHUFS = int(options.n_shuf_batch.real)
    N_BATCH = int(options.n_shuf_batch.imag)
    if options.sanity_check:
        PROC_TO_RUN = sanity_checker
    process_subdirs(options.directory)
