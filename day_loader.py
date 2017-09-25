import h5py
import numpy as np
from glob import glob
import place_decoder
import transient_detection as tdet
import pandas as pd
import visual_analyzer
import matplotlib.pyplot as plt
from joblib import Memory
import yaml

DIVS = 11
TRAIN_FRACTION = 0.5
MAKE_MOVIE = False
N_SHUFS = 5000
N_BATCH = 20
LOOKBACK = 8
FUTURE = 10

memory = Memory(cachedir='/tmp/tmp_day_loader_joblib', verbose=0)

def get_by_ext(dirname, ext):
    fs = list(glob(dirname + '/*.' + ext))
    if len(fs) != 1:
        raise ValueError('%s must have 1 .%s file' % (dirname, ext))
    return fs[0]

def load_from_dir(dirname):
    meta = yaml.load(open(dirname+'/meta.yaml'))

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

    return real_traces, xy_pos, t_ranges, meta

def process_dir_proc(dirname, proc):
    print "Loading files..."
    traces, xy, t_ranges, meta = load_from_dir(dirname)
    print "Detecting transients..."
    transients = memory.cache(tdet.detect)(traces)
    proc(transients, xy, t_ranges, dirname, label=meta['day_type'])

def my_eval_proc(transients, xy, t_ranges, dirname, label='unlabeled'):
    print "Evaluating decoder..."
    err, errmat, inference_mats, actual_mats, times, encoded_data =\
        memory.cache(place_decoder.evaluate)(transients, xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, LOOKBACK, t_ranges)
    #np.random.shuffle(transients.T)
    #base_err = place_decoder.evaluate(transients, xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, LOOKBACK, t_ranges)[0]
    print 'avg err rate is %f%%' % (100*err)
    #print 'baseerr rate is %f%%' % (100*base_err)
    errmat[np.isnan(errmat)] = -0.01
    print np.int64(np.round(errmat*100))
    print 'calculated from %d trials (nonprobe)' % len(t_ranges)
    visual_analyzer.viz_encoding(encoded_data, dirname, label)
    print 'made visualization for encoded data'
    if MAKE_MOVIE:
        print "making movie..."
        visual_analyzer.make_movie(inference_mats, actual_mats, times, DIVS,\
                                                 dirname+'/decoded.mp4')
        print 'made movie'
    print

def future_prediction(transients, xy, t_ranges, dirname, label='unlabeled'):
    print 'Running future prediction...'
    future = np.arange(FUTURE)
    plt.figure()
    for lb in xrange(LOOKBACK):
        frac = lb*1./LOOKBACK
        self_errs = place_decoder.self_predictor(xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, lb, t_ranges, future)
        errs = place_decoder.predictor(transients, xy, DIVS, TRAIN_FRACTION, N_SHUFS, N_BATCH, lb, t_ranges, future)
        plt.plot(future, self_errs, '-o', color=(1-frac,0,frac), label=('$%d \leftarrow_s$' % lb))
        plt.plot(future, errs, '-x', color=(0,1-frac,frac), label=('$%d \leftarrow$' % lb))
    plt.ylim([0,1])
    plt.xlabel('frames ahead to predict')
    plt.ylabel('prediction error')
    plt.title('Future prediction / self prediction errors in session %s' % label)
    plt.legend(loc='best', ncol=LOOKBACK*2/3)
    plt.savefig(dirname+'/future_self_pred.png')


def process_subdirs(dirname, proc_to_run):
    for d in glob(dirname + '/*'):
        print 'Now entering into directory "%s" :::::' % d
        process_dir_proc(d, proc_to_run)


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
    parser.add_option("-S", "--number-of-shuffles", dest="n_shuf_batch", default=(500+20j),
            help="USE N_SHUFS+N_BATCH*j shuffles per batch for calculating mutual information p-values [default 500 shuffles 20 batches]", metavar="N_SHUFS", type="complex")
    parser.add_option("--future", dest="future", default=-1,
            help="Run a future decoding up to FRAMES frames ahead [default: -1 (off)]", metavar='FRAMES', type="int")
    (options, args) = parser.parse_args()
    MAKE_MOVIE = options.movie
    DIVS = options.divs
    TRAIN_FRACTION = options.train_fraction
    N_SHUFS = int(options.n_shuf_batch.real)
    N_BATCH = int(options.n_shuf_batch.imag)
    if options.future >= 0:
        FUTURE = options.future
        proc_to_run = future_prediction
    else:
        proc_to_run = my_eval_proc
    process_subdirs(options.directory, proc_to_run)
