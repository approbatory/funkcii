import h5py
import numpy as np
from glob import glob
import place_decoder
import transient_detection as tdet
import pandas as pd
import visual_analyzer

DIVS = 13
TRAIN_FRACTION = 0.5

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

def process_dir(dirname):
    print "Loading files..."
    traces, xy, t_ranges = load_from_dir(dirname)
    print "Detecting transients..."
    transients = tdet.detect(traces)
    print "Evaluating decoder..."
    err, errmat, inference_mats, actual_mats, times =\
        place_decoder.evaluate(transients, xy, DIVS, TRAIN_FRACTION, t_ranges)
    print 'avg err rate is %f%%' % (100*err)
    errmat[np.isnan(errmat)] = -0.01
    print np.int64(np.round(errmat*100))
    print 'calculated from %d trials (nonprobe)' % len(t_ranges)
    print "making movie..."
    visual_analyzer.make_movie(inference_mats, actual_mats, times, DIVS,\
                                                 dirname+'/decoded.mp4')
    print 'made movie'
    print

def process_subdirs(dirname):
    for d in glob(dirname + '/*'):
        print 'Now entering into directory "%s" :::::' % d
        process_dir(d)

if __name__ == '__main__':
    process_subdirs('.')
