import numpy as np
import scipy.signal as sig


def subtract_median_window(traces, window_size=101):
    return traces - sig.medfilt(traces, [1,window_size])
def sliding_average(traces, window_size=3):
    return sig.convolve(traces, np.ones((1,window_size), dtype=int), 'same')/(1.*window_size)
def detect_maxima(traces, z=2, width=5, sep=3):
    threshold = z*np.reshape(np.std(traces, axis=-1), (-1,1))
    passing_threshold = traces >= threshold
    repeated_passing = sig.convolve(passing_threshold, np.ones((1,width)), 'same') >= width
    local_maximum = np.pad(np.diff(np.int64(np.diff(traces) > 0)) < 0, ((0,0),(1,1)), 'constant', constant_values=(0,0))
    well_sep_max = local_maximum
    for i in range(sep+1):
        well_sep_max &= ~np.roll(local_maximum,i+1)
    valid_transient_peak = repeated_passing & well_sep_max
    return valid_transient_peak
def process_transient_dots(valid_peaks, traces, offset_delay=3):
    rows, cols = valid_peaks.nonzero()
    peak_vals = traces[rows, cols]
    num_peaks = len(cols)
    transient_occurance_cols = []
    for ix in xrange(num_peaks):
        i, j = rows[ix], cols[ix]
        dropping = True
        old_val = traces[i,j]
        while dropping and j>0:
            j-=1
            new_val = traces[i,j]
            if new_val > old_val:
                dropping = False
            old_val = new_val
        # transient occurance is midpoint of peak & previous trough, pushed back due to GCaMP delay
        transient_occurance_cols.append((j + cols[ix])/2 - offset_delay)
        #rise_time = cols[ix] - j
    return rows, np.array(transient_occurance_cols), peak_vals #rows, cols, and values at peaks (not at transient start)

def expand_transients(valid_peaks, traces, offset_delay=3):
    rows, cols = valid_peaks.nonzero()
    num_peaks = len(cols)
    transient_canvas = np.zeros_like(valid_peaks, dtype=np.int)
    for ix in xrange(num_peaks):
        i, j = rows[ix], cols[ix]
        dropping = True
        old_val = traces[i,j]
        while dropping and j > 0:
            j-=1
            new_val = traces[i,j]
            if new_val > old_val:
                dropping = False
            old_val = new_val
        start_of_transient = j - offset_delay
        end_of_transient = cols[ix] - offset_delay
        transient_canvas[i,start_of_transient:end_of_transient] = 1
    return transient_canvas

def include_history(transients, lookback=8):
    accum = transients
    for i in xrange(1, lookback+1):
        accum = np.vstack((accum, np.roll(transients, i)))
    return accum

def detect(real_traces, is_sparse=False):
    transients = np.copy(real_traces)
    transients_sub_med = subtract_median_window(transients)
    transients_cleaned = sliding_average(transients_sub_med)
    transients_maxima = detect_maxima(transients_cleaned)
    transient_canvas = expand_transients(transients_maxima, transients_cleaned)
    ####transient_history = include_history(transient_canvas)
    transient_history = transient_canvas
    #transient_dots = process_transient_dots(transients_maxima, transients_cleaned)
    #cell_ids, time_inds, magnitudes = transient_dots

    #return transient_dots
    if is_sparse:
        cell_ids, time_inds = transient_history.nonzero()
        return cell_ids, time_inds, None
    else:
        return transient_history
