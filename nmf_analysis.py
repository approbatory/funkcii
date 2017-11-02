from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from day_loader import process_subdirs, l1_err
import place_decoder
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
import numpy as np
from joblib import Memory
import scipy.ndimage as ndimage

class Opt:
    pass
opt = Opt()
opt.DIVS=11
opt.LOOKBACK=7
opt.FUTURE=51
opt.TRAIN_FRACTION=0.5
opt.N_SHUFS=500
opt.N_BATCH=20

memory = Memory(cachedir='/tmp/tmp_day_loader_joblib', verbose=0)

def genmap(xy_, imshape, sigma, mask=None, normwith=None, reg=False):
    if mask is None:
        inds = (xy_[:,1], xy_[:,0])
    else:
        inds = (xy_[mask,1], xy_[mask,0])
    image = np.zeros(imshape)
    np.add.at(image, inds, 1)
    if reg:
        image += 1
    image = ndimage.gaussian_filter(image, sigma=sigma)
    if normwith is not None:
        image /= normwith
    return image

def place_maps(transients, xy, t_ranges, dirname, opt, label='unlabeled'):
    cell_inds = transients.sum(1).argsort()[::-1]
    #downsample (subsample) every 4 pixels
    DOWNSAMP = 4
    #maxima and minima
    min_x, max_x = xy[:,0].min(), xy[:,0].max()
    min_y, max_y = xy[:,1].min(), xy[:,1].max()
    #converting xy positions to indices in an image
    xy_ = np.int64(np.round(xy - xy.min(0)))/DOWNSAMP
    imshape = xy_.max(0)[::-1] + 1
    #one cm was 8 px, now reduced
    cm = 8/DOWNSAMP
    #using a 3.5cm gaussian blur
    sigma = 3.5*cm
    #generating occupancy map
    image = genmap(xy_, imshape, sigma, reg=True)
    #plot the map
    ###plt.figure()
    ###plt.matshow(image)
    ###plt.title('total occupancy in %s' % label)
    occ_image = image
    #save results to output
    saver = opt.savers[label] if hasattr(opt, 'savers') else {cell_ind : genmap(xy_, imshape, sigma, mask= transients[cell_ind] == 1, normwith=occ_image) for cell_ind in cell_inds}
    ###swatch_plot(saver, 'place maps %s' % label, order=cell_inds)
    return label, saver

def swatch_plot(swatches, title, order=None, myrows=None, mycols=None):
    if len(swatches) == 0:
        return
    if len(swatches) == 1:
        ind = -1
        for k in swatches:
            ind = k
            swatch = swatches[k]
        #plt.figure()
        plt.matshow(swatch)
        plt.axis('off')
        plt.title(title + ' : %d' % ind)
        return
    #cols, rows = np.int64(np.sqrt(len(swatches))+0.999), np.int64(0.999+1.0*len(swatches)/np.int64(np.sqrt(len(swatches)) + 0.999))
    num_swatches = len(swatches) if order is None else len(order)
    cols = min([num_swatches, 11]) if mycols is None else mycols
    rows = max([np.int64((1.0*num_swatches)/cols + 0.9999), 2]) if myrows is None else myrows
    f, axs = plt.subplots(rows, cols, sharex='col', sharey='row')
    if rows > 1:
        f.set_size_inches(cols,rows*1.8)
    else:
        f.set_size_inches(cols/0.75, rows/0.75)
    f.suptitle(title, fontsize=24) if title is not None else None
    looper = swatches.keys() if order is None else order
    for ix, cell_ind in enumerate(looper):
        if cols == 1 or rows == 1:
            ax = axs[ix]
        else:
            ax = axs[ix/cols,ix%cols]
        ax.matshow(swatches[cell_ind])
        ax.set_title('%d' % cell_ind)
    for ax in axs.flat: ax.axis('off')

N_COMP = 17
def perform_nmf(n, X):
    model = NMF(n_components=n)
    W = model.fit_transform(X)
    H = model.components_
    return (W,H)




def peaks_analysis(W,H, hshape, subsets):
    transitions = [[] for subset in subsets]
    one_true_block = lambda bs: 2==np.sum(np.diff(np.hstack([False,bs,False])))
    for i,w in enumerate(W):
        subsums = np.array(map(lambda ss:np.sum(w[ss]), subsets))
        subset_indices = np.flatnonzero(subsums == subsums.max())
        for subset_index in subset_indices:
            ss = subsets[subset_index]
            wsub = w[ss]
            mean_val = np.mean(wsub)
            passing = wsub > mean_val
            if one_true_block(passing):
                peak_only = wsub * (wsub > mean_val)
                cm = np.sum(peak_only * np.arange(len(ss))) / np.sum(peak_only)
                transitions[subset_index].append((i,cm))
    for transition in transitions:
        transition.sort(key=lambda x:x[1])
    unclean = [i for i,w in enumerate(W) if all(i not in map(lambda x:x[0],transition) for transition in transitions)]
    return map(lambda x: zip(*x), transitions), unclean



def inspect_days(dir_path):
    savers = process_subdirs(dir_path, place_maps, opt)
    opt.savers = {k:v for (k,v) in savers}
    for label in opt.savers:
        maps = opt.savers[label]
        hshape = maps[0].shape
        X = np.vstack([maps[k].ravel() for k in range(len(maps))])
        #TODO choose k (n_comp) based on first k of no improvement
        W, H = perform_nmf(N_COMP, X)
        #ad hoc:change later TODO
        if label == 'allo':
            subsets = [[2,9,11,3,14,0,5,6], [4,15,12,1,16,7,0,5,6], [10,13,8]]
            subset_names = ['top', 'bottom', 'out']
        elif label == 'ego':
            ord_top = []
            ord_bot = []
            for i, h in enumerate(H):
                h = h.reshape(hshape)
                y_cm = (h * np.arange(hshape[0])[:,None]).sum()/h.sum()
                x_cm = (h * np.arange(hshape[1])[None,:]).sum()/h.sum()
                if y_cm < hshape[0]/2:
                    ord_top.append((i,x_cm))
                else:
                    ord_bot.append((i,10*hshape[1] - x_cm))
            ord_top.sort(key=lambda x:x[1])
            ord_bot.sort(key=lambda x:x[1])
            subsets = [map(lambda x:x[0], ord_top), map(lambda x:x[0], ord_bot)]
            subset_names = ['top', 'bottom']
        elif label == 'ego_to_allo':
            subsets = [[3, 9, 15, 7, 12, 13, 2, 8], [4, 11, 6, 0, 16, 13, 2, 8], [1, 14, 5, 10]]
            subset_names = ['tl2bl', 'br2bl', 'out']
        else:
            swatch_plot({i : h.reshape(hshape) for i,h in enumerate(H)}, 'components')
            plt.show()
            subsets = eval(raw_input('Split ' + label + 'into subsets: '))
            subset_names = eval(raw_input(label + ': Give them names in a list: '))
        subset_maps, unclean = peaks_analysis(W, H, hshape, subsets)
        for name, (order, cms) in zip(subset_names, subset_maps):
            plt.figure(); plt.plot(cms, 'x'); plt.grid(True); plt.title('%s: %s cm plot' % (label,name))
            swatch_plot(maps, '%s: %s' % (label, name), order=order)
        #swatch_plot(maps, 'unclean', order=unclean)
        #plt.show()
    plt.show()

if __name__ == "__main__":
    inspect_days('../hpc/assets')
