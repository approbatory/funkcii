import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
def make_movie(inference_mats, actual_mats, times, divs, fname):
    fig, ax = plt.subplots()
    blank = np.zeros((divs,divs))
    mat = ax.matshow(blank, vmin=0, vmax=1)
    def animate(i):
        buf = actual_mats[i]
        buf[inference_mats[i]==1] = 0.5
        mat.set_data(buf)
        plt.title('time : %d' % times[i])
        if i%1000==0: print i,'/',len(times),'   ',
        return mat,
    def init():
        mat.set_data(np.ma.array(blank, mask=True))
        return mat,
    ani = anim.FuncAnimation(fig, animate, np.arange(len(times)),\
            init_func=init, interval=25, blit=True)
    mywriter = anim.FFMpegWriter(fps=25)
    ani.save(fname, writer=mywriter)

def viz_encoding(encoded_data, dirname, label):
    import os
    fname = os.path.join(dirname, 'viz_enc_'+label+'.png')
    cat_labels, pc, pegc = encoded_data
    plt.figure()
    pegc = pegc[pegc.sum(1).argsort()]
    img = plt.matshow(pegc, vmin=0, vmax=1, aspect='auto')
    plt.ylabel('event')
    plt.xlabel('category')
    plt.colorbar()
    plt.title('%s session hit rates' % label)
    plt.savefig(fname)

