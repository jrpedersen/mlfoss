# Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from numpy import linalg as LA
import math
from sklearn.metrics import confusion_matrix
#from pdf2image import convert_from_path


def save_readme(experiment_name, args_dict):
    os.makedirs('experiments/'+experiment_name, exist_ok=True)
    f = open(   'experiments/'+experiment_name+'/readme.txt', 'w+' )
    f.write( 'BATCH_SIZE, ' + str(BATCH_SIZE) + '\n' )
    f.write( 'STart learning rate ' + str(lr) + '\n' )
    f.write( 'log_interval, ' + str(log_interval) + '\n' )
    f.write( 'BATCH_SIZE, ' + str(BATCH_SIZE) + '\n' )
    f.write( 'BATCH_SIZE, ' + str(BATCH_SIZE) + '\n' )
    f.close()
    return None

def cyclic_learningrate(lr_original,lr):
    return 10**(np.log10(lr_original)-((-np.log10(lr) + np.log10(lr_original)+1) % 3))
def cyclic_learningrate2(lr_original,lr):
    work_number = (work_number)
    return 10**(np.log10(lr_original)-((-np.log10(lr) + np.log10(lr_original)+1) % 3))

def cosinus_lr(lr_original,lr):
    return np.cos()

def timeSince(since, now):
    """ Returns formatted time.
    """
    #now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def explore_model(model):#, layershape):
    fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(5, 4))
    c=0
    for name, param in model.named_parameters():
        #print(name)
        l=param.detach().numpy()
        #print(l.shape)

        idx = 5
        if name[-8:] == 'weight_v':
            #print(l.shape)
            #print(l[0,0:3])

            ax[c,0].imshow(l[idx,0])
            ax[c,1].imshow(l[idx,1])
            ax[c,2].imshow(l[idx,2])

            ax[c,3].imshow(l[idx+1,0])
            ax[c,4].imshow(l[idx+1,1])
            ax[c,5].imshow(l[idx+1,2])

            c+=1
    for ax_r in ax:
        for ax_i in ax_r:
                ax_i.get_yaxis().set_visible(False)
                ax_i.get_xaxis().set_visible(False)
    fig.tight_layout()


    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    #Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0,1],[y,y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    fig.savefig('report/figures/viskernels.pdf', bbox_inches='tight')
    #plt.show()
    return None


def print_filters(model, layershape):
    """
    Print filters of the first layer.
    TO do:
     Change the function to also include the name of the chosen layer as
     the way to pick out filters.

     Create function to evaluate symmetries of the layers and score them on
     a scale from random to symmetric know filter.

     See how the filters evolve during training.
    """
    #for param in model.parameters():
    for name, param in model.named_parameters():
        print(name)
        l=param.detach().numpy()

        if name == 'conv1.weight': #l.shape == (64,3,5,5):

            for idx in range(0,5):
                plt.imshow(l[idx,0])
                plt.show()


    # Evaluating filters.
    if 0:
        for name, param in model.named_parameters():
            if name == 'conv1.weight': #l.shape == (64,3,5,5):
                l=param.detach().numpy()
                for filter in l:
                    print(filter[:,0,0] )
    return None

def mirror_filters(filter):
    """
    Input: 2d filters
    Return: 8 new filters which are the symmetric and antisymmetric parts, and
    then horizontal and vertical and opposit diagonal also.
    """
    hor = filter + np.flip(filter, axis=0)
    ver = filter + np.flip(filter, axis=1)
    sym = filter + np.transpose(filter)
    mys = filter + np.flip(np.transpose(np.flip(filter)))

    hor_anti = filter - np.flip(filter, axis=0)
    ver_anti = filter - np.flip(filter, axis=1)
    sym_anti = filter - np.transpose(filter)
    mys_anti = filter - np.flip(np.transpose(np.flip(filter)))

    return (hor/2, hor_anti/2,
            ver/2, ver_anti/2,
            sym/2, sym_anti/2,
            mys/2, mys_anti/2)

def measure_symanti(filter1, filter2):
    """
    Measure degree of symmetry of matrices. As proposed here:
    https://math.stackexchange.com/questions/2048817/metric-for-how-symmetric-a-matrix-is
    """
    norm1, norm2 = LA.norm(filter1), LA.norm(filter2)
    measure =  (norm1 - norm2) / (norm1 + norm2)
    return measure

def eval_layer(conv2dlayer):
    """
    Input is numpy array of dimensions:
    out_channels *
    in_channels  *
    filter_x     *
    filter_y
    """
    measure_list = list()
    ## Lopi version.
    for filter in conv2dlayer:
        # FOr first run only look at the the 0 layer.
        calc_filts_tuple = mirror_filters(filter[0])
        measure_list.append([measure_symanti(calc_filts_tuple[0], calc_filts_tuple[1]),
                             measure_symanti(calc_filts_tuple[2], calc_filts_tuple[3]),
                             measure_symanti(calc_filts_tuple[4], calc_filts_tuple[5]),
                             measure_symanti(calc_filts_tuple[6], calc_filts_tuple[7])])
    measure_list = np.array(measure_list)

    fig_qhist = plt.figure(figsize=(8*1.2,6*1.2))
    ax = [fig_qhist.add_subplot(2,2,1),
          fig_qhist.add_subplot(2,2,2),
          fig_qhist.add_subplot(2,2,3),
          fig_qhist.add_subplot(2,2,4)]
    for idx in range(0,4): ax[idx].hist(measure_list[:,idx],density=False)
    #hist_hor = ax[0].hist(measure_list[:,0])
    #plt.show()
    return fig_qhist, ax

def update_layer(conv2dlayer_t_work_on, next_layer):
    return None

def print_baseline(validloader, ind_sets, ksplit):
    bl=0
    for batch_idx,(data, target) in enumerate(validloader): 
        bl += target.sum().item()
    print("Baseline = {:.3f}".format( 1- bl/float(len(ind_sets[ksplit][1]))))
    return None

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float)
        for i in range(len(cm)):
            cm[i] = cm[i]/np.sum(cm[i])*100
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.yaxis.label.set_size(16)
    ax.xaxis.label.set_size(16)
    ax.xaxis.set_label_position('top')
    ax.set_ylim(cm.shape[1]-0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            T = format(cm[i, j], fmt)
            if normalize:
                T = T+ "%"
            ax.text(j, i, T,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig,ax

#def convert_pdf(file_path, pdf_name):
#    #file_path = 'experiments/'+experiment_name+'/'
#    images = convert_from_path(file_path+pdf_name, output_folder=file_path, fmt="png", output_file='rp')
#    return None





#Latex margin lenghts

#marignsize 11.5pc
#pc: pica is 12 pt and 4.218 mm and 0.16605 inches

#11.5 pc = 48.507 mm
#		= 1.909575 inches
#
#https://tex.stackexchange.com/questions/8260/what-are-the-various-units-ex-em-in-pt-bp-dd-pc-expressed-in-mm
#
#
# Phyton plt subfigure definiton:
# Playing with constants for Latex margin figures
pc = 400/2409 # the pc unit relative to inchens
goldenRatio = 1.618 # ratio between width and height
marginWidth = 11.5 # width of latex margin document
textWidth   = 28#36.35
resize = 10 # scale=0.1 in latex

def margin_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(marginWidth*pc*ratio[0], marginWidth*pc*ratio[1]/goldenRatio))
def text_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(textWidth*pc*ratio[0], textWidth*pc*ratio[1]/goldenRatio))
def full_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=((textWidth+marginWidth)*pc*ratio[0], (textWidth+marginWidth*ratio[1])*pc/goldenRatio))

#fig, ax = plt.subplots(figsize=(marginWidth*pc, marginWidth*pc/goldenRatio))

if __name__ == '__main__':
    print("Running utils as main, aka. debug mode")
    #lrtest = [0.01, 0.001, 0.0001]
    #for lrt in lrtest: print(cyclic_learningrate(0.01,lrt))
    #convert_pdf('tav_lrs3_newnorm','resultsplot.pdf')
#    convert_pdf('report/figures/','ex_le.pdf')
