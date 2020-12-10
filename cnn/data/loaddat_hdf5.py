import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from statsmodels.distributions.empirical_distribution import ECDF
import time
import h5py


def load_hdf5(filename):
    data = {}
    c=0
    with h5py.File(filename+'.hdf5', 'r') as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        # Get the data
        for key in list(f.keys()):
            if key[-3:] == 'lab': c+=1
            data[key] = np.array(f[key])
    dname = 'image'
    npdata = np.array([data[dname+str(idx)] for idx in range(0,c)])
    labels = np.array([data[dname+str(idx)+'_lab'] for idx in range(0,c)])
    bboxes = np.array([data[dname+str(idx)+'_bbox'] for idx in range(0,c)])
    return npdata, labels, bboxes

def split_train_test(dname, npdata, seed,trP = 0.8, teP = 0.2):
    if (trP+teP)!=1:
        print('Train and test split doesnt sum to one.')

    r=np.random
    r.seed(seed)
    data_size = len(npdata)
    cut = math.floor(data_size*trP)
    if dname == 'MMII':
        # For MMII there seems to be a symmetry between the pictures with
        # and without FOs, thats why I assign them pairwise.
        indicies = np.arange(0,data_size/2)
        r.shuffle(indicies)
        indicies= np.concatenate((indicies[:,np.newaxis],indicies[:,np.newaxis]+10),axis=1)
        indicies = np.ravel(indicies)
    elif dname == 'CD':
        # Four kinds of meats.
        # 0-10, 11-19, 20-28, 29-39
        # Random is fine to begin with
        indicies = np.arange(0,data_size)
        r.shuffle(indicies)
    elif dname == 'san':
        # Comes in pairs of threes, working in an U shape from large thikness to low to large again.
        indicies = np.arange(0,data_size)
        r.shuffle(indicies)
    else:
        indicies = np.arange(0,data_size)
        r.shuffle(indicies)
    indicies = indicies.astype(int)


    # Use function sliding_window_split to get final versions of the data.
    #train = npdata[indicies[0:cut]]
    #train_labels = labels[indicies[0:cut]]
    #train_bbox = bboxes[indicies[0:cut]]

    #test  = npdata[indicies[cut: ]]
    #test_labels = labels[indicies[cut:]]
    return indicies[:cut], indicies[cut:]#train, train_labels, train_bbox, test, test_labels

def use_transformations(indata, trans, trans_params, vis=1):
    """If trans_params the dictionary containing transformation paramerters is
        empty we assume it is the training set we are working on."""
    work_data = indata[:]
    trans_params_work={}
    for fn in trans:     # iterate over list of functions
        if not trans_params:
            work_data, param = fn(work_data,None)
            trans_params_work[fn.__name__] = param
        else:
            work_data, param = fn(work_data, trans_params[fn.__name__])

        if vis:
            print('After transformation: '+fn.__name__)
            for ch in range(0,work_data.shape[1]):
                print('Channel: ' + str(ch))
                plt.imshow(work_data[1,ch])
                plt.colorbar()
                plt.show()
            print(work_data.shape)
    return work_data, trans_params_work

def sliding_window_split(images, labels, num_pix_x, num_pix_y):
    """
    Input data should be a list of dimensions:
    examples * im/dict * h/l * 385*552

    Returns a image with dimensions
    (#windows * h/l * (num_pix_x*num_pix_y)

    Some notes:
    the images has the shape: (384,352) -> (384,555).
    thus I have made the quick solution to only include the first 552 pixels.

    """
    num_rows = images.shape[-2]
    num_cols = images.shape[-1]

    rows_float = num_rows / num_pix_x
    cols_float = num_cols / num_pix_y

    rows_odd = math.ceil((2*rows_float)%2) - 1
    cols_odd = math.ceil((2*cols_float)%2) - 1

    rows = math.floor(rows_float)
    cols = math.floor(cols_float)

    x_end = 2*rows + rows_odd
    y_end = 2*cols + cols_odd

    windows = []
    labels_win = []
    win_pos=[]

    for im in range(0,images.shape[0]):
        for x in range(0,x_end):#2*rows-1):
            extendx = x== x_end-1
            xs =int( ((x%2)*num_pix_x/2 + math.floor(x/2)*num_pix_x) *
                     (1-extendx) + extendx * (num_rows-num_pix_x))
            for y in range(0,y_end):#2*cols-1):
                extendy = y==y_end-1
                ys =int( ((y%2)*num_pix_y/2 + math.floor(y/2)*num_pix_y) *
                         (1-extendy) + extendy * (num_cols-num_pix_y))
                windows.append(images[im, :,xs:xs+num_pix_x, ys:ys+num_pix_y])
                labels_win.append(0 < np.sum(labels[im,xs:xs+num_pix_x, ys:ys+num_pix_y]))
                win_pos.append((x,y))
                #windows.append(np.concatenate([images[im, :,xs:xs+num_pix_x, ys:ys+num_pix_y].ravel(),
                #               np.array([0 < np.sum(labels[im,xs:xs+num_pix_x, ys:ys+num_pix_y])]) ]))
    return np.stack(windows, axis=0), np.array(labels_win), win_pos


if __name__ == '__main__':
    import filters
    seed=42
    save=1
    save_full=1
    # Files to load
    files = ['xray_MMII',
             'xray_pTD',
    #         'xray_CD',
             'xray_CD_full']#,
    #         'xray_san']
    #files = ['xray_san']
    # Define global transformations.
    trans_gl =[]#[filters.ratio]#, filters.normalization]
    trans_params_gl = {}
    # Define local transformations.
    trans_lo =[]
    trans_params_lo = {}

    # Run loop:
    for idx in range(len(files)):

        npdata, labels, bboxes = load_hdf5(files[idx])
        # Split into training and testing.
        train_indicies, test_indicies = split_train_test(files[idx][5:], npdata, seed)
        # Train
        train_gl, trans_param_gl = use_transformations(npdata[train_indicies],
                                                       trans_gl,
                                                       trans_params_gl,vis=0)

        norm_gl, norm_params =  use_transformations(train_gl,
                                              [filters.normalization],
                                              {},vis=0)
        print(norm_params)
        train_win, train_labels_win, train_win_pos = sliding_window_split(train_gl, labels[train_indicies], 32,32)

        train_lo, trans_param_lo = use_transformations(train_win,
                                                       trans_lo,
                                                       trans_params_lo,vis=0)
        # Test
        test_gl, _ = use_transformations(npdata[test_indicies],
                                         trans_gl,
                                         trans_params_gl,vis=0)

        norm_gl_test, _ =  use_transformations(test_gl,
                                              [filters.normalization],
                                              {},vis=0)

        test_win, test_labels_win, test_win_pos = sliding_window_split(test_gl, labels[test_indicies], 32,32)
        test_lo, _ = use_transformations(test_win,
                                         trans_lo,
                                         trans_params_lo,vis=0)
        # Save
        fname_train_globalref = 'TRAIN_'+files[idx][:]+'_globalref_unnorm_'+str(train_gl.shape[1])+'ch.hdf5'
        fname_train = 'TRAIN_'+files[idx][:]+'_unnorm_'+str(train_gl.shape[1])+'ch.hdf5'
        fname_test  = 'TEST_' +files[idx][:]+'_unnorm_'+str(test_gl.shape[1])+'ch.hdf5'
        fname_test_globalref = 'TEST_'+files[idx][:]+'_globalref_unnorm_'+str(train_gl.shape[1])+'ch.hdf5'

        #fname_test_globalref = 'MIX_TEST_globalref_unnorm_'+str(test_gl.shape[1])+'ch.hdf5'
        fname_full_globalref = 'full_'+files[idx][:]+'_globalref_unnorm_'+str(test_gl.shape[1])+'ch.hdf5'
        list_trans_gl =[]
        list_trans_lo =[]
        for tr in trans_gl: list_trans_gl += [tr.__name__.encode("ascii", "ignore")]
        for tr in trans_lo: list_trans_lo += [tr.__name__.encode("ascii", "ignore")]

        if save:
            with h5py.File(fname_train_globalref, "w") as f1:
                dset1 = f1.create_dataset("images", data=norm_gl, dtype='f')
                #dset1.attrs['transformations_global'] = list_trans_gl
                for im in range(train_gl.shape[0]):
                    dset_bbox = f1.create_dataset("bboxes"+str(im), data=bboxes[train_indicies[im]], dtype='f')
            with h5py.File(fname_train, "w") as f1:
                dset1 = f1.create_dataset("images", data=train_lo, dtype='f')
                dset2 = f1.create_dataset("labels", data=train_labels_win, dtype='f')
                dset3 = f1.create_dataset("normalization", data = norm_params['normalization'], dtype = 'f')
                dset1.attrs['transformations_global'] = list_trans_gl
                dset1.attrs['transformations_local'] = list_trans_lo
                # Save transformations as metadata
            with h5py.File(fname_test, "w") as f1:
                dset1 = f1.create_dataset("images", data=test_lo, dtype='f')
                dset2 = f1.create_dataset("labels", data=test_labels_win, dtype='f')
                # Save transformations as metadata
                dset1.attrs['transformations_global'] = list_trans_gl
                dset1.attrs['transformations_local'] = list_trans_lo
        if save_full:
            # Full globalref
            with h5py.File(fname_full_globalref, "w") as f1:
                dset1 = f1.create_dataset("images", data=npdata, dtype='f')
                #dset1.attrs['transformations_global'] = list_trans_gl
                for im in range(npdata.shape[0]):
                    dset_bbox = f1.create_dataset("bboxes"+str(im), data=bboxes[im], dtype='f')
            # Train globalref
            with h5py.File(fname_train_globalref, "w") as f1:
                dset1 = f1.create_dataset("images", data=norm_gl, dtype='f')
                #dset1.attrs['transformations_global'] = list_trans_gl
                for im in range(train_gl.shape[0]):
                    dset_bbox = f1.create_dataset("bboxes"+str(im), data=bboxes[train_indicies[im]], dtype='f')
            # Test globalref
            with h5py.File(fname_test_globalref, "w") as f1:
                dset1 = f1.create_dataset("images", data=test_gl, dtype='f')
                dset2 = f1.create_dataset('bin_mask', data=labels[test_indicies], dtype='f')

                #dset1.attrs['transformations_global'] = list_trans_gl
                for im in range(test_gl.shape[0]):
                    dset_bbox = f1.create_dataset("bboxes"+str(im), data=bboxes[test_indicies[im]], dtype='f')
                #for dset_idx in range(len(all_global_test)):
                #    for im in range(len(all_global_test[dset_idx])):
                #        dset1 = f1.create_dataset("dset"+str(dset_idx)+"image"+str(im), data=all_global_test[dset_idx][im], dtype='f')
                #        dset_bbox = f1.create_dataset("dset"+str(dset_idx)+"bboxes"+str(im), data=all_boxes_test[dset_idx][im], dtype='f')
