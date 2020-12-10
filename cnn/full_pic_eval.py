import numpy as np
import torch
import math
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Custom
import data.mat_to_hdf5 as ldm
import data.loaddat_hdf5 as ldd
import data.filters as filters

def slides_to_pic(slides, labels,original_pic_dim):#)num_pix_x, num_pix_y):
    """
    Reconstruct the full picture from the individual slides.
    """
    # Setting up the basic dimonsions of the pictures.
    num_pix_x = slides.shape[2]
    num_pix_y = slides.shape[3]
    num_rows = original_pic_dim[0]
    num_cols = original_pic_dim[1]
    rows_float = num_rows / num_pix_x
    cols_float = num_cols / num_pix_y
    rows_odd = math.ceil((2*rows_float)%2) - 1
    cols_odd = math.ceil((2*cols_float)%2) - 1
    rows = math.floor(rows_float)
    cols = math.floor(cols_float)
    x_end = 2*rows + rows_odd
    y_end = 2*cols + cols_odd
    # Create empty list of boxes to draw
    boxes_tb_drawn =[]
    # Loading the pictures and in this case only adding the lowenergy part.
    #totalimsize = int((len(slides[0]) - 1)/2)
    #slides_re = slides[:,:totalimsize].reshape(-1,num_pix_x,num_pix_y)
    # Create the resulting initial picture.
    pics = []
    for channel in range(slides.shape[1]):
        pic = np.zeros((num_rows,num_cols))
        c=0
        for x in range(0,x_end):#2*rows-1):
            extendx = x== x_end-1
            xs =int( ((x%2)*num_pix_x/2 + math.floor(x/2)*num_pix_x) * (1-extendx) + extendx * (num_rows-num_pix_x))
            for y in range(0,y_end):#2*cols-1):
                extendy = y==y_end-1
                ys =int( ((y%2)*num_pix_y/2 + math.floor(y/2)*num_pix_y) * (1-extendy) + extendy * (num_cols-num_pix_y))
                pic[xs:xs+num_pix_x, ys:ys+num_pix_y] = slides[c,channel]
                if labels[c] ==1 and channel==0:
                    boxes_tb_drawn.append([xs,ys])
                c+=1
        pics.append(pic)
    return np.array(pics), boxes_tb_drawn

def image_with_boxes(pic, list_of_boxes, nx, ny,title):
    # Create figure and axes
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # Display the image
    ax.imshow(pic)

    rect = patches.Rectangle((0,0),nx,ny,linewidth=1,edgecolor='r',facecolor='r')
    # Add the patch to the Axes
    ax.add_patch(rect)

    for box in list_of_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[1],box[0]),nx,ny,linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    #plt.title(title)
    fig.savefig('ex_boxes_scale.pdf', bbox_inches='tight')
    plt.show()


seed=42
if __name__ == '__main__':
    files = ['C:/Users/ext1150/foss_data/erda_data/',
             'C:/Users/ext1150/20191106_PreTestData/',
             'C:/Users/ext1150/foss_data/20190705_CustomerData/',
             'C:/Users/ext1150/foss_data/sanity/']
    #files = ['../../eScience/Projects/FOSS-ML/FirstData/','../../eScience/Projects/FOSS-ML/Squares/']
    load_funcs = [ldm.load_xray_MMII,
                  ldm.load_xray_pTD,
                  ldm.load_xray_CD,
                  ldm.load_xray_san]

    voc_labels = ('fod', 'fod2','meat', 'skinne') # The labels I used when labelling data,
    #label_map = {k: v + 1 for v, k in enumerate(voc_labels)} #needs to fit to create_objma
    label_map = {k: v  for v, k in enumerate(voc_labels)} #needs to fit to create_objma

    # load
    data_id = 2
    images = load_funcs[data_id](files[data_id])
    print(images.shape)
    objmas, bboxes = ldm.create_objma(files[data_id], images, label_map)
    train_indicies, test_indicies = ldd.split_train_test(files[data_id][5:], images, seed)

    trans_gl =[filters.ratio]#, filters.normalization]
    trans_params_gl = {}
    train_gl, trans_param_gl = ldd.use_transformations(images[train_indicies],
                                                   trans_gl,
                                                   trans_params_gl,vis=0)
    w_size = (32,32)
    win, labels_win, win_pos = ldd.sliding_window_split(train_gl, objmas[train_indicies], w_size[0],w_size[1])
    # Plot an image with red boxes.
    image_idx = 3
    windows_number = int(win.shape[0]/len(train_indicies))
    print(windows_number)
    pic, boxes = slides_to_pic(win[image_idx*windows_number : (image_idx+1)*windows_number],
                               labels_win[image_idx*windows_number : (image_idx+1)*windows_number],
                               images.shape[-2:])
    print(pic.shape)
    image_with_boxes(pic[0], boxes, w_size[0], w_size[1],'picture')
