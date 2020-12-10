# Loads images and saves to hdf5 files, one for training, and one for testing.
# Load packages
import numpy as np
import scipy.io
import h5py
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import PIL.Image
import os.path
#test

def parse_annotation(annotation_path,label_map):
    """Converts the bounding boxes from xml to python dict."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes=list()
    labels=list()

    for object in root.iter('object'):
        label = object.find('name').text.lower().strip()
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
    return {'boxes': boxes, 'labels': labels}#, 'difficulties': difficulties}

def transform_boxes_boolarray(shape,dict_labels):
    """Transform dicts of bounding boxes to boolean matrix."""
    #Note: Array matrix, dimensions are hardcoded to be the dimensions of the pictures.
    barray = np.zeros(shape)
    ones   = np.ones(shape)
    if len(dict_labels['labels'])>1: # Only go if more than background in the dictionary
        for idx in range(0,len(dict_labels['boxes'])): # Loop over all the obejcts
            if dict_labels['labels'][idx] < 2: # But not the background
                bndbox=dict_labels['boxes'][idx]
                barray[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] += ones[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
    return barray

def load_xray_MMII(file,out='post'):
    """ Load x-ray data."""
    matload = scipy.io.loadmat(file+'MMII_sample_data.mat') # Read from .mat file.
    matFOD  = scipy.io.loadmat(file+'MMII_sample_data_with_FO.mat')
    data_noFOD = matload['Data'] # Get data from dictionary.
    data_wiFOD = matFOD['Data']

    def add_landh_energy(data,list1,list3):
        #In this case there is 4 different pictures.
        for _im in range(0,len(data)):
            # Here the first part is the low energy picture, the second the high energy picture.
            list1.append(np.array([data[_im][0][0][0][0][0][0][0],
                                   data[_im][0][0][0][0][0][0][1]]) )
            list3.append(np.array([data[_im][0][0][0][0][0][0][2],
                                   data[_im][0][0][0][0][0][0][3]]) )
        return None
    raw =[]
    post=[]
    add_landh_energy(data_noFOD, raw,post)
    add_landh_energy(data_wiFOD, raw,post)

    # Step to make all images have same width.
    list_rows = [e.shape[2] for e in post]
    min_row = int(np.min(np.array(list_rows)))
    post2 = [post[idx][:,:, :min_row] for idx in range(0,len(post))]
    post2 = np.array(post2)

    if out=='post': return post2
    if out=='raw': return raw

def load_xray_pTD(file):
    """Load the data named: preTestData"""
    matload = scipy.io.loadmat(file+'preTestData') # Read from .mat file.
    data= matload['dataPreTest'] # Get data from dictionary.
    list1 =[]
    for _im in range(0,len(data)): list1.append(np.array([data[_im][0][0][0][0][0][0][0], data[_im][0][0][0][0][0][0][1]]) )

    list_rows = [e.shape[2] for e in list1]
    min_row = int(np.min(np.array(list_rows)))
    list2 = [list1[idx][:,:, int((list_rows[idx]-min_row)/2):int((list_rows[idx]-min_row)/2) + min_row] for idx in range(0,len(data))]
    list2 = np.array(list2)
    return list2

def load_xray_CD(file):
    """Load the data named: 20190705_CustomerData/FODdata."""
    matload = scipy.io.loadmat(file+'FODdata.mat')
    data= matload['data']
    c=0
    list1 =[]
    for _im in range(0,len(data)): list1.append(np.array([data[_im][0][0][0][0][0][0][0], data[_im][0][0][0][0][0][0][1]]) )
    list_rows = [e.shape[2] for e in list1]
    min_row = int(np.min(np.array(list_rows)))

    list2 = [list1[idx][:,:, int((list_rows[idx]-min_row)/2):int((list_rows[idx]-min_row)/2) + min_row] for idx in range(0,len(data)) ]
    list2 = np.array(list2)
    return list2

def load_xray_CD_full(file):
    """Load the data named: 20190705_CustomerData/FODdata."""
    matload = scipy.io.loadmat(file+'FODdata.mat')
    data= matload['data']
    c=0
    list1 =[]
    for _im in range(0,len(data)): list1.append(np.array([data[_im][0][0][0][0][0][0][0], data[_im][0][0][0][0][0][0][1]]) )
    list_rows = [e.shape[2] for e in list1]
    min_row = int(np.min(np.array(list_rows)))

    list2 = [list1[idx][:,:, int((list_rows[idx]-min_row)/2):int((list_rows[idx]-min_row)/2) + min_row] for idx in range(0,len(data)) ]
    list2 = np.array(list2)
    return list2

def CD_full_labels(file_dest, images, label_map, verbose=0):
    """ Load labels made through labelme."""
    # FInd same minimum as in the image loader.
    list_rows = [e.shape[2] for e in images]
    min_row = int(np.min(np.array(list_rows)))

    # Loop over all images:
    objmas = []
    bboxes = []
    for im_idx in range(len(images)):
        label_png = file_dest + 'labels/im'+str(im_idx)+'_json/label.png'
        if os.path.exists(label_png):
            lbl = np.asarray(PIL.Image.open(label_png))
            objmas.append( lbl[:, int((list_rows[im_idx]-min_row)/2):int((list_rows[im_idx]-min_row)/2) + min_row] == 1) # Modfications
            bboxes.append([])
            if verbose:
                #plt.imshow(lbl)
                #plt.show()
                #plt.imshow(images[im_idx][0])
                #plt.show()
                plt.imshow(objmas[-1])
                plt.show()
        else:
            objmas.append(np.zeros(images[im_idx,0].shape))
            bboxes.append([])
    return np.array(objmas), bboxes

def load_xray_san(file):
    """Load the data named: sanity/sanity."""
    matload = scipy.io.loadmat(file+'sanity.mat')
    data= matload['data']
    c=0
    list1 =[]
    for _im in range(0,len(data[0])): list1.append(np.array([data[0][_im][0][0][0], data[0][_im][0][0][1]]) )
    list_rows = [e.shape[2] for e in list1]
    min_row = int(np.min(np.array(list_rows)))
    list2 = [list1[idx][:,:, int((list_rows[idx]-min_row)/2):int((list_rows[idx]-min_row)/2) + min_row] for idx in range(0,len(list1)) ]
    list2 = np.array(list2)
    return list2

def create_objma(file, images,label_map):
    """Create object matrix."""
    objma = list() # example * im/dict * h/l * 385*552
    list_bbox = []
    for idx in range(0,len(images)):
        list_bbox.append(parse_annotation(file+'labels/im'+str(idx)+'.xml', label_map))
        objma.append(transform_boxes_boolarray(images[idx][0].shape, list_bbox[-1]))
    objma = np.array(objma)

    bbox_nparray = []
    for im_dict in list_bbox:
        bbox_nparray.append([])
        for ob_id in range(len(im_dict['labels'])):
            if im_dict['labels'][ob_id] == 1:
                bbox_nparray[-1].append(im_dict['boxes'][ob_id])
    return objma, bbox_nparray



if __name__ == '__main__':
    save=1
    files = ['C:/Users/ext1150/foss_data/erda_data/',
             'C:/Users/ext1150/20191106_PreTestData/',
             'C:/Users/ext1150/foss_data/20190705_CustomerData/',
             'C:/Users/ext1150/foss_data/20190705_CustomerData/',
             'C:/Users/ext1150/foss_data/sanity/']
    #files = ['../../eScience/Projects/FOSS-ML/FirstData/','../../eScience/Projects/FOSS-ML/Squares/']
    load_funcs = [load_xray_MMII,
                  load_xray_pTD,
                  load_xray_CD,
                  load_xray_CD_full,
                  load_xray_san]

    voc_labels = ('fod','meat', 'fod2', 'skinne') # The labels I used when labelling data,
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)} #needs to fit to create_objma

    for data_id in range(len(files)):
        images = load_funcs[data_id](files[data_id])
        if files[data_id] == files[data_id-1]:
            objmas, bboxes = CD_full_labels(files[data_id],images,label_map)
        else:
            objmas, bboxes = create_objma(files[data_id], images, label_map)


        if save:
            with h5py.File(load_funcs[data_id].__name__[5:]+'.hdf5', "w") as f1:
                for im in range(0,len(images)):
                    dset1 = f1.create_dataset('image'+str(im), data=images[im], dtype='f')
                    dset2 = f1.create_dataset('image'+str(im)+'_lab', data=objmas[im], dtype='f')
                    dset3 = f1.create_dataset('image'+str(im)+'_bbox', data=np.array(bboxes[im]), dtype='f')
