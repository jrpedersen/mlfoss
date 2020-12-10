# Load packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import numpy as np
import scipy.ndimage as ndi

import math
import numbers
import time
import copy

import h5py
import cv2
import xml.etree.ElementTree as ET
from scipy.special import binom
import torch
#import torchvision
#import torchvision.transforms as transforms
#import bezier_func
# FUnctions from https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):#200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    #print(curve)
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    """ Count clockwise sort, each row is y,x. The y,x order doesnt matter later"""
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a) # Counter clockwise sort
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0) # Insert first point as endpoint
    d = np.diff(a, axis=0) # Get differences.
    ang = np.arctan2(d[:,1],d[:,0]) # Get angle corresponding to differences here x,y as input
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi) # Convert all angles to positives
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi # Make ang mean of angles in and out of point
    ang = np.append(ang, [ang[0]]) # Set end angle again
    a = np.append(a, np.atleast_2d(ang).T, axis=1) # Make 3. column in points containing the angles.
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

def create_shape_filter(seed):
    """
    Create a random shape with a random size
    Check length and width are used right
    """
    # Object radius function
    #object = lambda x: 4
    object = lambda x: 4*np.sin(x)**2 + 8
    rad = 0.2
    edgy = 0.05

    obj_scale = 40
    a = get_random_points(n=7, scale=obj_scale) + [0,0]
    #x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    c = get_bezier_curve(a,rad=rad, edgy=edgy)
    c -= obj_scale/2
    #object = np.zeros((obj_scale,obj_scale))
    #for points in c:
    #    object += 0
    fig, ax = plt.subplots()

    #mpatches.Polygon(points.T, closed=True, **kwargs)
    #ax.add_patch(poly)
    plt.plot(c)
    plt.show()

    length, width = 40,40
    center = (int(width/2), int(length/2))

    sf = np.zeros((width,length))
    """radius_matrix = np.zeros((width,length))

    for row in range(0,width):
        for col in range(0,length):
            r = np.sqrt((row - center[0] + 0.5)**2 + (col - center[1] + 0.5)**2)
            #angle = (np.arccos(row/r), np.arcsin(col/r))
            angle = np.angle((row - center[0]) + 1j * (col - center[1]))

            #real

            sf[row,col] = object(angle) > r
            #radius_matrix[row,col] = (row - center[0])**2 + (col - center[1])**2
"""




    #print(radius_matrix)
    #plt.imshow(sf)
    #plt.show()

    return sf

def parse_annotation(annotation_path):
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

def create_shape(size, rad=0.2, edgy=0.05, num_points=200):
    """Create random shape dependent only on rand."""
    ob_size = (size[0]*10+size[1]*10)/2
    a = get_random_points(n=num_points, scale=ob_size)
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    x -= size[0]*10/2
    y -= size[1]*10/2
    c = x+ 1j *y
    angles = np.angle(c)
    radi = (x**2 + y**2)
    canvas = np.zeros((size[0]*10,size[1]*10))

    for row in range(size[0]*10):
        for col in range(size[1]*10):
            radi_cent = ((row-size[0]*10/2)**2 + (col-size[1]*10/2)**2)
            ang_cent  = np.angle((row-size[0]*10/2) + 1j* (col-size[1]*10/2))
            min_angles = np.argmin(np.abs(angles - ang_cent))
            canvas[row,col] = radi_cent < radi[min_angles]
    canvas_c = canvas.copy()
    canvas_reduced = canvas_c.reshape(int(size[0]),10,int(size[1]),10).sum(axis=(1,3)) /100.0
    return canvas_reduced

def image_with_boxes(pic, list_of_boxes,title):
    # Create figure and axes
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # Display the image
    ax.imshow(pic)
    for box in list_of_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0],box[1]),(box[2]-box[0]),(box[3]-box[1]),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    #plt.title(title)
    fig.savefig('ex_boxes_scale.pdf', bbox_inches='tight')
    plt.show()

def remove_background(images, filtersize):
    """
    Applies a median filter and substracts it from original to return image w/out
    background.

    Try: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.median_filter.html
    instead.
    """
    fg = np.zeros(images.shape)
    for im_idx in range(images.shape[0]):
        for e_idx in range(images.shape[1]):
            background = ndi.filters.median_filter(images[im_idx,e_idx],filtersize)
            #im8int = (255*(images[im_idx,e_idx]-np.min(images[im_idx,e_idx]))/(np.max(images[im_idx,e_idx])- np.min(images[im_idx,e_idx]))).astype(np.uint8)
            #background = cv2.medianBlur(im8int,filtersize)
            foreground = images[im_idx,e_idx] - background
            fg[im_idx,e_idx] = foreground#-np.min(foreground)

        #ex = cv2.blur(im.astype(np.float32),(25,25))
        #fg.append(foreground-np.min(foreground))
    return np.array(fg)

def gather_fos(dataset, mix=0,verbose=0):
    filename = dataset
    data = {}
    c=0
    with h5py.File(filename, 'r') as f:
        # List all groups
        if verbose: print("Keys: %s" % f.keys())
        # Get the data
        keys = list(f.keys())
        for key in list(f.keys()):
            data[key] = np.array(f[key])
    keys = [key for key in keys if key[5]=='i']
    if mix:
        fo = []
        num_dsets = np.max(np.array([int(key[4]) for key in keys])) + 1
        for num in range(num_dsets):
            num_images = np.max(np.array([int(key[10:]) for key in keys if int(key[4]) == num])) + 1
            images = []
            for idx in range(num_images):
                images.append( data['dset'+str(num)+'image'+str(idx)])
            images = np.stack(images,axis=0)
            images = remove_background(images,(25,25))
            for idx in range(num_images):
                for idx2 in range(len(data['dset'+str(num)+'bboxes'+str(idx)])):
                    bndbox=np.array(data['dset'+str(num)+'bboxes'+str(idx)][idx2]).astype(int)
                    fo.append(images[idx,:,bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]])
    else:
        images = data['images']
        images = remove_background(images,(25,25))

        fo = []

        for idx in range(images.shape[0]):
            for idx2 in range(len(data['bboxes'+str(idx)])):
                bndbox=np.array(data['bboxes'+str(idx)][idx2]).astype(int)
                fo.append(images[idx,:,bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]])

    if verbose:
        if mix:
            print('Verbose is broken for mix')
            return None

        bboxes = [data['Bbox_ni'+str(idx)] for idx in range(0,c)]
        print('Total number of images loaded: ',c)
        print('Total number of FOs: ', len(fo))

        fig = plt.figure(figsize=(10, 8))#,constrained_layout=True)
        gs = fig.add_gridspec(6, 3)
        axr = []
        ex_num=0
        for _ in range(6):
            ex = np.random.randint(len(fo))
            axr.append(  [fig.add_subplot(gs[ex_num,0]),fig.add_subplot(gs[ex_num,1]),fig.add_subplot(gs[ex_num,2])])
            axr[-1][0].imshow(fo[ex][0])
            axr[-1][1].imshow(fo[ex][1])
            axr[-1][2].imshow(fo[ex][2])
            ex_num+=1

        ax_tot = axr
        for _axes in ax_tot:
            for ax in _axes:
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
        #plt.imshow(fo[69][2])
        #plt.show()


        #print('Bbox example: \n', bboxes[1])
        ex=6
        #image_with_boxes(images[ex,2],bboxes[ex],'Test')

    return fo

def filter_fo(old_fo, low_limit,upp_limit):
    fo = old_fo[:]
    del_list=[]
    c=0
    for fo_id in range(len(fo)):
        if np.min(fo[fo_id]) < low_limit:
            del_list.append(c)
        elif np.min(fo[fo_id]) > upp_limit:
            del_list.append(c)
        c+=1

    for to_be_del in range(len(del_list)):
        fo.pop(del_list[-1-to_be_del])
    return fo

def create_top_image(size_im, shape, fo, verbose = 0):
    """Function to create new image. Should return a 32x32 numpy array to be added on top of any image as a transformation with a random probability."""
    top_image = np.zeros((len(fo),size_im[0],size_im[1]))
    top_four = np.argpartition(fo[0].ravel(), -4)[-4:]
    values = [np.mean(fo[ch].ravel()[top_four]) for ch in range(len(fo))]
    if verbose:
        print(top_four)
        print(fo[0].ravel()[top_four])
        print(values)
    #old
    #top_pix = np.argmax(fo[0])
    #shp_le  = shape * values[0]#fo[0].ravel()[top_pix]
    #shp_he  = shape * values[1]#fo[1].ravel()[top_pix]
    #shp_ldh = shape * values[2]#fo[2].ravel()[top_pix]
    #afo = np.stack((shp_le,shp_he,shp_ldh))
    afo = []
    for ch in range(len(fo)): afo.append(shape * values[ch])
    afo = np.stack(afo, axis=0)

    rpos = np.random.randint(size_im[0]-afo.shape[1], size=2) # Assume image is quadratic.
    top_image[:,rpos[0]:rpos[0]+afo.shape[1], rpos[1]:rpos[1]+afo.shape[2]] += afo

    return top_image#np.moveaxis(top_image, 0 ,-1)#.reshape((32,32,2), order='F')

def new_topim(fo, num_shapes=10, num_afo=60):
    # Shapes
    shapes = []
    for _ in range(num_afo):
        shapes.append(create_shape((8,8), 0.2, 0.05, 7))
    # Final images
    top_images = []
    for _ in range(num_afo):
        top_images.append(create_top_image((32,32), shapes[np.random.choice(len(shapes))],fo[np.random.choice(len(fo))]))
    return top_images

def filter_fo(old_fo, low_limit,upp_limit):
    fo = old_fo[:]
    del_list=[]
    c=0
    for fo_id in range(len(fo)):
        if np.min(fo[fo_id]) < low_limit:
            del_list.append(c)
        elif np.min(fo[fo_id]) > upp_limit:
            del_list.append(c)
        c+=1

    for to_be_del in range(len(del_list)):
        fo.pop(del_list[-1-to_be_del])
    return fo

class AddAFO(object):
    """Add artificial foreign object to image.

    Args:
        size (sequence or int):
    """

    def __init__(self, size, top_images):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        #self.do_it = np.random.uniform < prob
        self.top_images = top_images
        self.choises = len(top_images)

    #def __name__(self): return 'AddAFO'
    def __call__(self, img):
        """
        Args:
            .

        Returns:
            .
        """
        img = img + torch.as_tensor(self.top_images[np.random.choice(self.choises)],dtype=torch.float)
        #img = img[:,:,:2] + (create_top_image(self.size, shapes[r.choice(len(self.shapes))],fo[r.choice(len(self.fo))]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandFlipRot(object):
    """Random flip and rotate image.
    """
    def __init__(self, size, p=0.5):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.p = p

    def __call__(self, img):
        if self.p < np.random.random():
            img = img.permute(0,2,1)
        if self.p < np.random.random():
            img = torch.flip(img,[1])
        if self.p < np.random.random():
            img = torch.flip(img,[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class AddAFO2(object):
    """Add artificial foreign object to image.

    Args:
        size (sequence or int):
    """

    def __init__(self, size, shapes, fo):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        #self.do_it = np.random.uniform < prob
        self.shapes = shapes
        self.fo = fo
    #def __name__(self): return 'AddAFO'
    def __call__(self, img):
        """
        Args:
            .

        Returns:
            .
        """
        img = img + torch.as_tensor(create_top_image(self.size, self.shapes[np.random.choice(len(self.shapes))],self.fo[np.random.choice(len(self.fo))]),dtype=torch.float)
        #img = img[:,:,:2] + (create_top_image(self.size, shapes[r.choice(len(self.shapes))],fo[r.choice(len(self.fo))]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class dataset_im_list(torch.utils.data.Dataset):
    """
    I have a torch dataset class for loading of the pictures.
    """
    def __init__(self, in_file, nx,ny,transform=None):
        super(dataset_im_list, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.n_images, self.num_cols = self.file['Images'].shape

        self.nx, self.ny = nx,ny #picdim[0], picdim[1]
        #self.pictot = self.nx * self.ny
        self.transform=transform

    def __getitem__(self, index):
        inpic = self.file['Images'][index]
        label = torch.tensor(inpic[-1]).type(torch.LongTensor)
        pic =  inpic[:-1].reshape(self.nx, self.ny,3, order = 'F')

        if self.transform:
            for t in self.transform:
                #pic = t(pic)
                if format(t)[:6] != 'AddAFO' :
                    pic = t(pic)
                else:
                    if 0.5 > np.random.random():
                        pic = t(pic)
                        label = torch.tensor(1).type(torch.LongTensor)


        return pic,label#pic.astype('float32'), label

    def __len__(self):
        return self.n_images

    def num_fo(self):
        c=0
        for i in range(0,self.n_images):
            c+=int(self.file['Images'][i,-1])

        return c
