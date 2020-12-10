### Script containing all filters to be used in the preproccessing step
## Need:
    #Ratio channel
    #Histogram equilization, CLAHE
    #Anscombe
    #Bilatteral
## Nice:
    #Fourier transform
    #PCA
import numpy as np
import math

def ratio(images, magic):
    """
    len(Data[0]) = length of dataset
    Data[:,0]    = channel one
    """
    if magic is None:
        magic = 0.001#np.mean(images)
    return np.concatenate((images, (images[:,0,:][:,np.newaxis,:]+magic)/(images[:,1][:,np.newaxis,:]+magic)), axis=1), magic

def hist_eq(images, params):
    """Histogram eq.
    max(f(x),0) with polynomial where highest order negative.
    """
    cutoff = None#0.2#None#0.0#[0,0,0]
    nbins=100
    deg = 10

    new_images = []
    fits = []
    max_intens = []
    for ch in range(0,images.shape[1]):
        if cutoff:
            images_cut = images[:,ch,:,:][images[:,ch,:,:] > cutoff].reshape(-1)
        else:
            images_cut = images[:,ch,:,:].reshape(-1)

        hist, bins = np.histogram(images_cut, bins = nbins)
        hist_cum = np.cumsum(hist)

        bins_c = bins[:-1]+ (bins[1:] - bins[:-1])/2
        fit6 = np.poly1d(np.polyfit(bins_c, hist_cum/hist_cum[-1], deg))

        new_images.append((fit6(images[:,[ch],:])*np.max(images[:,ch])))
        fits.append(fit6)

    new_images = np.concatenate(new_images, axis=1)


    if 0:
        print(bins_c[0:5])
        print(bins_c.shape, hist_cum.shape)
        print(fit6(0.3))
        plt.plot(bins_c, hist/hist_cum[-1])
        plt.plot(bins_c, hist_cum/hist_cum[-1])
        plt.plot(bins_c,fit6(bins_c))
        plt.show()

        print(new_images.shape)
    return new_images, fits

def gamma_corr(images, gamma):
    return images**(gamma**-1), gamma

def anscombe(images,param):
    """
    The generalized anscombe transformation.
    Param = gain
    """
    if param:
        for ch in range(0,images.shape[1]):
            new_images = param * images[:,[ch],:] + (param**2)**(3/8) + np.var(images[:,[ch],:]) - param * np.mean(images[:,[ch],:])
    else:
        param = 0.2
        for ch in range(0,images.shape[1]):
            new_images = param * images[:,[ch],:] + (param**2)**(3/8) + np.var(images[:,[ch],:]) - param * np.mean(images[:,[ch],:])
    return new_images, param

## Fourier or PCA?

## Normalization.
def normalization(images, list_norms):
    if list_norms is None:
        list_norms = []
        for ch in range(0, images.shape[1]):
            mean_ch, std_ch = np.mean(images[:,[ch],:]), np.std(images[:,[ch],:])
            list_norms.append(np.array([mean_ch,std_ch]))
        list_norms = np.array(list_norms)

        new_images = (images - list_norms[np.newaxis,:,0,np.newaxis, np.newaxis])/list_norms[np.newaxis,:,1,np.newaxis, np.newaxis]
    else:
        new_images = (images - list_norms[np.newaxis,:,0,np.newaxis, np.newaxis])/list_norms[np.newaxis,:,1,np.newaxis, np.newaxis]

    return new_images, list_norms



def clahe():
    """CLAHE"""
    return None

def bilatteral():
    return None

### Nice.
def FT(): return None
def PCA(): return None


### Slicing function
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

    for im in range(0,images.shape[0]):
        for x in range(0,x_end):#2*rows-1):
            extendx = x== x_end-1
            xs =int( ((x%2)*num_pix_x/2 + math.floor(x/2)*num_pix_x) * (1-extendx) + extendx * (num_rows-num_pix_x))
            for y in range(0,y_end):#2*cols-1):
                extendy = y==y_end-1
                ys =int( ((y%2)*num_pix_y/2 + math.floor(y/2)*num_pix_y) * (1-extendy) + extendy * (num_cols-num_pix_y))

                windows.append(np.concatenate([images[im, :,xs:xs+num_pix_x, ys:ys+num_pix_y].ravel(),
                               np.array([0 < np.sum(labels[im,xs:xs+num_pix_x, ys:ys+num_pix_y])]) ]))

    #for el in windows: print(el.shape)
    return np.stack(windows, axis=0)
