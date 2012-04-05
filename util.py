import numpy as np
import matplotlib.pyplot as plt
from numpy import array, any, floor, concatenate, zeros
try:
    from pythor3.operation import fbcorr
except ImportError:
    from scipy.ndimage import correlate
    def fbcorr(im, f, mode, stride, plugin=None, plugin_kwargs=None):
        assert mode=='same'
        if f.ndim == 3 and f.shape[0] == 1:
            f = f[0, :, :]
            output = correlate(im, f, mode='constant')[::stride, ::stride]
            return output
        elif f.ndim == 4:
            if f.shape[3] == 1:
                f = f[:, :, :, 0]
                output = [correlate(im, fi, mode='nearest')
                        [::stride, ::stride]
                    for fi in f]
                return np.asarray(output).transpose(1, 2, 0)
            else:
                raise NotImplementedError()


def imshow(img, title=None):
    if img.ndim == 2:
        plt.imshow(img)
    elif img.ndim == 3:
        R = np.ceil(np.sqrt(img.shape[2]))
        C = np.ceil(img.shape[2] / R)
        R, C = map(int, (R, C))
        for i in range(img.shape[2]):
            plt.subplot(R, C, i + 1); plt.imshow(img[:, :, i])
    if title:
        plt.title(title)
    plt.show()

def max_pyramid_height(im_shape, filter_shape):

    im_shape = array(im_shape)
    filter_shape = array(filter_shape)

    if any(im_shape == 1):  # 1D image
        im_shape = (prod(array(im_shape)),)
        filter_shape = (prod(filter_shape),)
    elif any(filter_shape == 1):
        filter_shape = (filter_shape[0], filter_shape[0])

    if any(im_shape < filter_shape):
        height = 0
    else:
        height = 1 + max_pyramid_height(floor(im_shape/2.0),
                                        filter_shape)

    return height


def correlate_and_downsample(im, filt, stride=1):
    #print 'c&d shapes', im.shape, filt.shape
    f = filt.copy()

    if f.ndim is not 4 and f.ndim - im.ndim != 1:
        if im.ndim is 2:
            f.shape = (1, filt.shape[0], filt.shape[1])
        elif im.ndim is 3:
            f.shape = (1, filt.shape[0], filt.shape[1], 1)

    use_cthor = True
    if use_cthor:
        p='cthor'
        pkw = {'variant':'simple:debug'}
    else:
        p = 'scipy_naive'
        pkw = {}

    result = fbcorr(im, f, mode='same', stride=stride,
                    plugin=p, plugin_kwargs=pkw)

    return result

def upsample_and_correlate(im, filt, stride=(2,2), apron=(0,0)):
    #print 'u&c shapes', im.shape, filt.shape

    if im.ndim is 2:
        tmp = zeros( array(im.shape) * array(stride) + (array(apron)*2+1))
        tmp[apron[0]:-apron[0]:stride[0], apron[0]:-apron[0]:stride[0]] = im
    elif im.ndim is 3:
        target_shape = concatenate(((array(im.shape[0:2]) * array(stride) 
                                     + array(apron)), [im.shape[2]]))
        tmp = zeros(target_shape)
        tmp[apron[0]-1:-apron[0]+2:stride[0], 
            apron[1]-1:-apron[1]+2:stride[1],:] = im

    f = filt.copy()
    f.shape = (1, f.shape[0], f.shape[1], 1)

    return fbcorr(tmp, f, mode='same')
