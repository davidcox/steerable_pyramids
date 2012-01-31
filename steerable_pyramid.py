from util import (max_pyramid_height, correlate_and_downsample,
                  upsample_and_correlate)

from sp3filters import SPFilterSet3
from numpy import array, empty, ones
from scipy.signal import resample
import scipy.ndimage as ndi

class SteerablePyramid:
    
    def __init__(self, filter_set=None):
        
        if filter_set is None:
            filter_set = SPFilterSet3()
        
        self.filter_set = filter_set        
    
    def process_image(self, im, pyramid_height=None, upsample=True):
        if pyramid_height is None:
            fs = self.filter_set.lo_filt.shape[0]
            pyramid_height = max_pyramid_height(im.shape, fs)
        
        
        hi0 = correlate_and_downsample(im, self.filter_set.hi0_filt)
        lo0 = correlate_and_downsample(im, self.filter_set.lo0_filt)
        
        self.residual_hipass = hi0
        pyramid = self.build_sp_levels(lo0, pyramid_height)
        
        if upsample:
            return self.upsample_pyramid(pyramid)
        else:
            return pyramid
            
        
    def build_sp_levels(self, im, height):
        """ Recursively build the levels of a steerable pyramid 
        """
        
        if height <= 0:
            return [[im]]
        
        bands = []  
        
        use_band_fb = True    
        if use_band_fb:
            bands_tmp = correlate_and_downsample(im, self.filter_set.band_fb)
            for i in range(0, bands_tmp.shape[2]):
                bands.append(bands_tmp[:,:,i])
        else:
            for filt in self.filter_set.band_filts:
                band = correlate_and_downsample(im, filt)
                bands.append(band)
        
        lo = correlate_and_downsample(im, self.filter_set.lo_filt, 2)
        
        print lo.shape
        pyramid_below = self.build_sp_levels(lo, height-1)
        
        return [bands] + pyramid_below
        
    def upsample_pyramid(self, pyramid):
        
        target_shape = self.residual_hipass.shape
        
        result = []
        for level in pyramid:
            new_level = []
            for band in level:
                band_shape = band.shape
                if len(target_shape) > len(band_shape):
                    band_shape = (band_shape[0], band_shape[1], 1)
                
                zf = array(target_shape) / array(band_shape)
                
                band.shape = band_shape
                
                tmp = ones(target_shape)
                if any(zf != 1):
                    ndi.zoom(band, zf, tmp, order=1)
                    upsamped = tmp
                else:
                    upsamped = band
                
                new_level.append(upsamped)
            result.append(new_level)
        
        return result
                

if __name__ == "__main__":

    from scipy.misc import lena
    import numpy as np
    import matplotlib.pylab as plt
    import time
    from scipy.ndimage import zoom
    
    im = lena().astype(np.float32)
    #im = zoom(im, 4)
    
    tic = time.time()
    sp = SteerablePyramid()
    
    upsamp = sp.process_image(im)
    print "run time: %f" % (time.time() - tic)
    
    for i in range(0, 4):
        im = upsamp[i][0].copy()
        im.shape = (im.shape[0], im.shape[1])
        plt.imshow(im, cmap='gray')
        plt.show()
    
    