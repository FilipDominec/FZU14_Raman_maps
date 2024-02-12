import numpy as np

def retouch_outliers(y, sigma_criterion=2, peaks=True, notches=False, iterations=13):
    kernel = [.5,0,.5] # averaging of neighbors #kernel = np.exp(-np.linspace(-2,2,5)**2) ## Gaussian
    for n in range(iterations):
        conv = np.convolve(y, kernel, mode='same')
        norm = np.convolve(np.ones_like(y), kernel, mode='same')
        smooth = conv/norm    # find the average value of neighbors
        rms_noise = np.average((y[1:]-y[:-1])**2)**.5   # estimate what the average noise is (rms derivative)
        if peaks and notches:
            outlier_mask = (np.abs(y-smooth) > rms_noise*sigma_criterion)    # find all points with difference from average less than 3sigma
        elif peaks:
            outlier_mask = ((y-smooth) > rms_noise*sigma_criterion)    # find all points with difference from average less than 3sigma
        elif notches:
            outlier_mask = ((y-smooth) < rms_noise*sigma_criterion)    # find all points with difference from average less than 3sigma
        #y[outlier_mask] = np.roll(y,1)[outlier_mask] # smooth[outlier_mask+1]
        y[outlier_mask] = smooth[outlier_mask]
    return y


def smooth(y, width=10):
    kernel = 2**(-np.linspace(-2, 2, width)**2) # truncated Gaussian
    conv = np.convolve(y, kernel, mode='same')
    norm = np.convolve(np.ones_like(y), kernel, mode='same')
    return conv/norm


def rm_bg(y, iter=35, coef=0.75, blurpx=50):
    """ subtracts smooth slowly varying background, keeping peaks and similar features,
    (almost) never resulting in negative values """
    def edge_safe_convolve(arr,ker): 
        return np.convolve(np.pad(arr,len(ker),mode='edge'),ker,mode='same')[len(ker):-len(ker)]

    y0 = y[:]
    ker = 2**-np.linspace(-2,2,blurpx)**2; 
    for i in range(iter):
      y = edge_safe_convolve(y,ker/np.sum(ker))
      y = y - ( np.abs(y-y0) + y - y0)*coef
    return y0-y


def norm(y):
    return y/np.mean(y)

