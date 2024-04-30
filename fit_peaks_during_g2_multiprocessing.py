import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
from tqdm import tqdm
import h5py
import pandas as pd
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
from lmfit import Model
from lmfit.lineshapes import lorentzian



def worker(img):
    gimg = gaussian_filter(img, 35)
    x, y = np.indices(gimg.shape)
    mod = Model(lorentzian2d, independent_vars=['x', 'y'])
    params = mod.make_params()
    params['amplitude'].set(value=1e7, min=1e6, max=1e9)
    params['rotation'].set(value=0.75, min=0.5, max=1)
    params['sigmax'].set(value=165, min=150, max=300)
    params['sigmay'].set(value=150, min=100, max=300)
    params['centery'].set(value=365, min=300, max=400)
    params['centerx'].set(value=425, min=400, max=500)
    params['offset'].set(value=800, min=500, max=1000)

    error = 1 / np.sqrt(gimg).ravel()
    result = mod.fit(img.ravel(), x=x.ravel(), y=y.ravel(), params=params, weights=1/error)

    iter_dict = result.params.valuesdict()
    iter_dict['rotation_error'] = result.params['rotation'].stderr
    iter_dict['amplitude_error'] = result.params['amplitude'].stderr
    iter_dict['sigmax_error'] = result.params['sigmax'].stderr
    iter_dict['sigmay_error'] = result.params['sigmay'].stderr
    iter_dict['centerx_error'] = result.params['centerx'].stderr
    iter_dict['centery_error'] = result.params['centery'].stderr
    iter_dict['offset_error'] = result.params['offset'].stderr

    return iter_dict


def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 rotation=0, offset = 0):
    # https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay) + offset
    
if __name__ == '__main__':
    temp = 230
    file = rf'./data\4-4-2024\{temp:.0f}k_xpcs_1.h5'
    roi = np.s_[150:1000, 1100:1900]

    print('started loading data')
    with h5py.File(file, 'r') as f:
        data = f['entry']['data']['data'][(..., *roi)]

    print('data loaded')
    with mp.Pool(processes=mp.cpu_count()) as pool:
        out = list(tqdm(pool.imap(worker, (im for im in data),
                            chunksize=1), total=len(data) - 1, desc='fitting lorentzian peak'))
        
    
    df = pd.DataFrame(out)
    df.to_csv(f'./data/4-4-2024/{temp}K_2hour_wait_lorentzian_fits_df.csv')