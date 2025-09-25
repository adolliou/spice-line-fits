import copy, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .skew_parameter_search import detrend_dopp
import matplotlib as mpl
from astropy.visualization import ImageNormalize, AsymmetricPercentileInterval, SqrtStretch, LinearStretch, LogStretch


def get_range(data, stre="linear", imax=99.5, imin=2):
    """
    :param data:
    :param stretch: 'sqrt', 'log', or 'linear' (default)
    :return: norm
    """
    if np.isnan(data).sum() == data.size:
        return None

    isnan = np.isnan(data)
    data = data[~isnan]
    do = False
    if imax > 100:
        vmin, vmax = AsymmetricPercentileInterval(imin, 100).get_limits(data)
        vmax = vmax * imax/100
    else:
        vmin, vmax = AsymmetricPercentileInterval(imin, imax).get_limits(data)

    #    print('Vmin:', vmin, 'Vmax', vmax)
    if stre == "linear":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    elif stre == 'sqrt':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    elif stre == 'log':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    else:
        raise ValueError('Bad stre value: either linear, log or or sqrt')
    return norm




def color_dopp_img(dopp,vmin,vmax,mask,gfac=1.0/2.2):
    dscal = mask*np.clip(2*(dopp-(vmin+vmax)*0.5)/(vmax-vmin),-1,1)
    return (1.0-np.abs(np.array([np.clip(dscal,-1,0),np.clip(dscal,-1,1),np.clip(dscal,0,1)])).transpose([1,2,0]))**gfac

def cbar_dopp_img(cbaxis,dopmin,dopmax,gfac=1.0/2.2, unit='Angstrom'):
    cbimg = np.outer(np.ones(5),np.linspace(dopmin,dopmax,255))
    cbaxis.imshow(color_dopp_img(cbimg,dopmin,dopmax,mask=np.ones(cbimg.shape),gfac=gfac),extent=[dopmin,dopmax,0,1],aspect='auto')
    cbaxis.set(xlabel='Wavelength shift ('+unit+')')

def doppler_plot(linefits,dopp_err_thold=0.025, axis=None, cbaxis=None, doppmin=-0.1, doppmax=0.1, err_thold=0.025, ymin=0, ymax=None):

	if(axis is None):
		fig = plt.figure(figsize=[7,11])
		gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[10, 1], hspace=0.25)	
		cbaxis = fig.add_subplot(gs[1,:])
		axes = []
		axes.append(fig.add_subplot(gs[0,0]))
	else:
		axes = [axis]

	dopp_err = linefits['centers'].uncertainty.array.squeeze()
	metadat = linefits['centers'].meta
	dopp_err_mask = (dopp_err > 0)*(dopp_err < err_thold)
	dopp_detrend = detrend_dopp(linefits['centers'])
	# dopp_detrend = linefits['centers'].data.squeeze()
	dopp_center = np.median(dopp_detrend[dopp_err_mask])
	dopp_err_falloff = (dopp_err > 0)*np.clip(err_thold/dopp_err,None,1)**2
    
	dopp_errmod = copy.deepcopy(dopp_detrend) - dopp_center
	# dopp_errmod = np.sign(dopp_errmod)*np.clip(np.abs(dopp_errmod)-np.abs(1.0*dopp_err),0,None)+dopp_center
    
	if(ymax is None): ymax = dopp_errmod.shape[1]
	# if(ymax is None): ymax = dopp_errmod.shape[1]
	# axes[0].imshow(color_dopp_img(dopp_errmod[:,ymin:ymax].T, dopp_center+doppmin, dopp_center+doppmax, dopp_err_falloff[:,ymin:ymax].T), aspect = metadat['CDELT2']/metadat['CDELT1'])
	norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=0.075)
	im = axes[0].imshow(dopp_errmod.T, origin="lower", interpolation="none", norm=norm, aspect = metadat['CDELT2']/metadat['CDELT1'], cmap="bwr")
	plt.colorbar(im, cax=cbaxis, label=metadat['BUNIT'], orientation="horizontal")

	# axes[0].imshow(color_dopp_img(dopp_errmod[:,ymin:ymax].T, dopp_center+doppmin, dopp_center+doppmax, dopp_err_falloff[:,ymin:ymax].T), 
	# 			aspect = metadat['CDELT2']/metadat['CDELT1'], origin="lower", interpolation="none")
	# if(cbaxis is not None):
	# 	cbar_dopp_img(cbaxis, doppmin, doppmax, unit=metadat['BUNIT'])

def amplitude_plot(linefits, axis=None, cbaxis=None, max_amp=None, ymin=0, ymax=None):
	if(axis is None):
		fig = plt.figure(figsize=[7,11])
		gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[10, 1], hspace=0.25)	
		cbaxis = fig.add_subplot(gs[1,:])
		axes = []
		axes.append(fig.add_subplot(gs[0,0]))
	else:
		axes = [axis]
	
	metadat = linefits['amplitudes'].meta
	if(ymax is None): ymax = linefits['amplitudes'].data.squeeze().shape[1]
	amp_plot = linefits['amplitudes'].data.squeeze()[:,ymin:ymax]
	max_amp = 1.5*np.mean(amp_plot)+5*np.std(amp_plot)
	norm = get_range(amp_plot, stre="log", imin=1, imax=99.5)
	im = axes[0].imshow(amp_plot.T, origin="lower", interpolation="none", norm=norm, aspect = metadat['CDELT2']/metadat['CDELT1'])
	plt.colorbar(im, cax=cbaxis, orientation="horizontal", label=metadat['BUNIT'])

	# axes[0].imshow(amp_plot.T**0.5,vmin=0,vmax=max_amp**0.5,
    #            aspect = metadat['CDELT2']/metadat['CDELT1'],cmap=plt.get_cmap('gray'), 
	# 		   origin="lower", interpolation="none")
	# axes[0].set(title='Line fit peak intensity')
	# cbaxis.imshow(np.outer(np.ones(2),(np.arange(100)))**0.5,extent=[0,max_amp,0,1],cmap=plt.get_cmap('gray'),aspect='auto', 
	# 		     origin="lower", interpolation="none")
	# cbaxis.set(xlabel='Peak intensity ('+metadat['BUNIT']+')')