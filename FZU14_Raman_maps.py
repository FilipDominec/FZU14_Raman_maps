#!/usr/bin/python3  
#-*- coding: utf-8 -*-
## TODO extent=(-3, 3, 3, -3)

## Import common moduli
import sys, os, time, collections
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import spectral_preprocessing
import parametrizer

# Static settings
#x_label, y_label = 'PL peak (nm)', 'Intensity (a.u.)'
x_label, y_label = 'Raman peak (cm⁻¹)', 'Intensity (a.u.)'
interp_style = 'hanning'
interp_style2 = 'hanning'
DEBUG = 0 

## Load data from Horiba Labspec TXT format
if len(sys.argv) >= 2:
    filename = sys.argv[1]
else:
    import tkinter
    import tkinter.filedialog as fd
    root = tkinter.Tk() 
    # at least on Ubuntu, I had trouble with the interactive plot opening in background, found no solution
    #root.wm_attributes('-topmost', -1) 
    root.withdraw()
    #root.iconify()
    filename = fd.askopenfilename(filetypes=[("Converted Labspec L6M maps", "*.txt"), ("All files", "*.*"),])
    #root.destroy()
    #time.sleep(1)

#matplotlib.rcParams["savefig.directory"] = Path(filename).resolve().parent # easier saving of output graph FIXME
# TypeError: expected str, bytes or os.PathLike object, not tuple

#matplotlib.rcParams["tk.window_focus"] = True
#

## Old text format from Horiba
#wl = np.genfromtxt(filename, unpack=True, max_rows=1) # 1st row is the spectral ordinate (nm or reciprocal cm)
#raw = np.genfromtxt(filename, unpack=True, skip_header=1)
#x, y, rawI = np.unique(raw[0]), np.unique(raw[1]), raw[2:] # interpret three columns
#if not DEBUG: 
    #I = rawI.reshape([len(wl), len(x), len(y)])
#else:
    #I = rawI.reshape([len(wl), len(x), len(y)])[::2, ::2, ::2] # for debug only: decimate data to plot it quicker
    #wl = wl[::2]

## New text format from Horiba/Robert H.

#Relative X
with open(filename) as of:
    for n,l in enumerate(of.readlines()):
        if 'Relative X' in l: line_relX = n
        if 'Relative Y' in l: line_relY = n
        if 'Counts' in l: line_relCounts = n

    of.seek(0)
    x = np.unique(np.genfromtxt(of, unpack=True, skip_header=line_relX, max_rows=1)[3:]) # index 33 should be the first after empty line
    of.seek(0)
    y = np.unique(np.genfromtxt(of, unpack=True, skip_header=line_relY, max_rows=1)[3:]) # and below this line
    of.seek(0)
    raw = np.genfromtxt(of, unpack=True, skip_header=line_relCounts+1) # 36 comes from #line containing 'Counts'
print('counts.shape', raw[1:,:].shape, x, y)
wl = raw[0,:]
I = raw[1:,:].T.reshape([len(wl), len(x), len(y)])

## spectra preprocessing
I = np.apply_along_axis(spectral_preprocessing.retouch_outliers, 0, I)
I = np.apply_along_axis(spectral_preprocessing.smooth, 0, I)
I = np.apply_along_axis(spectral_preprocessing.rm_bg, 0, I)
I = np.apply_along_axis(spectral_preprocessing.norm, 0, I)


## Generate custom colormaps
def isoluminant_rainbow_rgb(val):
    val = np.pi-val  ## reversed
    return (1+np.cos((val)*np.pi))*.35, (1+np.cos((val-2/3)*np.pi))*.25, (1+np.cos((val-4/3)*np.pi))*.5
def isoluminant_greenpink_rgb(val):
    return val**.8*.8, (1-val**1.5)*.5, val**.8
def greyscale_rgb(val):
    return [val*.8] * 3 
def cmap_segmentize(fn):
    return dict([
        [channel,      [[x]+[fn(x)[channel_idx]]*2 for x in np.linspace(0,1)]] 
        for (channel_idx,channel) in enumerate(('red', 'green', 'blue'))])

## Colormaps are for colorbars (main plots compose each pixel's RGB colors procedurally)
isolum_rainbow_colormap = matplotlib.colors.LinearSegmentedColormap('isolum_rainbow_colormap', cmap_segmentize(isoluminant_rainbow_rgb), 50)
isolum_greenpink_colormap = matplotlib.colors.LinearSegmentedColormap('isolum_greenpink_colormap', cmap_segmentize(isoluminant_greenpink_rgb), 50)
brightness_colormap = matplotlib.colors.LinearSegmentedColormap('brightness_colormap', cmap_segmentize(greyscale_rgb), 50)


## Prepare plotting with mpl's object model
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 14))
fig.canvas.draw() ## TODO test foreground!

# (changing suggested filename this way is a bit hack-ish, as reported here: https://github.com/matplotlib/matplotlib/issues/27774)
fig.canvas.get_default_filename = lambda: Path(filename).resolve().stem + '.png'
fig.canvas.manager.set_window_title(Path(filename).resolve().name)

axs[0,1].remove(); axs[0,2].remove() # make space for interactive sliders

sp_select = []
hs = []
zs = []
ims = []
bbars = []
bbars2 = []
cims = []
cims_overlay = []
cbars = []
cbars2 = []

def generate_subplots():

    ## Spectrogram in upper left:
    Iflat = I.reshape([len(wl),-1])
    axs[0,0].plot(wl[::3], np.apply_over_axes(np.mean, I, (1,2))[::3,0,0], alpha=1, c='k',lw=1)
    axs[0,0].fill_between(wl, np.quantile(Iflat, .001, axis=1), np.quantile(Iflat, .999, axis=1), alpha=.15, color='k')
    axs[0,0].fill_between(wl, np.quantile(Iflat, .01, axis=1), np.quantile(Iflat, .99, axis=1), alpha=.25, color='k')
    axs[0,0].fill_between(wl, np.quantile(Iflat, .1, axis=1), np.quantile(Iflat, .9, axis=1), alpha=.5, color='k')
    axs[0,0].set_xlabel(x_label); 
    axs[0,0].set_ylabel(y_label); 
    axs[0,0].set_xlim(xmin=min(wl), xmax=max(wl)); 
    axs[0,0].grid()

    for c in 'bgr':
        hs.append(axs[0,0].axvspan(100,200, facecolor=c, alpha=0.3))

    ## 2D maps 
    for i in range(axs.shape[1]):
        sp_select.append(axs[0,0].plot([1,2], [1,2], alpha=1, c='r',lw=1))
        zs.append(I.mean(axis=0))

        ## Middle portion
        ims.append(axs[1,i].imshow(np.abs(zs[i])**.3, interpolation=interp_style, cmap=None)) # no cmap, will use R-G-B
        #bar = plt.colorbar(ims[-1], ax=axs[1,i], pad=0.01) 
        axs[1,i].set_xlabel(u"Position x (μm)"); 
        axs[1,i].set_ylabel(u"Position y (μm)"); 
        bbars.append(plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=isolum_greenpink_colormap), ax=axs[1,i], pad=-0.05))
        bbars[-1].ax.set_title(f'width of {x_label}', rotation='vertical', x=.65, y=.5, 
                fontsize=8, fontweight='semibold', color='white', verticalalignment='center')
        bbars[-1].ax.tick_params(size=0, labelsize=8)
        bbars[-1].outline.set_visible(False)
        bbars2.append(plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=brightness_colormap), ax=axs[1,i], pad=0.01))
        bbars2[-1].ax.set_title(f'mean {y_label}', rotation='vertical', x=.65, y=.5, 
                fontsize=8, fontweight='semibold', color='white', verticalalignment='center')
        bbars2[-1].ax.tick_params(size=0, labelsize=8)
        bbars2[-1].outline.set_visible(False)


        ## Bottom portion
        cims.append(axs[2,i].imshow(np.abs(zs[i])**.3, interpolation=interp_style, cmap=None))
        #cims_overlay.append(axs[2,i].imshow(np.abs(zs[i])**.3, interpolation=interp_style2, cmap=None))
        cbars.append(plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=isolum_rainbow_colormap), ax=axs[2,i], pad=-0.05))
        cbars[-1].ax.set_title(f'wavelength of {x_label}', rotation='vertical', x=.65, y=.5, 
                fontsize=8, fontweight='semibold', color='white', verticalalignment='center')
        cbars[-1].ax.tick_params(size=0, labelsize=8)
        cbars[-1].outline.set_visible(False)
        cbars2.append(plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=brightness_colormap), ax=axs[2,i], pad=0.01))
        cbars2[-1].ax.set_title(f'mean {y_label}', rotation='vertical', x=.65, y=.5, 
                fontsize=8, fontweight='semibold', color='white', verticalalignment='center')
        cbars2[-1].ax.tick_params(size=0, labelsize=8)
        cbars2[-1].outline.set_visible(False)


def rcm_to_index(current):  
    #return int((current-np.min(wl)) / (np.max(wl)-np.min(wl)) * I.shape[0]) # fast but wrong - x-axis may be nonlinear
    return int(np.interp(current, wl, np.arange(len(wl))) +.5)


def half_max_x(y, *args, half=0.85):
    """ returns the average two points at X: where the curve first rises above half maximum, and where it last drops below it """
    def lin_interp(x, y, i, halfmax):
        return x[i] + (x[i+1] - x[i]) * ((halfmax - y[i]) / (y[i+1] - y[i]))
    x = args[0][0]
    halfmax = max(y)*half
    signs = np.sign(np.add(y, -halfmax))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    if len(zero_crossings_i) >= 2:
        return (lin_interp(x, y, zero_crossings_i[0], halfmax), lin_interp(x, y, zero_crossings_i[-1], halfmax))
    else:
        return (np.NaN,np.NaN)


def update_plots(slider_name, new_value):
    for bandn in range(1,3+1): 
        if not slider_name or (slider_name.startswith('band') and bandn==int(slider_name.lstrip('band')[0])):
            center_cm, width_cm = p.get(f'band{bandn}c'), p.get(f'band{bandn}w')
            print(center_cm, width_cm)
            left_cm, right_cm = center_cm-width_cm/2, center_cm+width_cm/2
            left_cm = max(min(wl), left_cm)
            right_cm = min(max(wl), right_cm)

            hs[bandn-1].set_xy([[left_cm, 0.], [left_cm, 1.], [right_cm,  1.], [right_cm,  0.], [left_cm, 0.]]) # TODO use blit

            y_slice = I[rcm_to_index(left_cm):rcm_to_index(right_cm),:,:]
            x_slice = wl[rcm_to_index(left_cm):rcm_to_index(right_cm)]

            mean_values = y_slice.mean(axis=0)

            #discriminator = 0.05 # avoid searching for maximum position of very weak peaks
            #peak_values = y_slice.argmax(axis=0)*1.0
            #peak_values[mean_values<np.max(mean_values)*discriminator] = np.NaN
            #peak_values[mean_values<np.max(mean_values)*discriminator] = np.NaN

            # OPTION 1: crossings 
            crossings = np.apply_along_axis(
                    half_max_x,
                    0, 
                    y_slice,
                    (x_slice,)) # 
            peak_positions = (crossings[1,:,:] + crossings[0,:,:])/2
            peak_widths = (crossings[1,:,:] - crossings[0,:,:])

            # OPTION 2: sunken-function averaging & moments
            #peak_positions = np.apply_along_axis( np.mean, 0, y_slice*x_slice) / np.apply_along_axis( np.sum, 0, y_slice)
            #peak_widths = (crossings[1,:,:] - crossings[0,:,:])

            minpp, maxpp = np.nanmin(peak_positions), np.nanmax(peak_positions)
            minpw, maxpw = np.nanmin(peak_widths), np.nanmax(peak_widths)

            brightness = mean_values/np.nanmax(mean_values)**1    *2
            peak_positions_normed = (peak_positions-minpp)/(maxpp-minpp)  # WTFZ, should not be mutiplied!?

            minpw = 0  # force zero width
            maxpw *= 2  # force 
            peak_widths_normed = (peak_widths-minpw)/(maxpw-minpw) 
            peak_widths_normed[np.isnan(peak_widths_normed)] = 0# (maxpw+minpw)/2 # Artificial value to indicate peak position not detected

            sp_select[bandn-1][0].set_data( x_slice, y_slice[:,0,0]) 
            #sp_select[bandn-1][0].set_data( x_slice, (y_slice - np.linspace(y_slice[0],y_slice[-1], len(y_slice)))[:,0,0]) 
            #sp_select[bandn-1][0].set_data( x_slice, spectral_preprocessing.rm_bg( y_slice[:,0,0])) 

            #cims[bandn-1].set(data=peak_positions, clim=(np.nanmin(peak_positions), np.nanmax(peak_positions)))

            # TODO isoluminant for energy   X   brighness for intensity
            #https://colorcet.holoviz.org/user_guide/Continuous.html#testing-perceptual-uniformity

            #2nd row
            rgb_values1 = np.dstack([brightness*ar for ar in isoluminant_greenpink_rgb(peak_widths_normed)])
            ims[bandn-1].set(data=rgb_values1) # , clim=(0,1))
            bbars[bandn-1].mappable.set_norm( matplotlib.colors.Normalize(minpw,maxpw))
            bbars2[bandn-1].mappable.set_norm( matplotlib.colors.Normalize(0,np.nanmax(mean_values)))


            #3rd row
            rgb_values2 = np.dstack([brightness*ar for ar in isoluminant_rainbow_rgb(peak_positions_normed)])
            rgb_values2[np.where(np.isnan(rgb_values2))] = 0
            cims[bandn-1].set( data=rgb_values2, clim=(0, 1)) # np.nanmax(rgb_values2)))

            #masked_data = np.ma.masked_where(np.isnan(np.sum(rgb_values2, axis=2)), rgb_values2)
            #cims_overlay[bandn-1].set(data=masked_data) # , clim=(0, 1)) # , clim=(0,1))

            ## TODO update cbar1s, cbar2s according to intensity and position of the peak
            cbars[bandn-1].mappable.set_norm( matplotlib.colors.Normalize(minpp,maxpp))
            cbars2[bandn-1].mappable.set_norm( matplotlib.colors.Normalize(0,np.nanmax(mean_values)))


    fig.canvas.draw_idle()
    fig.canvas.flush_events()

wl1, wl2, wlr = np.min(wl), np.max(wl), np.max(wl)-np.min(wl)
default_params = collections.OrderedDict([
    ['band1c',     (wl1,    wl1+wlr*.25, wl2)],
    ['band1w',     (0,      wlr/6,   wlr)],
    ['band2c',     (wl1,    wl1+wlr*.5,  wl2)],
    ['band2w',     (0,      wlr/6,   wlr)],
    ['band3c',     (wl1,    wl1+wlr*.75, wl2)],
    ['band3w',     (0,      wlr/6,   wlr)],
    ])

generate_subplots()

p = parametrizer.Parametrizer(
        figure=fig,
        default_params=default_params, 
        update_function=update_plots)

update_plots(None, None)

## ==== Outputting ====

#fig.canvas.mpl_connect('close_event', quit);  

plt.tight_layout()
plt.show()
cfm = plt.get_current_fig_manager() # 'fig'
print(cfm)
cfm.window.activateWindow()
cfm.window.raise_()
fig.canvas.draw()
#AttributeError: '_tkinter.tkapp' object has no attribute 'raise_'


fig.savefig(f"output{'debug' if DEBUG else ''}.png") # 


#if DEBUG:
    #I = I[::2, 6:15, 6:15]
    #wl = wl[::2]
