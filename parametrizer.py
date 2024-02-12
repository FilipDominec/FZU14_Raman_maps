#!/usr/bin/python3  
#-*- coding: utf-8 -*-

import matplotlib
import collections
 
"""
A class that simplifies user interaction with interactive Matplotlib graphs. 

For a given set of parameters, it generates a visible array of matplotlib.sliders, the value of which the user can 
modify by clicking/dragging. The slider values are easy to save into a file and reload them upon your application 
restart. 
"""

class Parametrizer():
    def __init__(self, 
            figure, 
            default_params, 
            update_function, 
            sliderheight=.02, 
            slider_vpos=.825, 
            extra_spaced_sliders=[]):

        self.fig = figure
        self.default_params = default_params
        self.update = update_function

        self.variable_parameters = {}
        self.load_saved_parameters()
        self.sliderheight = sliderheight
        self.slider_vpos = slider_vpos
        self.extra_spaced_sliders = extra_spaced_sliders
        self.create_sliders()

    ## Loading and access to the previously saved image processing parameters
    def load_saved_parameters(self, settingsfilename='./saved_parameters.dat'):
        try:
            with open(settingsfilename) as settingsfile:
                for n, line in enumerate(settingsfile.readlines()):
                    try:
                        key, val = line.split('=', 1)
                        self.variable_parameters[key.strip()] = float(val)
                    except ValueError:
                        print("Warning: could not process value `{}` for key `{}` in line #{} in `{}`".format(key, val, n, settingsfilename))
        except IOError:
            print("Warning: could not read `{}` in the working directory; using default values".format(settingsfilename))
        #return self.variable_parameters


    ## GUI: Generate sliders for each image-processing parameter in pre-determined part of the window (todo - allow user specify it)
    def create_sliders(self):
        self.paramsliders = collections.OrderedDict()
        
        for key,item in list(self.default_params.items())[::-1]:
            self.paramsliders[key] = matplotlib.widgets.Slider(
                    self.fig.add_axes([0.4, self.slider_vpos, 0.55, self.sliderheight]), 
                    key, item[0], item[2], 
                    valinit=self.variable_parameters.get(key.strip(), item[1])) # if no saved value, use default
            #self.paramsliders[key].on_changed(self.update)
            #self.paramsliders[key].on_changed(lambda x:self.update(key,x))
            from functools import partial
            
            self.paramsliders[key].on_changed(partial(self.update, key))
            self.slider_vpos += self.sliderheight*1.4 if key in (self.extra_spaced_sliders) else self.sliderheight

        self.button = matplotlib.widgets.Button(
                self.fig.add_axes([.8, 0.72, 0.1, self.sliderheight]),  ## TODO 
                'Save settings', 
                color='.7', 
                hovercolor='.9')
        self.button.on_clicked(self.save_values)

    def get(self,name):
        return self.paramsliders[name].val #    if __name__ == '__main__'   else self.variable_parameters[name] todo noninteractive

    def save_values(self, event): 
        print("\n\nSaving slider settings:\n")
        with open('./saved_parameters.dat', 'w') as of:
            for key,item in self.paramsliders.items(): 
                save_line = key + ' '*(40-len(key)) + ' = ' + str(item.val)
                of.write(save_line+'\n')
                print(save_line)


