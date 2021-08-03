
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:36:24 2021

@author: elham
"""

import os
import xarray as xr
import numpy as np
from numpy import clip
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.ops import control_flow_ops
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from dedalus import public as de
from misc_functions import deviatoric_part
from misc_functions import array_of_tf_components
import fluid_functions as ff
from dedalus import public as de
from sklearn.model_selection import train_test_split
import tensorflow.keras
import tensorflow.keras.losses
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.python.ops import math_ops



# =============================================================================
# Parameters
# =============================================================================

# Domain
L = 1
Bx = By = (-np.pi*L, np.pi*L)
N = 4096
Nx = Ny = N
N_filter = 1024
mesh = None

# =============================================================================
# Definition of quantities
# =============================================================================

x_basis = de.Fourier('x', N_filter, interval=Bx, dealias=3/2)
y_basis = de.Fourier('y', N_filter, interval=By, dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=None)



dx = domain.bases[0].Differentiate
dy = domain.bases[1].Differentiate

kx = domain.elements(0)
ky = domain.elements(1)
k = np.sqrt(kx**2 + ky**2)



# =============================================================================
# load data
# =============================================================================



def load_images(inputPath):
    
    images = sorted(glob.glob("/home/elham64/scratch/data/**/*.nc", recursive=True))
    
    # initialize our list of input and label of dataset 
    input_dset = []
    label_dset = []
    
    # loop over the indexes of the snapshots
    for img in images:
    
        dset = xr.open_dataset(img)
        comps = ['xx', 'yy','xy']
        ux = dset['ux'].data
        uy = dset['uy'].data
        comps = ['xx', 'yy','xy']
 
        ux_field = domain.new_field(name = 'ux')
        uy_field = domain.new_field(name = 'uy')
        ux_field['g'] = ux
        uy_field['g'] = uy

   
        # Compute inputs from velocity fields
        du_field = ff.uxuy_derivatives(domain,ux_field,uy_field)
   
        # Velocity field derivatives
        du = [du['g'] for du in du_field]
        du = np.moveaxis(np.array(du), 0, -1)
        du = du.astype(np.float64)
        meansdu = du.mean(axis=(0,1), dtype='float64')
        stdsdu = du.std(axis=(0,1), dtype='float64')
        du = (du - meansdu) / stdsdu
        input_dset.append(du)
        
        
        # Implicit subgrid stress
        tau = [dset['im_t'+c].data for c in comps]
        tau = np.moveaxis(np.array(tau), 0, -1)
        tau = tau.astype(np.float64)
        means = tau.mean(axis=(0,1), dtype='float64')
        stds = tau.std(axis=(0,1), dtype='float64')
        tau = (tau - means) / stds
        label_dset.append(tau)
        
        
    return np.array(input_dset), np.array(label_dset) 



tot_input, tot_label = load_images("/home/elham64/scratch/data/**/*.nc")  



X_train, X_test, y_train, y_test = train_test_split(tot_input, tot_label, test_size=0.3, random_state=42)


########################################################
#Power spectrum
########################################################
def build_azimuthal_mask(image_size):
    x1 = np.arange(image_size/2+1)
    x2 = np.arange(-image_size/2+1,0,1)
    x_tot = np.concatenate([x1, x2])
    x,y = np.meshgrid(x_tot,x_tot)
    R = np.sqrt((x)**2+(y)**2)
    masks = np.array(list(map(lambda r : (R >= r-.5) & (R < r+.5),np.arange(1,int((image_size/2)),1))))
    norm = np.sum(masks, axis=(1,2),keepdims=True)
    masks = masks/norm
    n = len(masks)
    
    return tf.reshape(tf.cast(masks,dtype=tf.float32),(1,n,image_size,image_size))


class PowerSpectrum(object):
    """ 
    Class for calculating a the power spectrum (1D or 2D) of an image in tensorflow.
    
    """
    def __init__(self,image_size, az_mask):
        """
        image_size: only needed
        
        """
        self.image_size = image_size  
        self.az_mask = az_mask        
        
    def power2D(self,x):
        x = tf.signal.fft2d(tf.cast(x,dtype=tf.complex64))
        x = tf.cast(x,dtype=tf.complex64)
        x = tf.abs(x)
        return tf.square(x)    
        
    def az_average(self,x):
        x=tf.reshape(x,(-1,1,self.image_size,self.image_size))
        return tf.reduce_sum(tf.reduce_sum(tf.multiply(self.az_mask,x),axis=3),axis=2)        
    
    def power1D(self,x):
        x = self.power2D(x)
        az_avg = self.az_average(x)
        
        return az_avg

def penalized_loss(az_mask):
    
    def costum_loss(y_true, y_pred):
         
        
        y_true_R = y_true[:,:,:,0]
        y_true_G = y_true[:,:,:,1]
        y_true_B = y_true[:,:,:,2]

        y_pred_R = y_pred[:,:,:,0]
        y_pred_G = y_pred[:,:,:,1]
        y_pred_B = y_pred[:,:,:,2]


        inputimage = PowerSpectrum(N_filter, az_mask)
        PS_true_R = inputimage.power1D(y_true_R)
        PS_true_G = inputimage.power1D(y_true_G)
        PS_true_B = inputimage.power1D(y_true_B)
        
        PS_pred_R = inputimage.power1D(y_pred_R)
        PS_pred_G = inputimage.power1D(y_pred_G)
        PS_pred_B = inputimage.power1D(y_pred_B)

        #diff_R = tf.math.log(tf.abs(PS_true_R/PS_pred_R)) 
        #diff_G = tf.math.log(tf.abs(PS_true_G/PS_pred_G))
        #diff_B = tf.math.log(tf.abs(PS_true_B/PS_pred_B))

        diff_R = (tf.math.log(PS_true_R)-tf.math.log(PS_pred_R))/N_filter**4
        diff_G = (tf.math.log(PS_true_G)-tf.math.log(PS_pred_G))/N_filter**4
        diff_B = (tf.math.log(PS_true_B)-tf.math.log(PS_pred_B))/N_filter**4

        w_diff = 1
        w_loss_PS = 1

        Loss_R_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_R, y_pred_R)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_R))
        Loss_G_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_G, y_pred_G)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_G))
        Loss_B_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_B, y_pred_B)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_B))

        #Loss_R = tf.reduce_mean(tf.abs(tf.math.log(tf.abs(diff_R))))
        #Loss_G = tf.reduce_mean(tf.abs(tf.math.log(tf.abs(diff_G))))
        #Loss_B = tf.reduce_mean(tf.abs(tf.math.log(tf.abs(diff_B))))


        #Loss_R = tf.reduce_mean(tf.square(tf.subtract(y_true_R, y_pred_R)))+tf.reduce_mean(tf.square(diff_R))
        #Loss_G = tf.reduce_mean(tf.square(tf.subtract(y_true_G, y_pred_G)))+tf.reduce_mean(tf.square(diff_G))
        #Loss_B = tf.reduce_mean(tf.square(tf.subtract(y_true_B, y_pred_B)))+tf.reduce_mean(tf.square(diff_B))

        Loss_R = tf.where((PS_true_R/N_filter**4)<5*10**(-11), Loss_R_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_R, y_pred_R))))
        Loss_G = tf.where((PS_true_G/N_filter**4)<5*10**(-11), Loss_G_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_G, y_pred_G))))
        Loss_B = tf.where((PS_true_B/N_filter**4)<10**(-10), Loss_B_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_B, y_pred_B))))  

        Loss_tot = Loss_R + Loss_G + Loss_B 
        
        return Loss_tot
    
    return costum_loss
     
######################################################
#Define Model1 and Model2
######################################################


class PeriodicPadding2D(keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(PeriodicPadding2D, self).__init__(**kwargs)
        
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config
        
    
    def wrap_pad(self, input, size):
        M1 = tf.concat([input[:,:, -size:], input, input[:,:, 0:size]], 2)
        M1 = tf.concat([M1[:, -size:, :], M1, M1[:, 0:size, :]], 1)
        return M1
                

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
        return shape

    def call(self, x, mask=None):
        return self.wrap_pad(x, self.padding[0])
    

class PeriodicPaddingUpsample2D(keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(PeriodicPaddingUpsample2D, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config
        
    
    def wrap_pad(self, input, size):
        M1 = tf.concat([input, input[:,:, 0:size]], 2)
        M1 = tf.concat([M1, M1[:, 0:size, :]], 1)
        return M1
                

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] + self.padding[0], input_shape[2] + self.padding[1], input_shape[3])
        return shape

    def call(self, x, mask=None):
        return self.wrap_pad(x, self.padding[0])



      
def build_Unet(inputs):
    
    #inputs = Input(shape=(width, height, channels))
    
    pd = 'valid'
    kinit = 'he_normal'
    alpha = 1e-9
    filter_sizes = (3, 3)
    
    c1 = PeriodicPadding2D(padding=(1,1))(inputs)
    c1 = Conv2D(16, filter_sizes, kernel_initializer=kinit, padding=pd) (c1) #activation=act) (c1)
    c1 = LeakyReLU(alpha=alpha)(c1)
    c1 = PeriodicPadding2D(padding=(1,1))(c1)
    c1 = Conv2D(16, filter_sizes, kernel_initializer=kinit, padding=pd) (c1) #activation=act) (c1)
    c1 = LeakyReLU(alpha=alpha)(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = PeriodicPadding2D(padding=(1,1))(p1)
    c2 = Conv2D(32, filter_sizes, kernel_initializer=kinit, padding=pd) (c2) #activation=act) (c2)
    c2 = LeakyReLU(alpha=alpha)(c2)
    c2 = PeriodicPadding2D(padding=(1,1))(c2)
    c2 = Conv2D(32, filter_sizes, kernel_initializer=kinit, padding=pd) (c2) #activation=act) (c2)
    c2 = LeakyReLU(alpha=alpha)(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = PeriodicPadding2D(padding=(1,1))(p2)
    c3 = Conv2D(64, filter_sizes, kernel_initializer=kinit, padding=pd) (c3) #activation=act) (c2)
    c3 = LeakyReLU(alpha=alpha)(c3)
    c3 = PeriodicPadding2D(padding=(1,1))(c3)
    c3 = Conv2D(64, filter_sizes, kernel_initializer=kinit, padding=pd) (c3) #activation=act) (c2)
    c3 = LeakyReLU(alpha=alpha)(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = PeriodicPadding2D(padding=(1,1))(p3)
    c4 = Conv2D(128, filter_sizes, kernel_initializer=kinit, padding=pd) (c4) #activation=act) (c5) 
    c4 = LeakyReLU(alpha=alpha)(c4)
    c4 = PeriodicPadding2D(padding=(1,1))(c4)
    c4 = Conv2D(128, filter_sizes, kernel_initializer=kinit, padding=pd) (c4) #activation=act) (c5)
    c4 = LeakyReLU(alpha=alpha)(c4)
    c4 = Dropout(0.5) (c4) 
    
    u5 = PeriodicPaddingUpsample2D(padding=(1,1))(UpSampling2D(size = (2,2))(c4))
    u5 = Conv2D(64, (2,2), kernel_initializer=kinit, padding=pd) (u5) #activation=act) (c6)
    u5 = concatenate([u5, c3], axis = 3)
    c5 = PeriodicPadding2D(padding=(1,1))(u5)
    c5 = Conv2D(64, filter_sizes, kernel_initializer=kinit, padding=pd) (c5) #activation=act) (c7)
    c5 = LeakyReLU(alpha=alpha)(c5)
    c5 = PeriodicPadding2D(padding=(1,1))(c5)
    c5 = Conv2D(64, filter_sizes, kernel_initializer=kinit, padding=pd) (c5) #activation=act) (c7)
    c5 = LeakyReLU(alpha=alpha)(c5)
    
    u6 = PeriodicPaddingUpsample2D(padding=(1,1))(UpSampling2D(size = (2,2))(c5))
    u6 = Conv2D(32, (2,2), kernel_initializer=kinit, padding=pd) (u6) #activation=act) (c6)
    u6 = concatenate([u6, c2], axis = 3)
    c6 = PeriodicPadding2D(padding=(1,1))(u6)
    c6 = Conv2D(32, filter_sizes, kernel_initializer=kinit, padding=pd) (c6) #activation=act) (c7)
    c6 = LeakyReLU(alpha=alpha)(c6)
    c6 = PeriodicPadding2D(padding=(1,1))(c6)
    c6 = Conv2D(32, filter_sizes, kernel_initializer=kinit, padding=pd) (c6) #activation=act) (c7)
    c6 = LeakyReLU(alpha=alpha)(c6)
    
    u7 = PeriodicPaddingUpsample2D(padding=(1,1))(UpSampling2D(size = (2,2))(c6))
    u7 = Conv2D(16, (2,2), kernel_initializer=kinit, padding=pd) (u7) #activation=act) (c6)
    u7 = concatenate([u7, c1], axis = 3)
    c7 = PeriodicPadding2D(padding=(1,1))(u7)
    c7 = Conv2D(16, filter_sizes, kernel_initializer=kinit, padding=pd) (c7) #activation=act) (c7)
    c7 = LeakyReLU(alpha=alpha)(c7)
    c7 = PeriodicPadding2D(padding=(1,1))(c7)
    c7 = Conv2D(16, filter_sizes, kernel_initializer=kinit, padding=pd) (c7) #activation=act) (c7)
    c7 = LeakyReLU(alpha=alpha)(c7)
    
    y_out = Conv2D(3, (1, 1), activation='linear', name="Unet_output") (c7)
    
    
    return y_out


     
def build_general(width, height, channels):
    
    
    inputs = Input(shape=(width, height, channels), name= 'inputs')
    Unet_out = build_Unet(inputs)
    model = Model(inputs=[inputs], outputs=[Unet_out])
    az_mask = build_azimuthal_mask(1024)

    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=opt, loss=penalized_loss(az_mask = az_mask), metrics=['accuracy'])
    

    model.summary()
    
    
    
    results = model.fit(X_train, y_train, validation_split=0.2, batch_size=1, epochs=25000, verbose=1)
    preds_tau = model.predict(X_test, verbose=1)
  
    return results, preds_tau



H, preds = build_general(1024, 1024, 4)



###############################################################
# plot the total loss, Unet loss, and PS loss
##################################################


plt.plot(H.history['val_loss'], label='validation')
plt.plot(H.history['loss'], label='train')
plt.ylabel("$loss$")
plt.xlabel("$epoch$")
plt.legend()
plt.savefig('loss_Ny_25000.png')
plt.close()



###############################################################
# plot the accuracies
##################################################

plt.plot(H.history['val_accuracy'], label='validation')
plt.plot(H.history['accuracy'], label='train')
plt.ylabel("$accuracy$")
plt.xlabel("$epoch$")
plt.legend()
plt.savefig('accuracy_Ny_25000.png')
plt.close()


#################################################
#Plot Power spectrum
################################### 

class PowerSpectrum_plot(object):
    """ Class for calculating a the power spectrum (1D or 2D) of an image in tensorflow.
        Expects square image as input.
    """
    def __init__(self,image_size=None):
        """image_size: only needed
        """
        self.image_size = image_size
        self.az_mask=self.build_azimuthal_mask()
        

    def power2D(self,x):
        x = np.fft.fft2(x)
        x = np.abs(x)
        return np.square(x)
     
    def build_azimuthal_mask(self):
        
            
        x1 = np.arange(self.image_size/2+1)
        x2 = np.arange(-self.image_size/2+1,0,1)
        x_tot = np.concatenate([x1, x2])
        x,y = np.meshgrid(x_tot,x_tot)
        R = np.sqrt((x)**2+(y)**2)
        masks = np.array(list(map(lambda r : (R >= r-.5) & (R < r+.5),np.arange(1,int(self.image_size/2),1))))
        norm = np.sum(masks,axis=(1,2),keepdims=True)
        masks = masks/norm
        n = len(masks)
        return np.reshape(masks,(1,n,self.image_size,self.image_size))
        
    def az_average(self,x):
        x=np.reshape(x,(-1,1,self.image_size,self.image_size))
        return np.sum(np.sum(np.multiply(self.az_mask,x),axis=3),axis=2)
    
    def power1D(self,x):
        x = self.power2D(x)
        az_avg = self.az_average(x)
        
        return az_avg
    

fig, ax = plt.subplots(1,3,figsize=(10*3,8))

titles = [r'$\tau_{xy}$',r'$\tau_{xx}$',r'$\tau_{yy}$',]

l = 0
for i in [0,1,2]:
    if l == 3:
        break
    for axis in ['top','bottom','left','right']:
        ax[l].spines[axis].set_linewidth(3)
    ax[l].tick_params(direction='out',width=2,length=5,labelsize = 20)

   
    image_ps = PowerSpectrum_plot(N_filter)
    PS_label = image_ps.power1D(y_test[0,:,:,i])
    PS_pred = image_ps.power1D(preds[0,:,:,i])
    
    
    xplot = np.linspace(0, 511, 511)
    PS_l = PS_label[0,:]/N_filter**4
    PS_t = PS_pred[0,:]/N_filter**4

    ax[l].plot(xplot,PS_l,label='true')
    ax[l].plot(xplot,PS_t,label='pred')
    ax[l].set_title(titles[l],fontsize=26)

    ax[l].set_ylabel("$PS(k)$",fontsize=20)
    ax[l].set_xlabel("$k$",fontsize=20)
    ax[l].set_yscale("log")
    ax[l].set_xscale("log")
    plt.legend()

    l += 1
plt.savefig('power_spectrum_Ny_25000.png',bbox_inches='tight')


