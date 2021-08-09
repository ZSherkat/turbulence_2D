
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
from numpy import clip
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
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
def power2D(x):
    
    image_size = N_filter
    x1 = tf.range(image_size/2+1)
    x2 = tf.range(-image_size/2+1,0,1)
    x_tot = tf.concat([x1, x2],0)
    x_k,y_k = tf.meshgrid(x_tot,x_tot)
    R = tf.sqrt((x_k)**2+(y_k)**2)
    Rf = tf.reshape(R, [-1])
    
    
    x = tf.signal.fft2d(tf.cast(x,dtype=tf.complex64))
    x = tf.cast(x,dtype=tf.complex64)
    x = tf.abs(x)
    x = tf.square(x)
    x = tf.reshape(x, [-1])
    
        
    edges = [  0, 0.75,   1.5 ,   2.25,   3.  ,   3.75,   4.5 ,   5.25,
     6.  ,   6.75,   7.5 ,   8.25,   9.  ,   9.75,  10.5 ,  11.25,
    12.  ,  12.75,  13.5 ,  14.25,  15.  ,  15.75,  16.5 ,  17.25,
    18.  ,  18.75,  19.5 ,  20.25,  21.  ,  21.75,  22.5 ,  23.25,
    24.  ,  24.75,  25.5 ,  26.25,  27.  ,  27.75,  28.5 ,  29.25,
    30.  ,  30.75,  31.5 ,  32.25,  33.  ,  33.75,  34.5 ,  35.25,
    36.  ,  36.75,  37.5 ,  38.25,  39.  ,  39.75,  40.5 ,  41.25,
    42.  ,  42.75,  43.5 ,  44.25,  45.  ,  45.75,  46.5 ,  47.25,
    48.  ,  48.75,  49.5 ,  50.25,  51.  ,  51.75,  52.5 ,  53.25,
    54.  ,  54.75,  55.5 ,  56.25,  57.  ,  57.75,  58.5 ,  59.25,
    60.  ,  60.75,  61.5 ,  62.25,  63.  ,  63.75,  64.5 ,  65.25,
    66.  ,  66.75,  67.5 ,  68.25,  69.  ,  69.75,  70.5 ,  71.25,
    72.  ,  72.75,  73.5 ,  74.25,  75.  ,  75.75,  76.5 ,  77.25,
    78.  ,  78.75,  79.5 ,  80.25,  81.  ,  81.75,  82.5 ,  83.25,
    84.  ,  84.75,  85.5 ,  86.25,  87.  ,  87.75,  88.5 ,  89.25,
    90.  ,  90.75,  91.5 ,  92.25,  93.  ,  93.75,  94.5 ,  95.25,
    96.  ,  96.75,  97.5 ,  98.25,  99.  ,  99.75, 100.5 , 101.25,
   102.  , 102.75, 103.5 , 104.25, 105.  , 105.75, 106.5 , 107.25,
   108.  , 108.75, 109.5 , 110.25, 111.  , 111.75, 112.5 , 113.25,
   114.  , 114.75, 115.5 , 116.25, 117.  , 117.75, 118.5 , 119.25,
   120.  , 120.75, 121.5 , 122.25, 123.  , 123.75, 124.5 , 125.25,
   126.  , 126.75, 127.5 , 128.25, 129.  , 129.75, 130.5 , 131.25,
   132.  , 132.75, 133.5 , 134.25, 135.  , 135.75, 136.5 , 137.25,
   138.  , 138.75, 139.5 , 140.25, 141.  , 141.75, 142.5 , 143.25,
   144.  , 144.75, 145.5 , 146.25, 147.  , 147.75, 148.5 , 149.25,
   150.  , 150.75, 151.5 , 152.25, 153.  , 153.75, 154.5 , 155.25,
   156.  , 156.75, 157.5 , 158.25, 159.  , 159.75, 160.5 , 161.25,
   162.  , 162.75, 163.5 , 164.25, 165.  , 165.75, 166.5 , 167.25,
   168.  , 168.75, 169.5 , 170.25, 171.  , 171.75, 172.5 , 173.25,
   174.  , 174.75, 175.5 , 176.25, 177.  , 177.75, 178.5 , 179.25,
   180.  , 180.75, 181.5 , 182.25, 183.  , 183.75, 184.5 , 185.25,
   186.  , 186.75, 187.5 , 188.25, 189.  , 189.75, 190.5 , 191.25,
   192.  , 192.75, 193.5 , 194.25, 195.  , 195.75, 196.5 , 197.25,
   198.  , 198.75, 199.5 , 200.25, 201.  , 201.75, 202.5 , 203.25,
   204.  , 204.75, 205.5 , 206.25, 207.  , 207.75, 208.5 , 209.25,
   210.  , 210.75, 211.5 , 212.25, 213.  , 213.75, 214.5 , 215.25,
   216.  , 216.75, 217.5 , 218.25, 219.  , 219.75, 220.5 , 221.25,
   222.  , 222.75, 223.5 , 224.25, 225.  , 225.75, 226.5 , 227.25,
   228.  , 228.75, 229.5 , 230.25, 231.  , 231.75, 232.5 , 233.25,
   234.  , 234.75, 235.5 , 236.25, 237.  , 237.75, 238.5 , 239.25,
   240.  , 240.75, 241.5 , 242.25, 243.  , 243.75, 244.5 , 245.25,
   246.  , 246.75, 247.5 , 248.25, 249.  , 249.75, 250.5 , 251.25,
   252.  , 252.75, 253.5 , 254.25, 255.  , 255.75, 256.5 , 257.25,
   258.  , 258.75, 259.5 , 260.25, 261.  , 261.75, 262.5 , 263.25,
   264.  , 264.75, 265.5 , 266.25, 267.  , 267.75, 268.5 , 269.25,
   270.  , 270.75, 271.5 , 272.25, 273.  , 273.75, 274.5 , 275.25,
   276.  , 276.75, 277.5 , 278.25, 279.  , 279.75, 280.5 , 281.25,
   282.  , 282.75, 283.5 , 284.25, 285.  , 285.75, 286.5 , 287.25,
   288.  , 288.75, 289.5 , 290.25, 291.  , 291.75, 292.5 , 293.25,
   294.  , 294.75, 295.5 , 296.25, 297.  , 297.75, 298.5 , 299.25,
   300.  , 300.75, 301.5 , 302.25, 303.  , 303.75, 304.5 , 305.25,
   306.  , 306.75, 307.5 , 308.25, 309.  , 309.75, 310.5 , 311.25,
   312.  , 312.75, 313.5 , 314.25, 315.  , 315.75, 316.5 , 317.25,
   318.  , 318.75, 319.5 , 320.25, 321.  , 321.75, 322.5 , 323.25,
   324.  , 324.75, 325.5 , 326.25, 327.  , 327.75, 328.5 , 329.25,
   330.  , 330.75, 331.5 , 332.25, 333.  , 333.75, 334.5 , 335.25,
   336.  , 336.75, 337.5 , 338.25, 339.  , 339.75, 340.5 , 341.25,
   342.  , 342.75, 343.5 , 344.25, 345.  , 345.75, 346.5 , 347.25,
   348.  , 348.75, 349.5 , 350.25, 351.  , 351.75, 352.5 , 353.25,
   354.  , 354.75, 355.5 , 356.25, 357.  , 357.75, 358.5 , 359.25,
   360.  , 360.75, 361.5 , 362.25, 363.  , 363.75, 364.5 , 365.25,
   366.  , 366.75, 367.5 , 368.25, 369.  , 369.75, 370.5 , 371.25,
   372.  , 372.75, 373.5 , 374.25, 375.  , 375.75, 376.5 , 377.25,
   378.  , 378.75, 379.5 , 380.25, 381.  , 381.75, 382.5 , 383.25,
   384.  , 384.75, 385.5 , 386.25, 387.  , 387.75, 388.5 , 389.25,
   390.  , 390.75, 391.5 , 392.25, 393.  , 393.75, 394.5 , 395.25,
   396.  , 396.75, 397.5 , 398.25, 399.  , 399.75, 400.5 , 401.25,
   402.  , 402.75, 403.5 , 404.25, 405.  , 405.75, 406.5 , 407.25,
   408.  , 408.75, 409.5 , 410.25, 411.  , 411.75, 412.5 , 413.25,
   414.  , 414.75, 415.5 , 416.25, 417.  , 417.75, 418.5 , 419.25,
   420.  , 420.75, 421.5 , 422.25, 423.  , 423.75, 424.5 , 425.25,
   426.  , 426.75, 427.5 , 428.25, 429.  , 429.75, 430.5 , 431.25,
   432.  , 432.75, 433.5 , 434.25, 435.  , 435.75, 436.5 , 437.25,
   438.  , 438.75, 439.5 , 440.25, 441.  , 441.75, 442.5 , 443.25,
   444.  , 444.75, 445.5 , 446.25, 447.  , 447.75, 448.5 , 449.25,
   450.  , 450.75, 451.5 , 452.25, 453.  , 453.75, 454.5 , 455.25,
   456.  , 456.75, 457.5 , 458.25, 459.  , 459.75, 460.5 , 461.25,
   462.  , 462.75, 463.5 , 464.25, 465.  , 465.75, 466.5 , 467.25,
   468.  , 468.75, 469.5 , 470.25, 471.  , 471.75, 472.5 , 473.25,
   474.  , 474.75, 475.5 , 476.25, 477.  , 477.75, 478.5 , 479.25,
   480.  , 480.75, 481.5 , 482.25, 483.  , 483.75, 484.5 , 485.25,
   486.  , 486.75, 487.5 , 488.25, 489.  , 489.75, 490.5 , 491.25,
   492.  , 492.75, 493.5 , 494.25, 495.  , 495.75, 496.5 , 497.25,
   498.  , 498.75, 499.5 , 500.25, 501.  , 501.75, 502.5 , 503.25,
   504.  , 504.75, 505.5 , 506.25, 507.  , 507.75, 508.5 , 509.25,
   510.  , 510.75, 511.5 , 512.25, 513.  , 513.75, 514.5 , 515.25,
   516.  , 516.75, 517.5 , 518.25, 519.  , 519.75, 520.5 , 521.25,
   522.  , 522.75, 523.5 , 524.25, 525.  , 525.75, 526.5 , 527.25,
   528.  , 528.75, 529.5 , 530.25, 531.  , 531.75, 532.5 , 533.25,
   534.  , 534.75, 535.5 , 536.25, 537.  , 537.75, 538.5 , 539.25,
   540.  , 540.75, 541.5 , 542.25, 543.  , 543.75, 544.5 , 545.25,
   546.  , 546.75, 547.5 , 548.25, 549.  , 549.75, 550.5 , 551.25,
   552.  , 552.75, 553.5 , 554.25, 555.  , 555.75, 556.5 , 557.25,
   558.  , 558.75, 559.5 , 560.25, 561.  , 561.75, 562.5 , 563.25,
   564.  , 564.75, 565.5 , 566.25, 567.  , 567.75, 568.5 , 569.25,
   570.  , 570.75, 571.5 , 572.25, 573.  , 573.75, 574.5 , 575.25,
   576.  , 576.75, 577.5 , 578.25, 579.  , 579.75, 580.5 , 581.25,
   582.  , 582.75, 583.5 , 584.25, 585.  , 585.75, 586.5 , 587.25,
   588.  , 588.75, 589.5 , 590.25, 591.  , 591.75, 592.5 , 593.25,
   594.  , 594.75, 595.5 , 596.25, 597.  , 597.75, 598.5 , 599.25,
   600.  , 600.75, 601.5 , 602.25, 603.  , 603.75, 604.5 , 605.25,
   606.  , 606.75, 607.5 , 608.25, 609.  , 609.75, 610.5 , 611.25,
   612.  , 612.75, 613.5 , 614.25, 615.  , 615.75, 616.5 , 617.25,
   618.  , 618.75, 619.5 , 620.25, 621.  , 621.75, 622.5 , 623.25,
   624.  , 624.75, 625.5 , 626.25, 627.  , 627.75, 628.5 , 629.25,
   630.  , 630.75, 631.5 , 632.25, 633.  , 633.75, 634.5 , 635.25,
   636.  , 636.75, 637.5 , 638.25, 639.  , 639.75, 640.5 , 641.25,
   642.  , 642.75, 643.5 , 644.25, 645.  , 645.75, 646.5 , 647.25,
   648.  , 648.75, 649.5 , 650.25, 651.  , 651.75, 652.5 , 653.25,
   654.  , 654.75, 655.5 , 656.25, 657.  , 657.75, 658.5 , 659.25,
   660.  , 660.75, 661.5 , 662.25, 663.  , 663.75, 664.5 , 665.25,
   666.  , 666.75, 667.5 , 668.25, 669.  , 669.75, 670.5 , 671.25,
   672.  , 672.75, 673.5 , 674.25, 675.  , 675.75, 676.5 , 677.25,
   678.  , 678.75, 679.5 , 680.25, 681.  , 681.75, 682.5 , 683.25,
   684.  , 684.75, 685.5 , 686.25, 687.  , 687.75, 688.5 , 689.25,
   690.  , 690.75, 691.5 , 692.25, 693.  , 693.75, 694.5 , 695.25,
   696.  , 696.75, 697.5 , 698.25, 699.  , 699.75, 700.5 , 701.25,
   702.  , 702.75, 703.5 , 704.25, 705.  , 705.75, 706.5 , 707.25,
   708.  , 708.75, 709.5 , 710.25, 711.  , 711.75, 712.5 , 713.25,
   714.  , 714.75, 715.5 , 716.25, 717.  , 717.75, 718.5 , 719.25,
   720.  , 720.75, 721.5 , 722]
    
    
    n_hist = tfp.stats.histogram(Rf, edges)
    hist = tfp.stats.histogram(Rf, edges, weights=x)
    
    return hist/n_hist, edges   
      
    

    
def custom_loss(y_true, y_pred):
     
    
    y_true_R = y_true[:,:,:,0]
    y_true_G = y_true[:,:,:,1]
    y_true_B = y_true[:,:,:,2]
    
    y_pred_R = y_pred[:,:,:,0]
    y_pred_G = y_pred[:,:,:,1]
    y_pred_B = y_pred[:,:,:,2]
    

    PS_true_R, _ = power2D(y_true_R)
    PS_true_G, _ = power2D(y_true_G)
    PS_true_B, _ = power2D(y_true_B)
    
    PS_pred_R, _ = power2D(y_pred_R)
    PS_pred_G, _ = power2D(y_pred_G)
    PS_pred_B, _ = power2D(y_pred_B)
    
    LogTrueR = tf.where(PS_true_R>0, tf.math.log(PS_true_R), 0)
    LogTrueG = tf.where(PS_true_G>0, tf.math.log(PS_true_G), 0)
    LogTrueB = tf.where(PS_true_B>0, tf.math.log(PS_true_B), 0)
    
    diff_R = (LogTrueR-tf.math.log(PS_pred_R))/N_filter**4
    diff_G = (LogTrueG-tf.math.log(PS_pred_G))/N_filter**4
    diff_B = (LogTrueB-tf.math.log(PS_pred_B))/N_filter**4
    
    #tf.print("MA_log_PS_pred_R:", tf.reduce_mean(tf.abs(tf.math.log(PS_pred_R))), "Inside R loss function")
    #tf.print("MA_log_PS_pred_G:", tf.reduce_mean(tf.abs(tf.math.log(PS_pred_G))), "Inside G loss function")
    #tf.print("MA_log_PS_pred_B:", tf.reduce_mean(tf.abs(tf.math.log(PS_pred_B))), "Inside B loss function")
    
    #tf.print("MA_log_PS_true_R:", LogTrueR, "Inside R loss function")
    #tf.print("MA_log_PS_true_G:", LogTrueG, "Inside G loss function")
    #tf.print("MA_log_PS_true_B:", LogTrueB, "Inside B loss function")
    
    
    #tf.print("diff_R:", tf.reduce_mean(tf.abs(diff_R)), "diff R Inside loss function")
    #tf.print("diff_G:", tf.reduce_mean(tf.abs(diff_G)), "diff G Inside loss function")
    #tf.print("diff_B:", tf.reduce_mean(tf.abs(diff_B)), "diff B Inside loss function")
    
    w_diff = 1
    w_loss_PS = 1

    Loss_R_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_R, y_pred_R)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_R))
    Loss_G_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_G, y_pred_G)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_G))
    Loss_B_log = w_diff*tf.reduce_mean(tf.abs(tf.subtract(y_true_B, y_pred_B)))+w_loss_PS*tf.reduce_mean(tf.abs(diff_B))
    
    Loss_R = tf.where((PS_true_R/N_filter**4)<10**(-12), Loss_R_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_R, y_pred_R))))
    Loss_G = tf.where((PS_true_G/N_filter**4)<10**(-12), Loss_G_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_G, y_pred_G))))
    Loss_B = tf.where((PS_true_B/N_filter**4)<10**(-12), Loss_B_log, tf.reduce_mean(tf.abs(tf.subtract(y_true_B, y_pred_B))))  
    
    Loss_tot = Loss_R + Loss_G + Loss_B 
    
    return Loss_tot


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
    

    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    #opt = Adamax(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=opt, loss=custom_loss, metrics=['accuracy'])
    

    model.summary()
    
    
    
    results = model.fit(X_train, y_train, validation_split=0.2, batch_size=1, epochs=70000, verbose=1)
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
plt.savefig('loss_withlog_70000_Adam.png')
plt.close()



###############################################################
# plot the accuracies
##################################################

plt.plot(H.history['val_accuracy'], label='validation')
plt.plot(H.history['accuracy'], label='train')
plt.ylabel("$accuracy$")
plt.xlabel("$epoch$")
plt.legend()
plt.savefig('accuracy_withlog_70000_Adam.png')
plt.close()


#################################################
#Plot Power spectrum
################################### 

def pspec(k,field):
    
    pspec = (np.abs(field['c'])**2 + np.abs(field['c'])**2)/2

    n_count, bins = np.histogram(k.flatten(),bins=2*N_filter)

    n, bins = np.histogram(k.flatten(), bins=2*N_filter, weights=pspec.flatten())

    return bins, n/n_count

tau_pred = array_of_tf_components(preds)
tau_true = array_of_tf_components(y_test)


fig, ax = plt.subplots(1,3,figsize=(10*3,8))

titles = [r'$\tau_{xy}$',r'$\tau_{xx}$',r'$\tau_{yy}$',]

l = 0
for i in [0,1]:
    for j in [1,0]:
        if l == 3:
            break
        for axis in ['top','bottom','left','right']:
            ax[l].spines[axis].set_linewidth(3)
        ax[l].tick_params(direction='out',width=2,length=5,labelsize = 20)

        pred = domain.new_field(name='pred')
        true = domain.new_field(name='true')

        pred['g'] = tau_pred[i,j][0]
        true['g'] = tau_true[i,j][0]

        pred_bins, pred_ps = pspec(k,pred)
        true_bins, true_ps = pspec(k,true)

        ax[l].plot(pred_bins[:-1],pred_ps,label=pred.name)
        ax[l].plot(true_bins[:-1],true_ps,label=true.name)
        ax[l].set_title(titles[l],fontsize=26)

        ax[l].set_ylabel("$P(k)$",fontsize=20)
        ax[l].set_xlabel("$k$",fontsize=20)
        ax[l].set_yscale("log")
        ax[l].set_xscale("log")
        plt.legend()

        l += 1
plt.savefig('subgrid_ps_hist_withlog_70000_Adam.png',bbox_inches='tight')


fig, ax = plt.subplots(1,3,figsize=(10*3,8))

titles = [r'$\tau_{xy}$',r'$\tau_{xx}$',r'$\tau_{yy}$',]

l = 0
for i in [0,1,2]:
    if l == 3:
        break
    for axis in ['top','bottom','left','right']:
        ax[l].spines[axis].set_linewidth(3)
    ax[l].tick_params(direction='out',width=2,length=5,labelsize = 20)

   
    
    PS_label, edges = power2D(y_test[0,:,:,i])
    PS_pred, edges = power2D(preds[0,:,:,i])
    
    
    xplot = edges[:-1]
    PS_l = PS_label/N_filter**4
    PS_t = PS_pred/N_filter**4

    ax[l].plot(xplot,PS_l,label='true')
    ax[l].plot(xplot,PS_t,label='pred')
    ax[l].set_title(titles[l],fontsize=26)

    ax[l].set_ylabel("$PS(k)$",fontsize=20)
    ax[l].set_xlabel("$k$",fontsize=20)
    ax[l].set_yscale("log")
    ax[l].set_xscale("log")
    plt.legend()

    l += 1
plt.savefig('power_spectrum.png',bbox_inches='tight')


