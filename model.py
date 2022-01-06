import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from torch import optim
import torch.nn.functional as F
from barbar import Bar
import torch.nn as nn
import random 
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from memory import *


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def weights_init_memAE_git(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)  
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

PADDING_MODE = "reflect"

def conv2d( in_channel , out_channel, filter_size=(3,3),  stride=1 , padding = 0  , bias = True):
    return  nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = filter_size, stride = stride, bias=bias , padding = padding , padding_mode = PADDING_MODE )

def conv_transpose( in_channel,output_shape, filter_size=3,  stride=1 , padding = 0):
  return nn.ConvTranspose2d(in_channels=in_channel,out_channels = output_shape, kernel_size = filter_size, stride = stride,  bias=True , padding = padding)

class conv2d_Inception(nn.Module):
    def __init__(self, out_channel , max_filter_size ,  in_channel = 1 ):
        super(conv2d_Inception, self).__init__()
        
        n_branch = (max_filter_size+1) // 2
        nf_branch = out_channel // n_branch
   
        if n_branch >= 1:
          # 1x1
          self.s1_11_conv = conv2d( in_channel , nf_branch, filter_size=(1, 1))
       
        if n_branch >= 2:  
           # 3x3
          self.s3_11_conv = conv2d( in_channel,  nf_branch, filter_size=(1, 1))
          self.s3_1n_conv = conv2d( nf_branch,  nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s3_n1_conv = conv2d( nf_branch,  nf_branch, filter_size=(3, 1) , padding = (1,0)) 
        
        if n_branch >= 3:
           # 5x5
          self.s5_11_conv = conv2d( in_channel, nf_branch, filter_size=(1, 1))
          self.s5_1n_conv_1 = conv2d( nf_branch, nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s5_n1_conv_1 = conv2d( nf_branch, nf_branch, filter_size=(3, 1) , padding = (1,0))
          self.s5_1n_conv_2 = conv2d( nf_branch, nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s5_n1_conv_2 = conv2d( nf_branch, nf_branch, filter_size=(3, 1) , padding = (1,0))
        
        if n_branch >= 4:
          # 7x7
          self.s7_11_conv = conv2d( in_channel, nf_branch, filter_size=(1, 1))
          self.s7_1n_conv_1 = conv2d( nf_branch, nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s7_n1_conv_1 = conv2d( nf_branch, nf_branch, filter_size=(3, 1) , padding = (1,0))
          self.s7_1n_conv_2 = conv2d( nf_branch, nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s7_n1_conv_2 = conv2d( nf_branch, nf_branch, filter_size=(3, 1) , padding = (1,0))
          self.s7_1n_conv_3 = conv2d( nf_branch, nf_branch, filter_size=(1, 3) , padding = (0,1))
          self.s7_n1_conv_3 = conv2d( nf_branch, nf_branch, filter_size=(3, 1) , padding = (1,0))
       
        self.n_branch = n_branch
        
    def forward(self, x):
         # 1x1
        s1_11 = self.s1_11_conv(x)
        #print(s1_11.shape)
        if self.n_branch == 1:
            return s1_11
            
        # 3x3
        s3_11 = self.s3_11_conv(x)
        s3_1n = self.s3_1n_conv(s3_11)
        s3_n1 = self.s3_n1_conv(s3_1n)
        #print(s3_n1.shape)
        if self.n_branch == 2:
          return  torch.cat( ( s1_11 , s3_n1 ) , dim=(1))

        # 5x5
        s5_11 = self.s5_11_conv(x)
        s5_1n = self.s5_1n_conv_1(s5_11)
        s5_n1 = self.s5_n1_conv_1(s5_1n)
        s5_1n = self.s5_1n_conv_2(s5_n1)
        s5_n1 = self.s5_n1_conv_2(s5_1n)
        #print(s5_n1.shape)
        if self.n_branch == 3:
          return  torch.cat( ( s1_11 , s3_n1 ,s5_n1 ) , dim=(1))

        # 7x7
        s7_11 = self.s7_11_conv(x)
        s7_1n = self.s7_1n_conv_1(s7_11)
        s7_n1 = self.s7_n1_conv_1(s7_1n)
        s7_1n = self.s7_1n_conv_2(s7_n1)
        s7_n1 = self.s7_n1_conv_2(s7_1n)
        s7_1n = self.s7_1n_conv_3(s7_n1)
        s7_n1 = self.s7_n1_conv_3(s7_1n)
        #print(s7_n1.shape)
        return torch.cat( ( s1_11 , s3_n1 ,s5_n1 , s7_n1 ) , dim=(1))

class G_conv_bn_relu(nn.Module):
    def __init__(self, in_channel , out_channel, filter_size, stride=2, bn=True , padding = (2,2) , custom_padding = False ):
      super(G_conv_bn_relu, self).__init__()
      self.conv = conv2d(in_channel, out_channel, filter_size=filter_size, stride=stride , padding=padding)
      if bn:
          self.bn = nn.BatchNorm2d( num_features = out_channel )
          self.bn_run = True
      else:
          self.bn_run = False
      self.custom_padding = custom_padding

    def forward(self, x):
      x = self.conv(x)
      if self.bn_run:
        x = self.bn(x)
      x = F.leaky_relu(x)
      if self.custom_padding:
        x = x[:,:,:-1,:-1]
      return x 

class G_deconv_bn_dr_relu_concat(nn.Module):
    def __init__(self, in_channel, out_shape, filter_size, p_keep_drop , padding = 0 , batchnorm = True ):
      super(G_deconv_bn_dr_relu_concat, self).__init__()
      self.deconv1 = conv_transpose(in_channel, out_shape, filter_size=filter_size , stride = 2 , padding = padding)
      self.bn = nn.BatchNorm2d( num_features = out_shape )
      self.p_keep_drop = p_keep_drop
      self.batchnorm = batchnorm

    def forward(self, x , skip_input=None):
      x = self.deconv1(x)
      if self.batchnorm:
        x = self.bn(x)
      x = nn.Dropout(1-self.p_keep_drop)(x)
      x = F.relu(x)
      if skip_input is not None:
          x = torch.cat( ( x , skip_input ) , dim=(1))
      return x 


class Generator(nn.Module):
    def __init__(self, h, w, img_channel , flow_channel , keep_prob ,
     				image_mem_num , flow_mem_num , model_config_dict,
     				image_filters = 64, flow_filters = 64 ):
      super(Generator, self).__init__()
      filter_size = (4, 4) 
      flow_filter_size = (4, 4) 
      self.model_config_dict = model_config_dict
      '''Unet ENCODER for FRAME'''
      self.conv2d_Inception        = conv2d_Inception( image_filters, max_filter_size=7,in_channel=img_channel)
      self.G_conv_bn_relu_1        = G_conv_bn_relu( image_filters , image_filters, filter_size, stride=1,  bn=False  , padding = (2,2), custom_padding= True )
      self.G_conv_bn_relu_2        = G_conv_bn_relu( image_filters , image_filters*2, filter_size, stride=2,  bn=True , padding = (1,1) )
      self.G_conv_bn_relu_3        = G_conv_bn_relu( image_filters*2 , image_filters*4, filter_size, stride=2,  bn=True  , padding = (1,1))
      self.G_conv_bn_relu_4        = G_conv_bn_relu( image_filters*4 , image_filters*8, filter_size, stride=2,  bn=True  , padding = (1,1))
      self.G_conv_bn_relu_flow_5   = G_conv_bn_relu( image_filters*8 , image_filters*8, filter_size, stride=2,  bn=True  , padding = (1,1))
      self.G_conv_bn_relu_image_5  = G_conv_bn_relu( image_filters*8 , image_filters*8, filter_size, stride=2,  bn=True  , padding = (1,1))
      #################################### FRAME DECODER CONFIG ####################################
      '''Unet DECODER for FRAME'''
      if self.model_config_dict["use_im_memory"]:
        deconv_frame1_INPUTSIZE = image_filters*8 + image_filters*8 
      else:
        deconv_frame1_INPUTSIZE = image_filters*8 
      self.G_deconv_bn_dr_relu_concat_frame1 =  G_deconv_bn_dr_relu_concat( deconv_frame1_INPUTSIZE , image_filters*8, filter_size, keep_prob , padding = (1,1) )
      #################################### FRAME DECODER CONFIG ####################################
      if self.model_config_dict["use_im_skipcon_1"]:
        deconv_frame2_INPUTSIZE = image_filters*8 + image_filters*8 
      else:
        deconv_frame2_INPUTSIZE = image_filters*8 
      self.G_deconv_bn_dr_relu_concat_frame2 =  G_deconv_bn_dr_relu_concat( deconv_frame2_INPUTSIZE , image_filters*4, filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_im_skipcon_2"]:
        deconv_frame3_INPUTSIZE = image_filters*4 + image_filters*4
      else:
        deconv_frame3_INPUTSIZE = image_filters*4 
      self.G_deconv_bn_dr_relu_concat_frame3 =  G_deconv_bn_dr_relu_concat( deconv_frame3_INPUTSIZE , image_filters*2, filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_im_skipcon_3"]:
        deconv_frame2_INPUTSIZE = image_filters*2 + image_filters*2
      else:
        deconv_frame2_INPUTSIZE = image_filters*2
      self.G_deconv_bn_dr_relu_concat_frame4 =  G_deconv_bn_dr_relu_concat( deconv_frame2_INPUTSIZE , image_filters  , filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_im_skipcon_4"]:
        conv2d_frame1_INPUTSIZE = image_filters + image_filters
      else:
        conv2d_frame1_INPUTSIZE = image_filters
      self.conv2d_frame1 = conv2d( conv2d_frame1_INPUTSIZE , img_channel , filter_size=(3,3), stride=1 , padding = 1 , bias = True)
      #################################### FLOW DECODER CONFIG ####################################
      '''Unet DECODER for OPTICAL FLOW'''
      self.G_deconv_bn_dr_relu_concat_flow1 =  G_deconv_bn_dr_relu_concat( image_filters*8 , flow_filters*8  , flow_filter_size, keep_prob , padding = (1,1))
      if self.model_config_dict["use_flow_skipcon_1"]:
        deconv_flow2_INPUTSIZE = flow_filters*8 + flow_filters*8 
      else:
        deconv_flow2_INPUTSIZE = flow_filters*8 
      self.G_deconv_bn_dr_relu_concat_flow2 =  G_deconv_bn_dr_relu_concat( deconv_flow2_INPUTSIZE, flow_filters*4  , flow_filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_flow_skipcon_2"]:
        deconv_flow3_INPUTSIZE = flow_filters*4 + flow_filters*4
      else:
        deconv_flow3_INPUTSIZE = flow_filters*4
      self.G_deconv_bn_dr_relu_concat_flow3 =  G_deconv_bn_dr_relu_concat( deconv_flow3_INPUTSIZE , flow_filters*2  , flow_filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_flow_skipcon_3"]:
        deconv_flow4_INPUTSIZE = flow_filters*2 + flow_filters*2
      else:
        deconv_flow4_INPUTSIZE = flow_filters*2
      self.G_deconv_bn_dr_relu_concat_flow4 =  G_deconv_bn_dr_relu_concat( deconv_flow4_INPUTSIZE, flow_filters    , flow_filter_size, keep_prob, padding = 1 )
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_flow_skipcon_4"]:
        conv2d_flow1_INPUTSIZE = flow_filters + flow_filters
      else:
        conv2d_flow1_INPUTSIZE = flow_filters
      self.conv2d_flow1 = conv2d( conv2d_flow1_INPUTSIZE , flow_channel , filter_size=(3,3), stride=1 , padding = 1 , bias = True)      
      #################################### MEMORY CONFIG ####################################
      if self.model_config_dict["use_im_memory"]:
        if self.model_config_dict["use_im_local_mem"]:
          self.image_memmodule = LOCAL_MemModule(mem_dim=image_mem_num, fea_dim=512, fea_num=8*12)
        else:
          self.image_memmodule = GLOBAL_MemModule(mem_dim=flow_mem_num, fea_dim=512)
      #---------------------------------------------------------------------------------------------
      if self.model_config_dict["use_flow_memory"]:
        if self.model_config_dict["use_flow_local_mem"]:
          self.flow_memmodule = LOCAL_MemModule(mem_dim=flow_mem_num, fea_dim=512, fea_num=8*12)
        else:
          self.flow_memmodule = GLOBAL_MemModule(mem_dim=flow_mem_num, fea_dim=512)

    def image_encode(self, image):
      img_h0 = self.conv2d_Inception.forward(image)
      img_h1 = self.G_conv_bn_relu_1.forward(img_h0)
      img_h2 = self.G_conv_bn_relu_2.forward(img_h1)
      img_h3 = self.G_conv_bn_relu_3.forward(img_h2)
      img_h4 = self.G_conv_bn_relu_4.forward(img_h3)
      img_h5 = self.G_conv_bn_relu_image_5.forward(img_h4)
      flow_h5 = self.G_conv_bn_relu_flow_5.forward(img_h4)

      return img_h1 , img_h2 , img_h3 , img_h4 , img_h5 , flow_h5
    
    def appearance_decode(self, img_h1, img_h2 , img_h3 , img_h4 , h5):
      ### FRAME DECODER
      ############################################################
      if self.model_config_dict["use_im_skipcon_1"]:
        h4fr = self.G_deconv_bn_dr_relu_concat_frame1.forward(h5,img_h4)
      else:
        h4fr = self.G_deconv_bn_dr_relu_concat_frame1.forward(h5)
      ############################################################
      if self.model_config_dict["use_im_skipcon_2"]:
        h3fr = self.G_deconv_bn_dr_relu_concat_frame2.forward(h4fr,img_h3)
      else:
        h3fr = self.G_deconv_bn_dr_relu_concat_frame2.forward(h4fr)
      ############################################################
      if self.model_config_dict["use_im_skipcon_3"]:
        h2fr = self.G_deconv_bn_dr_relu_concat_frame3.forward(h3fr,img_h2)
      else:
        h2fr = self.G_deconv_bn_dr_relu_concat_frame3.forward(h3fr)
      ############################################################
      if self.model_config_dict["use_im_skipcon_4"]:
        h1fr = self.G_deconv_bn_dr_relu_concat_frame4.forward(h2fr,img_h1)
      else:  
        h1fr = self.G_deconv_bn_dr_relu_concat_frame4.forward(h2fr)
      ############################################################
      out_frame = self.conv2d_frame1(h1fr)
      return out_frame 

    def flow_decode(self , img_h1, img_h2 , img_h3 , img_h4 , img_h5 ):
      ### FLOW DECODER
      ############################################################
      if self.model_config_dict["use_flow_skipcon_1"]:
        h4fl = self.G_deconv_bn_dr_relu_concat_flow1.forward(img_h5,img_h4)
      else:
        h4fl = self.G_deconv_bn_dr_relu_concat_flow1.forward(img_h5)
      ############################################################
      if self.model_config_dict["use_flow_skipcon_2"]:
        h3fl = self.G_deconv_bn_dr_relu_concat_flow2.forward(h4fl,img_h3)
      else:
        h3fl = self.G_deconv_bn_dr_relu_concat_flow2.forward(h4fl)
      ############################################################
      if self.model_config_dict["use_flow_skipcon_3"]:
        h2fl = self.G_deconv_bn_dr_relu_concat_flow3.forward(h3fl,img_h2)
      else:
        h2fl = self.G_deconv_bn_dr_relu_concat_flow3.forward(h3fl)
      ############################################################
      if self.model_config_dict["use_flow_skipcon_4"]:
        h1fl = self.G_deconv_bn_dr_relu_concat_flow4.forward(h2fl,img_h1)
      else:
        h1fl = self.G_deconv_bn_dr_relu_concat_flow4.forward(h2fl)
      ############################################################
      out_flow = self.conv2d_flow1(h1fl)
      return out_flow

    def forward(self, image, flow ):
      ### ENCODE
      img_h1 , img_h2 , img_h3 , img_h4 , img_h5 , flow_h5 = self.image_encode(image)
      ### MEM
      if self.model_config_dict["use_im_memory"]:
        image_fea_h5 = self.image_memmodule(img_h5)
        image_fea_h5 = torch.cat( ( image_fea_h5 , img_h5 ) , dim = 1 )
      else:
        image_fea_h5 = flow_h5

      if self.model_config_dict["use_flow_memory"]:
        flow_fea_h5 = self.flow_memmodule(flow_h5)
      else:
        flow_fea_h5 = flow_h5

      ### DECODER
      out_flow = self.flow_decode(img_h1 ,img_h2 , img_h3 , img_h4 , flow_fea_h5)
      out_frame  = self.appearance_decode( img_h1 ,img_h2 , img_h3 , img_h4 , image_fea_h5)
      return out_flow, out_frame 


class D_conv_bn_active(nn.Module):
    def __init__(self, in_channel ,  out_channel, filter_size, stride=2, bn=True, active=True  , padding = (0,0) ):
      super(D_conv_bn_active, self).__init__()
      self.conv = conv2d(in_channel, out_channel, filter_size=filter_size, stride=stride , padding=padding)
      if bn:
          self.bn = nn.BatchNorm2d( num_features = out_channel )
          self.bn_run = True
      else:
          self.bn_run = False
      self.active = active

    def forward(self, x):
      x = self.conv(x)
      if self.bn_run:
        x = self.bn(x)
      if self.active:
        x = F.leaky_relu(x)
      return x 


class Discriminator(nn.Module):
    def __init__(self, channel , filters = 64):
      super(Discriminator, self).__init__()
      
      filter_size = (4, 4)

      self.D_conv_bn_active_1 = D_conv_bn_active( channel   , filters  , filter_size, stride=2,  bn=False , padding = (1,1))
      self.D_conv_bn_active_2 = D_conv_bn_active( filters   , filters*2, filter_size, stride=2,  bn=True  , padding = (1,1))
      self.D_conv_bn_active_3 = D_conv_bn_active( filters*2 , filters*4, filter_size, stride=2,  bn=True  , padding = (1,1))
      self.D_conv_bn_active_4 = D_conv_bn_active( filters*4 , filters*8, filter_size, stride=2,  bn=True  , active=False , padding = (1,1))

      #self.conv2d = conv2d( filters * 8 , 1 , filter_size=(1,1), stride=1 , padding = 0 , bias = True)

    def forward(self, x , flow_hat ):
      #print("DIS")
      h0 = torch.cat((x,flow_hat) , dim=(1))
      #print(h0.shape)
      h1 = self.D_conv_bn_active_1.forward(h0)
      #print(h1.shape)
      h2 = self.D_conv_bn_active_2.forward(h1)
      #print(h2.shape)
      h3 = self.D_conv_bn_active_3.forward(h2)
      #print(h3.shape)
      h4 = self.D_conv_bn_active_4.forward(h3)
      #print(h4.shape)
      return torch.sigmoid(h4) , h4