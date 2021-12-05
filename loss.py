
import torch
import torch.nn as nn

def l2_loss(x,x_hat):
  return torch.mean((x_hat - x) ** 2)

def l1_loss(x,x_hat):
  return torch.mean(torch.abs(x_hat - x))

def shrinkage_loss(x,x_hat):
  a = 2. 
  c = 1.
  intensity =  torch.abs(x_hat - x)
  scale = 1./(1. + torch.exp( a * ( c - intensity ) ) ) 
  return torch.mean(torch.sum( intensity*scale, dim=(2,3)) )

def get_d_loss(D_real_logits,D_real,D_fake_logits,D_fake):
  BCEWithLogits = nn.BCEWithLogitsLoss()
  return 0.5*BCEWithLogits(D_real_logits, torch.ones_like(D_real)) + 0.5*BCEWithLogits(D_fake_logits, torch.zeros_like(D_fake))

def gradient_loss_by_axis(image,axis):
  b,c,h,w = image.shape
  if axis == "vertical" :
    return torch.cat((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:]) , torch.zeros((b,c,1,w)).float().to(device))   , dim=(2))
  elif axis == "horizontal" :
    return torch.cat((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1]) , torch.zeros((b,c,h,1)).float().to(device))   , dim=(3))

def gradient_diff_loss( x , x_hat  ):
  x_ver = gradient_loss_by_axis( x, "vertical")
  x_hor = gradient_loss_by_axis( x, "horizontal")
  x_hat_ver = gradient_loss_by_axis( x_hat, "vertical")
  x_hat_hor = gradient_loss_by_axis( x_hat, "horizontal")
  return  torch.mean( torch.abs(x_ver - x_hat_ver) + torch.abs(x_hor - x_hat_hor)  )

def get_ampl(motion,NUM_TEMPORAL_FRAME):
  u = motion.reshape(-1,NUM_TEMPORAL_FRAME,2,128,192)[:,:,0,:,:]
  v = motion.reshape(-1,NUM_TEMPORAL_FRAME,2,128,192)[:,:,1,:,:]
  motion_ampl = torch.mean( torch.sqrt(u**2+v**2) , dim = 1 )
  max_ampl = torch.max(  motion_ampl.reshape(-1,128*192) , 1  )
  motion_ampl = motion_ampl/max_ampl[0].reshape(-1,1,1)
  return motion_ampl.reshape( -1 , 1, 128, 192 )

def motion_att_gradient_diff_loss( x , x_hat , motion ):
  x_ver = gradient_loss_by_axis( x, "vertical")
  x_hor = gradient_loss_by_axis( x, "horizontal")
  x_hat_ver = gradient_loss_by_axis( x_hat, "vertical")
  x_hat_hor = gradient_loss_by_axis( x_hat, "horizontal")
  motion_ampl = get_ampl(motion)
  return  torch.mean( ( torch.abs(x_ver - x_hat_ver) + torch.abs(x_hor - x_hat_hor) ) * motion_ampl  )

def motion_att_l2_loss(x,x_hat,motion):
  motion_ampl = get_ampl(motion)
  return torch.mean((x_hat - x) ** 2 * motion_ampl)

def l1_flow_loss(x,x_hat):
  ang_diff = torch.abs(x_hat[:,0,:,:] - x[:,0,:,:])
  ang_diff = torch.clip( ang_diff, 0., 1. )
  ang_diff = torch.minimum( ang_diff , 1. - ang_diff)
  #ang_diff = torch.min( ang_diff , dim=(1))
  amp_diff = torch.abs(x_hat[:,1,:,:] - x[:,1,:,:])
  loss = torch.mean( amp_diff ) + torch.mean( ang_diff )

  return loss