import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import glob
import torch
import wandb
from scipy.io import loadmat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_segments(seq):
    def find_ends(seq):
        tmp = np.insert(seq, 0, -10)
        diff = tmp[1:] - tmp[:-1]
        peaks = np.where(diff != 1)[0]
        #
        ret = np.empty((len(peaks), 2), dtype=int)
        for i in range(len(ret)):
            ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
        return ret
    #
    ends = find_ends(seq)
    return np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)

def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i+1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret

def load_raw_groundtruth(data_type,groundtruth_dir=None):
    if data_type == "ped2":
        return [[61, 180], [95, 180], [1, 146], [31, 180], [1, 129], [1, 159],
                  [46, 180], [1, 180], [1, 120], [1, 150], [1, 180], [88, 180]]
    if data_type == "belleview":
        frame_label = []
        for count in range(300,300+len(glob.glob(groundtruth_dir+"/GT/*"))-2):
            label_png = groundtruth_dir + "/GT/gt-" + str( count ).zfill(5) + ".png"
            label_png = cv2.imread(label_png,0)
            if label_png.sum() == 0 :
                frame_label.append(0)
            else:
                frame_label.append(1)
        return frame_label
    if data_type == "train":
        frame_label = []
        for count in range(13840,800+len(glob.glob(groundtruth_dir+"/GT/*"))-1):
            label_png = groundtruth_dir + "/GT/gt-" + str( count ).zfill(5) + ".png"
            label_png = cv2.imread(label_png,0)
            if label_png.sum() == 0 :
                frame_label.append(0)
            else:
                frame_label.append(1)
        return frame_label
    if data_type == "avenue":
      return load_ground_truth_Avenue(data_type,21)
    if data_type == "idiap":
      return np.load(groundtruth_dir+"/frame_labels.npy")[0]
    

def get_index_sample_and_label(images,raw_ground_truth,data_type,INDEX_STEP,NUM_TEMPORAL_FRAME):
  sample_video_frame_index = []
  sample_count = 0
  index_list = []
  labels_temp = []
  label_count = 0
  video_index_label = []
  for i in range(len(images)):
    if data_type == "ped2" :
        one_clip_labels = raw_ground_truth[i]
    if data_type == "avenue":
        one_clip_labels = raw_ground_truth[i]

    for j in range(len( images[i] )):
      if j + INDEX_STEP*NUM_TEMPORAL_FRAME <=  len(images[i]) - 1  :
        video_index = [i]
        sample_video_frame_index.append( video_index + [ j + k*INDEX_STEP for k in range(NUM_TEMPORAL_FRAME)  ]    )
        index_list.append(sample_count)
        sample_count += 1
        video_index_label.append(i)
        if data_type == "ped2" :
            if j < one_clip_labels[1] and j >= one_clip_labels[0] - 1  :
                labels_temp.append(1)
            else:
                labels_temp.append(0)
        if data_type == "belleview" or data_type == "train":
            labels_temp.append(raw_ground_truth[label_count])
        if data_type == "idiap":
          labels_temp.append(raw_ground_truth[label_count])
        if data_type == "avenue":
          if j < one_clip_labels[1] and j >= one_clip_labels[0] - 1  :
            labels_temp.append(1)
          else:
            labels_temp.append(0)

      label_count += 1

  return sample_video_frame_index , index_list , labels_temp, np.array(video_index_label)
  

def get_index_sample(images,INDEX_STEP,NUM_TEMPORAL_FRAME,TRAIN_VAL_SPLIT=0):
  sample_video_frame_index = []
  train_append = True
  sample_count = 0
  train_index = []
  val_index = []
  for i in range(len(images)):

    if random.random() <= TRAIN_VAL_SPLIT:
      train_append = False
    else:
       train_append = True
      
    for j in range(len( images[i] )):
      if j + INDEX_STEP*NUM_TEMPORAL_FRAME <=  len(images[i]) - 1  :
        video_index = [i]
        sample_video_frame_index.append( video_index + [ j + k*INDEX_STEP for k in range(NUM_TEMPORAL_FRAME)  ]    )
        if train_append:
          train_index.append(sample_count)
        else:
          val_index.append(sample_count)
        sample_count += 1
        
  return sample_video_frame_index , train_index , val_index


def extend_mag_channel(datum):
    mag, _ = cv2.cartToPolar(datum[0 , :, :], datum[ 1, :, :])
    return np.concatenate((datum, np.expand_dims(mag, axis=0)), axis=0)

class Loss_log():
  def __init__(self, loss_name_list,checkpoint_save_path): 
    self.loss_name_list = loss_name_list
    self.inter_loss = {}
    self.epoch_loss = {}
    self.checkpoint_save_path = checkpoint_save_path
    for name in self.loss_name_list:
      self.inter_loss[name] = []
      self.epoch_loss[name] = []

  def add_inter_loss( self, loss_ ):
    for name in loss_:
      self.inter_loss[name] = loss_[name]
  
  def end_epoch(self, plot ):
    # mean
    for name in self.loss_name_list:
      self.epoch_loss[name].append(np.mean( self.inter_loss[name] ))
    # clear
    for name in self.loss_name_list:
      self.inter_loss[name] = []
    if plot:
      epoch = list(range(len(self.epoch_loss[self.loss_name_list[0]])))
      fig , ax=plt.subplots( figsize=(20,5))
      for name in self.loss_name_list:
        ax.plot(epoch,self.epoch_loss[name],label=name)
      legend = ax.legend(loc='upper left')
      # try:
      #   plt.show()
      # except:
      fig.savefig(self.checkpoint_save_path+"/epoch_"+str(len(epoch))+"_train_chart.png")
        
    return int(len(epoch))

def flow_to_RGB(flow,bound=10):
  hsv = np.zeros((flow.shape[0],flow.shape[1],3))
  hsv[..., 2] = 255
  #flow = flow/127.5 - 1. 
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 1] = mag / (bound) * 255 
  bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
  return bgr


def visualizing(real_image,output_appe,real_flow,output_opt,checkpoint_save_path,epoch,tran_or_test="train"):
    real_img = (   np.array(  np.clip( real_image.detach().cpu() , -1.0 , 1.0 )   ) * 127 + 127.   ).astype(np.uint8)
    real_img = real_img[0][0,:,:]

    gen_img = (   np.array(  np.clip( output_appe.detach().cpu() , -1.0 , 1.0 )   ) * 127  + 127.  ).astype(np.uint8)
    gen_img = gen_img[0][0,:,:]

    aver_img =   np.mean( np.array(  np.clip( real_image.detach().cpu() , -1.0 , 1.0 ) )[0]  , axis = 0 ) 
    aver_img = (aver_img* 127. + 127.).astype(np.uint8)

    real_flow = np.array(  real_flow.detach().cpu() ) 
    real_flow = np.transpose(  real_flow[0,:2], ( 1,2, 0) )

    gen_flow = np.array(  output_opt.detach().cpu()  ) 
    gen_flow = np.transpose(  gen_flow[0,:2], ( 1,2, 0) )

    fig, axs = plt.subplots(1, 5, figsize=(30, 5))
    for ax, interp, img_ in zip(axs, ['real img', 'gen img', 'real flow', 'gen flow', 'averg' ], [real_img,gen_img,real_flow,gen_flow,aver_img  ]):
      if interp in ['real img', 'gen img', 'averg']:
        ax.imshow( img_ , "gray")
      else:
        ax.imshow( flow_to_RGB(img_.astype( np.float32) ))
      ax.set_title(interp.capitalize())
      ax.grid(True)
    # try:
    #   plt.show()
    # except:
    fig.savefig(checkpoint_save_path+"/"+tran_or_test+"_epoch_"+str(epoch)+"_train_app_flow.png")

def test_flow_vil(output_opt,real_flow,checkpoint_save_path,epoch):
  def scale_range(img):
    for i in range(img.shape[-1]):
        img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
    return img
  output_opt = output_opt[0].detach().cpu().numpy().transpose((1,2,0))
  real_flow = real_flow[0].detach().cpu().numpy().transpose((1,2,0))
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].imshow(scale_range(real_flow))
  axs[1].imshow(scale_range(output_opt))
  # try:
  #   plt.show()
  # except:
  fig.savefig(checkpoint_save_path+"/epoch_"+str(epoch)+"_train_flow.png")

def max_norm_score_by_clip(combine_max_score,video_index_label):
  max_video_combine_max_score = combine_max_score.copy()
  for video_index in np.unique(video_index_label):
    max_video_score = combine_max_score[np.where(video_index_label==video_index)[0]].max()
    min_video_score = combine_max_score[np.where(video_index_label==video_index)[0]].min()
    max_video_combine_max_score[np.where(video_index_label==video_index)[0]] = combine_max_score[np.where(video_index_label==video_index)[0]]/max_video_score
  return max_video_combine_max_score