
import numpy as np
import cv2
import torch
import numpy as np
from torch.utils import data
from utils import *

def resize(datum, size):
    if len(datum.shape) == 2:
        return cv2.resize(datum.astype(float), tuple(size))
    elif len(datum.shape) == 3:
        return np.dstack([cv2.resize(datum[:, :, i].astype(float), tuple(size)) for i in range(datum.shape[-1])])
    else:
        print('unexpected data size', datum.shape)
        return None

def load_raw_data(data_type,train_or_test,data_dir):
    if data_type == "ped2":
        return load_raw_ped2(train_or_test,data_dir)
    if data_type == "belleview":
        return load_raw_belleview(train_or_test,data_dir)

def load_raw_ped2(train_or_test,data_dir="/content/content/flownet2pytorch/FlowNet2-pytorch/"):
    if train_or_test == "train":
        all_train_images = []
        all_train_flows = []
        for i in range(1,17):
            all_data = np.load(data_dir+"ped2/training/Train"+str(i).zfill(2)+"_full.npy")
            image_data = all_data[:,:,:,:3]
            train_images = np.array([resize( cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY) , (192, 128)) for image_ in image_data]).astype(np.float32)
            flow_data = all_data[:,:,:,3:]
            train_flows = np.array([resize(flow_datum, (192, 128)) for flow_datum in flow_data]).astype(np.float32)
            all_train_images.append(train_images)
            all_train_flows.append(train_flows)
        return all_train_images,all_train_flows
    elif train_or_test == "test":
        all_test_images = []
        all_test_flows = []
        for i in range(1,13):
            all_data = np.load(data_dir+"ped2/testing/Test"+str(i).zfill(2)+"_full.npy")
            image_data = all_data[1:,:,:,:3]
            test_images = np.array([resize(  cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)  , (192, 128)) for image_ in image_data]).astype(np.float32)
            flow_data = all_data[1:,:,:,3:]
            test_flows = np.array([resize(flow_datum, (192, 128)) for flow_datum in flow_data]).astype(np.float32)
            all_test_images.append(test_images)
            all_test_flows.append(test_flows)
        return all_test_images,all_test_flows

def load_raw_belleview(train_or_test,data_dir):
    all_data = np.load(data_dir+"/"+train_or_test+"/001_full.npy")
    image_data = all_data[:,:,:,:3]
    train_images = np.array([resize( cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY) , (192, 128)) for image_ in image_data]).astype(np.float32)
    flow_data = all_data[:,:,:,3:]
    train_flows = np.array([resize(flow_datum, (192, 128)) for flow_datum in flow_data]).astype(np.float32)
    return [train_images],[train_flows]

class loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, sample_index , images , flows  ) : #target, transform):
        self.sample_index = sample_index
        self.images = images
        self.flows = flows

    def __getitem__(self, index):
        """return transformed items."""
        sample_index = self.sample_index[index]
        images = self.images[sample_index]
        image_tensor = (torch.tensor(images)) / 255.0 / 0.5 - 1
        flows = self.flows[sample_index]
        flow_tensor = torch.tensor( extend_mag_channel(flows) )  # / 255.0 / 0.5 - 1
        return image_tensor , flow_tensor 

    def __len__(self):
        """number of samples."""
        return len(self.sample_index)