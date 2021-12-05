from loss import *
from utils import *
from model import *
from dataset import *
from metric import *

import tqdm
import argparse
import wandb
import time

parser = argparse.ArgumentParser(description="flownet anomaly detection")
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs for training')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=192, help='width of input images')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--im_msize', type=int, default=500, help='number of the memory items')
parser.add_argument('--flow_msize', type=int, default=500, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--data_type', type=str, default='belleview', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--groundtruth_path', type=str, default='', help='directory of ground truth ')
parser.add_argument('--eval_method', type=str, default='all', help=' all , normal , custom ')
parser.add_argument('--mag_quantile', type=float, default=0.99, help=' mag cut for custom method ')
parser.add_argument('--min_flow_weight', type=float, default=0.1, help=' min flow weight in custom method ')
parser.add_argument('--wandb_log', type=bool, default=False, help='Wandb ML ops monitor')
parser.add_argument('--wandb_run_name', type=str, default="test", help='Wandb ML ops monitor wandb_run_name')


args = parser.parse_args()

if args.wandb_log:
  wandb.init(project=args.data_type, name = args.wandb_run_name, entity="khang-9966")

  wandb.config.update ( {
    "mag_quantile": args.mag_quantile,
    "min_flow_weight": args.min_flow_weight,
  } )

NUM_TEMPORAL_FRAME = 2
INDEX_STEP = 1
TRAIN_VAL_SPLIT = 0.0
BATCH_SIZE = args.batch_size


DATASET_TYPE = args.data_type

checkpoint_save_path = args.exp_dir
model_name = "original"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p_keep = 1.0
test_generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize)
test_generator = test_generator.to(device)

test_generator.load_state_dict(torch.load( checkpoint_save_path+"/"+model_name+"_final_gen_"+str(args.epochs) +".pt"))

test_generator.eval()

raw_ground_truth = load_raw_groundtruth(data_type=DATASET_TYPE,groundtruth_dir=args.groundtruth_path)

print("Test len: ",len(raw_ground_truth))

test_images , test_flows = load_raw_data(data_type=args.data_type,train_or_test="test",data_dir=args.dataset_path)

test_sample_video_frame_index , test_index , labels_temp = get_index_sample_and_label(test_images,raw_ground_truth,DATASET_TYPE,INDEX_STEP,NUM_TEMPORAL_FRAME)

test_image_list = []
for index in tqdm.tqdm(test_sample_video_frame_index):
  video_index = index[0]
  frame_index = index[1:]
  test_image_list.append(  np.array( [ test_images[video_index][img_index] for img_index in frame_index  ] )   )

test_flow_list = []
for index in tqdm.tqdm(test_sample_video_frame_index):
  video_index = index[0]
  frame_index = index[1:]
  test_flow_list.append(   np.transpose( np.array( [ test_flows[video_index][img_index] for img_index in frame_index[1:]  ] ) , (0,3,1,2)   ).reshape(2,128,192) )

test_data =  loader(test_index,test_image_list,test_flow_list)


test_loader = DataLoader(test_data, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers_test)

if args.eval_method == "all" or args.eval_method == "normal":
  score_frame_ = []
  for real_image,  real_flow  in Bar(test_loader):
    with torch.no_grad():

      plh_frame_true = real_image.float().to(device)
      plh_flow_true = real_flow.float().to(device)

      output_opt, output_appe = test_generator(plh_frame_true,plh_flow_true)

      output_appe = output_appe.detach().cpu().numpy().transpose((0,2,3,1))*0.5 + 0.5
      real_image = real_image.detach().cpu().numpy().transpose((0,2,3,1))*0.5 + 0.5
      real_flow = real_flow.detach().cpu().numpy().transpose((0,2,3,1))
      output_opt = output_opt.detach().cpu().numpy().transpose((0,2,3,1))
      
      # print(output_opt.min(),output_opt.max(),real_flow.min(),real_flow.max())
      # print(output_appe.min(),output_appe.max(),real_image.min(),real_image.max())
      for i in range(len(real_flow)):
        score_frame_.append(calc_anomaly_score_one_frame(real_image[i].astype(np.float32), output_appe[i].astype(np.float32), real_flow[i].astype(np.float32), output_opt[i].astype(np.float32)))

  score_frame = np.array(score_frame_)

  score_frame = flip_scores(score_frame)

  training_errors = np.mean(score_frame, axis=0)

  w_img, w_flow = training_errors[0].flatten(), training_errors[1].flatten()

  print('weights: w_img =', str(w_img), '- vw_flow =', str(w_flow))

  sequence_n_frame = len(test_sample_video_frame_index)

  scores_appe,scores_flow,scores_comb,scores_angle,scores_mag, appe_auc, appe_prc = full_assess_AUC(score_frame, labels_temp, w_img, w_flow, sequence_n_frame, True,
                  True, None,
                  save_pr_appe_SSIM_epoch=True)
  if args.wandb_log:
    metric_list = ["PSNRX","PSNRinv","PSNR","SSIM","MSE","maxSE","std","MSE1c","maxSE1c","std1c"]
    for i in range(len(metric_list)):
       wandb.log({ "AppearanceAUC " + metric_list[i] : appe_auc[i]} )
    for i in range(len(appe_prc)):
       wandb.log({ "AppearanceAP " + metric_list[i] : appe_prc[i]} )

if args.eval_method == "all" or args.eval_method == "custom":
  STEP = 4
  KERNEL_SIZE = 16
  mean_kernel = nn.Conv2d(1, 1, KERNEL_SIZE, stride=4 , bias = False).cuda()
  mean_kernel.weight.data.fill_(1/KERNEL_SIZE**2)

  test_loader = DataLoader( test_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers= args.num_workers_test )

  diff_map_flow_list = []
  diff_map_appe_list = []
  mag_map_list = []

  for real_image,  real_flow  in Bar(test_loader):

    with torch.no_grad():
      plh_frame_true = real_image.float().to(device)
      plh_flow_true = real_flow.float().to(device)

      output_opt, output_appe = test_generator(plh_frame_true,plh_flow_true)

      output_appe = output_appe*0.5 + 0.5
      plh_frame_true = plh_frame_true*0.5 + 0.5

      diff_map_flow = (output_opt-plh_flow_true)**2
      diff_map_appe = (output_appe-plh_frame_true)**2
      
      diff_map_flow = diff_map_flow.sum(1,keepdim=True)
      diff_map_appe = diff_map_appe.sum(1,keepdim=True)
      mag_map = plh_flow_true[:,-1:,:,:]

      diff_map_flow = mean_kernel(diff_map_flow).squeeze().cpu().numpy().reshape(plh_frame_true.shape[0],-1)
      diff_map_appe = mean_kernel(diff_map_appe).squeeze().cpu().numpy().reshape(plh_frame_true.shape[0],-1)
      mag_map       = mean_kernel(mag_map).squeeze().cpu().numpy().reshape(plh_frame_true.shape[0],-1)

      diff_map_flow_list += list(diff_map_flow)
      diff_map_appe_list += list(diff_map_appe)
      mag_map_list += list(mag_map)
      #break

  diff_map_flow_list = np.array(diff_map_flow_list)
  diff_map_appe_list = np.array(diff_map_appe_list)
  mag_map_list = np.array(mag_map_list)

  norm_mag_map = score_norm(mag_map_list.reshape(-1), quantile= args.mag_quantile , output_min= args.min_flow_weight , output_max = 1 - args.min_flow_weight ).reshape( len(mag_map_list) ,-1)
  norm_diff_map_flow = score_norm(diff_map_flow_list.reshape(-1), quantile=None, output_min=0 , output_max = 1 , log= True, max_value=100 ).reshape( len(diff_map_flow_list) ,-1)
  norm_diff_map_appe = score_norm(diff_map_appe_list.reshape(-1), quantile=None, output_min=0 , output_max = 1 , log= True,  max_value=1 ).reshape( len(diff_map_appe_list) ,-1)
  combine_score = norm_diff_map_flow*norm_mag_map + norm_diff_map_appe*(1-norm_mag_map)
  combine_max_score = [ max(one_frame) for one_frame in combine_score]

  max_app_mean = [ max(one_frame) for one_frame in norm_diff_map_appe]
  max_flow_mean = [ max(one_frame) for one_frame in norm_diff_map_flow]

  print("Appearence AUC: ",roc_auc_score(labels_temp, max_app_mean))
  print("Appearence AP: ",average_precision_score(labels_temp, max_app_mean))

  print("Flow AUC: ",roc_auc_score(labels_temp, max_flow_mean))
  print("Flow AP: ",average_precision_score(labels_temp, max_flow_mean))

  combine_max_score = normalize_maxmin_scores(combine_max_score)

  print("Combine AUC: ",roc_auc_score(labels_temp, combine_max_score))
  print("Combine AP: ",average_precision_score(labels_temp, combine_max_score))

  if args.wandb_log:
    wandb.log({"CombineAUC": roc_auc_score(labels_temp, combine_max_score)})
    wandb.log({"CombineAP": average_precision_score(labels_temp, combine_max_score) })
    wandb.log({"AppearenceAUC": roc_auc_score(labels_temp, max_app_mean)})
    wandb.log({"AppearenceAP": average_precision_score(labels_temp, max_app_mean) })
    wandb.log({"FlowAUC": roc_auc_score(labels_temp, max_flow_mean)})
    wandb.log({"FlowAP": average_precision_score(labels_temp, max_flow_mean) })
