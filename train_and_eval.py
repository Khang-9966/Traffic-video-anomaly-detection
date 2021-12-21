from loss import *
from utils import *
from model import *
from dataset import *
from metric import *
import os
import tqdm
import argparse
import wandb
import time 

parser = argparse.ArgumentParser(description="flownet anomaly detection")
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs for training')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=192, help='width of input images')
parser.add_argument('--g_lr', type=float, default=0.0002, help='generator initial learning rate')
parser.add_argument('--d_lr', type=float, default=0.00002, help='discriminator initial learning rate')
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
parser.add_argument('--wandb_log', type=bool, default=False, help='Wandb ML ops monitor')
parser.add_argument('--wandb_run_name', type=str, default="test", help='Wandb ML ops monitor wandb_run_name')
parser.add_argument('--eval_method', type=str, default='all', help=' all , normal , custom ')
parser.add_argument('--mag_quantile', type=float, default=0.99, help=' mag cut for custom method ')
parser.add_argument('--min_flow_weight', type=float, default=0.1, help=' min flow weight in custom method ')
parser.add_argument('--train_dropout', type=float, default=0.3, help=' train drop out')
parser.add_argument('--weight_init', type=bool, default=True, help='normal distribution weight init') 
parser.add_argument('--finalscore_maxnorm_byclip', type=bool, default=True, help='normal distribution weight init') 
######################################## MODEL CONFIG ##################################################
parser.add_argument('--use_flow_local_mem', type=bool, default=True, help='use local mem or global mem for image decoder')
parser.add_argument('--use_im_local_mem', type=bool, default=True, help='use local mem or global mem for flow decoder')
parser.add_argument('--use_im_memory', type=bool, default=True, help='use mem for image decoder')
parser.add_argument('--use_flow_memory', type=bool, default=True, help='use mem for flow decoder')
parser.add_argument('--use_im_skipcon_1', type=bool, default=False, help='use skip con 1 for image decoder') 
parser.add_argument('--use_im_skipcon_2', type=bool, default=False, help='use skip con 2 for image decoder') 
parser.add_argument('--use_im_skipcon_3', type=bool, default=False, help='use skip con 3 for image decoder') 
parser.add_argument('--use_im_skipcon_4', type=bool, default=False, help='use skip con 4 for image decoder') 
parser.add_argument('--use_flow_skipcon_1', type=bool, default=True, help='use skip con 1 for flow decoder') 
parser.add_argument('--use_flow_skipcon_2', type=bool, default=True, help='use skip con 2 for flow decoder') 
parser.add_argument('--use_flow_skipcon_3', type=bool, default=True, help='use skip con 3 for flow decoder') 
parser.add_argument('--use_flow_skipcon_4', type=bool, default=True, help='use skip con 4 for flow decoder') 

args = parser.parse_args()

if args.wandb_log:
  wandb.init(project=args.data_type,  name = args.wandb_run_name, 
        entity="khang-9966",
        config = {
            "batchsize": args.batch_size,
            "epochs": args.epochs,
            "h": args.h,
            "w": args.w,
            "glr": args.g_lr,
            "dlr": args.d_lr,
            "fdim": args.fdim,
            "mdim": args.mdim,
            "im_msize": args.im_msize,
            "flow_msize": args.flow_msize,
            "data_type": args.data_type,
            "wandb_run_name": args.wandb_run_name,
            "exp_dir": args.exp_dir,
            "mag_quantile": args.mag_quantile,
            "min_flow_weight": args.min_flow_weight,
            "full_model_dir" : args.exp_dir +"/"+ args.data_type +"-"+ args.wandb_run_name +"/",
            "train_dropout" : args.train_dropout,
            "weight_init" : args.weight_init,
            "finalscore_maxnorm_byclip" : args.finalscore_maxnorm_byclip,
            "use_im_local_mem" : args.use_im_local_mem,
            "use_flow_local_mem" : args.use_flow_local_mem,
            "use_im_memory" : args.use_im_memory,
            "use_flow_memory" : args.use_flow_memory,
            "use_im_skipcon_1" : args.use_im_skipcon_1,
            "use_im_skipcon_2" : args.use_im_skipcon_2,
            "use_im_skipcon_3" : args.use_im_skipcon_3,
            "use_im_skipcon_4" : args.use_im_skipcon_4,
            "use_flow_skipcon_1" : args.use_flow_skipcon_1,
            "use_flow_skipcon_2" : args.use_flow_skipcon_2,
            "use_flow_skipcon_3" : args.use_flow_skipcon_3,
            "use_flow_skipcon_4" : args.use_flow_skipcon_4,
          }
        ) # +"-"+str(time.ctime(int(time.time())) )

NUM_TEMPORAL_FRAME = 2
INDEX_STEP = 1
TRAIN_VAL_SPLIT = 0.0
BATCH_SIZE = args.batch_size

train_images , train_flows = load_raw_data(data_type=args.data_type,train_or_test="train",data_dir=args.dataset_path)

sample_video_frame_index , train_index , val_index = get_index_sample(train_images,INDEX_STEP,NUM_TEMPORAL_FRAME)

foreground_list = []
for index in tqdm.tqdm(sample_video_frame_index):
  video_index = index[0]
  frame_index = index[1:]
  foreground_list.append(  np.array( [ train_images[video_index][img_index] for img_index in frame_index  ] )   )

flow_list = []
for index in tqdm.tqdm(sample_video_frame_index):
  video_index = index[0]
  frame_index = index[1:]
  flow_list.append(   np.transpose( np.array( [ train_flows[video_index][img_index] for img_index in frame_index[1:]  ] ) , (0,3,1,2)   ).reshape(2,128,192) )

data_train =  loader(train_index,foreground_list,flow_list)

train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, 
                                  shuffle=True, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

discriminator = Discriminator(5)

discriminator = discriminator.to(device)
if args.weight_init:
  discriminator.apply(weights_init_memAE_git)
discriminator.train()

model_config_dict = {
  "use_im_local_mem" : args.use_im_local_mem,
  "use_flow_local_mem" : args.use_flow_local_mem,
  "use_im_memory" : args.use_im_memory,
  "use_flow_memory" : args.use_flow_memory,
  "use_im_skipcon_1" : args.use_im_skipcon_1,
  "use_im_skipcon_2" : args.use_im_skipcon_2,
  "use_im_skipcon_3" : args.use_im_skipcon_3,
  "use_im_skipcon_4" : args.use_im_skipcon_4,
  "use_flow_skipcon_1" : args.use_flow_skipcon_1,
  "use_flow_skipcon_2" : args.use_flow_skipcon_2,
  "use_flow_skipcon_3" : args.use_flow_skipcon_3,
  "use_flow_skipcon_4" : args.use_flow_skipcon_4,
}

p_keep = 1 - args.train_dropout
generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize, model_config_dict)

generator = generator.to(device)
if args.weight_init:
  generator.apply(weights_init_memAE_git)
generator.train()

weight_decay_ae=0.5e-5

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas = (0.5, 0.9),
                        weight_decay=weight_decay_ae)
generator_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr, betas = (0.5, 0.9),
                        weight_decay=weight_decay_ae)
# scheduler = optim.lr_scheduler.MultiStepLR(discriminator_optimizer, 
#             milestones=lr_milestones, gamma=0.1)

train_log = Loss_log(['G_loss', 'D_loss', 'opt_loss', 'gradi_loss' , 'inten_loss' ],args.exp_dir)

checkpoint_save_path = args.exp_dir +"/"+ args.data_type +"-"+ args.wandb_run_name +"/"

if not os.path.exists(checkpoint_save_path):
    os.makedirs(checkpoint_save_path)

model_name = "original"
# Training Loop
# Lists to keep track of progress
iters = 0
num_epochs = args.epochs
print("Starting Training Loop...")
# For each epoch
BCEWithLogits = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    iters = 0
    if epoch >= 150:
      break
    # For each batch in the dataloader
    for real_image,  real_flow  in Bar(train_loader):

        ############################
        # (1) Update D network
        ###########################
   
        discriminator.zero_grad()
        generator.zero_grad()
        
        plh_frame_true = real_image.float().to(device)
        plh_flow_true = real_flow.float().to(device)

        output_opt, output_appe= generator(plh_frame_true,plh_flow_true)

        # discriminator
        D_real, D_real_logits = discriminator(plh_frame_true, plh_flow_true)
        D_fake, D_fake_logits = discriminator(plh_frame_true, output_opt.clone().detach())

        D_loss = get_d_loss(D_real_logits,D_real,D_fake_logits,D_fake)
        D_loss.backward()
        discriminator_optimizer.step()

        ############################
        # (2) Update G network
        ###########################

        D_fake, D_fake_logits = discriminator(plh_frame_true, output_opt)
      
        G_loss = torch.mean(BCEWithLogits(D_fake_logits, torch.ones_like(D_fake)))
        loss_opt = l1_loss(plh_flow_true,output_opt)

        loss_gradi = gradient_diff_loss( output_appe , plh_frame_true )
        loss_inten = l2_loss(output_appe,plh_frame_true)

        loss_appe = loss_inten + loss_gradi 

        G_loss_total = loss_appe + 2*loss_opt +  0.25*G_loss 
        G_loss_total.backward()
        generator_optimizer.step()
        
        # Output training stats
        if iters % 200 == 0:
            visualizing(real_image,output_appe,real_flow,output_opt,checkpoint_save_path,epoch)
            test_flow_vil(output_opt,real_flow,checkpoint_save_path,epoch)

            print('TRAIN [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f ,loss_opt %.4f  ,loss_gradi %.4f ,loss_inten %.4f   '
                  % (epoch, num_epochs, iters, len(train_loader),
                     D_loss.item(), G_loss.item(), loss_opt.item()  , loss_gradi.item() ,  loss_inten.item()  ))
        train_log.add_inter_loss({
            'G_loss' : G_loss.item(), 'D_loss': D_loss.item(), 'opt_loss': loss_opt.item(), 'gradi_loss': loss_gradi.item()  , 'inten_loss': loss_inten.item() 
        })

        iters += 1
      
    now_epoch = train_log.end_epoch(True)
    torch.save(generator.state_dict(), checkpoint_save_path+"/"+model_name+"_gen_"+str(now_epoch) +".pt")
    torch.save(discriminator.state_dict(), checkpoint_save_path+"/"+model_name+"_dis_"+ str(now_epoch)+".pt")

    if args.wandb_log:
      for name in train_log.loss_name_list:
        wandb.log({ str(name).replace("_","") : float(train_log.epoch_loss[name][-1]) } , step=int(now_epoch) )
        
    #   wandb.log({"gen_dir": checkpoint_save_path+"/"+model_name+"_gen_"+str(now_epoch) +".pt", 
    #             "dis_dir": checkpoint_save_path+"/"+model_name+"_dis_"+ str(now_epoch)+".pt"}, step=now_epoch)

p_keep = 1.0
test_generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize,model_config_dict)
if args.weight_init:
  test_generator.apply(weights_init_memAE_git)
test_generator = test_generator.to(device)
test_generator.load_state_dict(generator.state_dict())

for real_image,  real_flow  in Bar(train_loader):
  with torch.no_grad():
    plh_frame_true = real_image.float().to(device)
    plh_flow_true = real_flow.float().to(device)
    output_opt, output_appe   = test_generator(plh_frame_true,plh_flow_true)

test_generator.eval()

torch.save(test_generator.state_dict(), checkpoint_save_path+"/"+model_name+"_final_gen_"+str(now_epoch) +".pt")

# if args.wandb_log:
#   wandb.log({"final_model_dir": checkpoint_save_path+"/"+model_name+"_final_gen_"+str(now_epoch) +".pt"})

p_keep = 1.0
test_generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize,model_config_dict)
test_generator = test_generator.to(device)

test_generator.load_state_dict(torch.load( checkpoint_save_path+"/"+model_name+"_final_gen_"+str(args.epochs) +".pt"))

test_generator.eval()


raw_ground_truth = load_raw_groundtruth(data_type=args.data_type,groundtruth_dir=args.groundtruth_path)

print("Test len: ",len(raw_ground_truth))

test_images , test_flows = load_raw_data(data_type=args.data_type,train_or_test="test",data_dir=args.dataset_path)

test_sample_video_frame_index , test_index , labels_temp, video_index_label = get_index_sample_and_label(test_images,raw_ground_truth,args.data_type,INDEX_STEP,NUM_TEMPORAL_FRAME)

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

test_step = 0
if args.eval_method == "all" or args.eval_method == "normal":
  score_frame_ = []
  for real_image,  real_flow  in Bar(test_loader):
    with torch.no_grad():

      plh_frame_true = real_image.float().to(device)
      plh_flow_true = real_flow.float().to(device)

      output_opt, output_appe = test_generator(plh_frame_true,plh_flow_true)
    
      if random.random() <= 0.3:
        visualizing(real_image,output_appe,real_flow,output_opt,checkpoint_save_path,test_step,"test")
      
      output_appe = output_appe.detach().cpu().numpy().transpose((0,2,3,1))*0.5 + 0.5
      real_image = real_image.detach().cpu().numpy().transpose((0,2,3,1))*0.5 + 0.5
      real_flow = real_flow.detach().cpu().numpy().transpose((0,2,3,1))
      output_opt = output_opt.detach().cpu().numpy().transpose((0,2,3,1))
      
      # print(output_opt.min(),output_opt.max(),real_flow.min(),real_flow.max())
      # print(output_appe.min(),output_appe.max(),real_image.min(),real_image.max())
      for i in range(len(real_flow)):
        score_frame_.append(calc_anomaly_score_one_frame(real_image[i].astype(np.float32), output_appe[i].astype(np.float32), real_flow[i].astype(np.float32), output_opt[i].astype(np.float32)))
    test_step += 1

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

if args.finalscore_maxnorm_byclip:
  maxnorm_byclip = max_norm_score_by_clip(combine_max_score,video_index_label)
  print("Combine maxperclip AUC: ",roc_auc_score(labels_temp, maxnorm_byclip))
  print("Combine maxperclip AP: ",average_precision_score(labels_temp, maxnorm_byclip))
  if args.wandb_log:
    wandb.log({"MaxClipCombineAUC": roc_auc_score(labels_temp, maxnorm_byclip)})
    wandb.log({"MaxClipCombineAP":  average_precision_score(labels_temp, maxnorm_byclip) })