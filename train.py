from loss import *
from utils import *
from model import *
from dataset import *
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


args = parser.parse_args()


if args.wandb_log:
  wandb.init(project=args.data_type,  name = args.wandb_run_name, entity="khang-9966") # +"-"+str(time.ctime(int(time.time())) )

  wandb.config = {
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "h": args.h,
    "w": args.w,
    "g_lr": args.g_lr,
    "d_lr": args.d_lr,
    "fdim": args.fdim,
    "mdim": args.mdim,
    "im_msize": args.im_msize,
    "flow_msize": args.flow_msize,
    "data_type": args.data_type,
    "exp_dir": args.exp_dir,
  }

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
# discriminator.apply(weights_init_normal)
discriminator.train()

p_keep = 0.7
generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize)

generator = generator.to(device)
# generator.apply(weights_init_normal)
generator.train()

weight_decay_ae=0.5e-5

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas = (0.5, 0.9),
                        weight_decay=weight_decay_ae)
generator_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr, betas = (0.5, 0.9),
                        weight_decay=weight_decay_ae)
# scheduler = optim.lr_scheduler.MultiStepLR(discriminator_optimizer, 
#             milestones=lr_milestones, gamma=0.1)

train_log = Loss_log(['G_loss', 'D_loss', 'opt_loss', 'gradi_loss' , 'inten_loss' ],args.exp_dir)

checkpoint_save_path = args.exp_dir
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
        wandb.log({ name : float(train_log.epoch_loss[name][-1]) } )
    #   wandb.log({"gen_dir": checkpoint_save_path+"/"+model_name+"_gen_"+str(now_epoch) +".pt", 
    #             "dis_dir": checkpoint_save_path+"/"+model_name+"_dis_"+ str(now_epoch)+".pt"}, step=now_epoch)

p_keep = 1.0
test_generator = Generator(128,192,2,3,p_keep,args.im_msize,args.flow_msize)
test_generator = test_generator.to(device)
test_generator.load_state_dict(generator.state_dict())

for real_image,  real_flow  in Bar(train_loader):
  with torch.no_grad():
    plh_frame_true = real_image.float().to(device)
    plh_flow_true = real_flow.float().to(device)

    output_opt, output_appe   = test_generator(plh_frame_true,plh_flow_true)

test_generator.eval()

torch.save(test_generator.state_dict(), checkpoint_save_path+"/"+model_name+"_final_gen_"+str(now_epoch) +".pt")

if args.wandb_log:
  wandb.log({"final_model_dir": checkpoint_save_path+"/"+model_name+"_final_gen_"+str(now_epoch) +".pt"})