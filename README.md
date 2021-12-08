# How to Run

## Dependencies
* pip install barbar
* pip install wandb
## Datasets
* USCD Ped2 [[dataset]()]
* CUHK Avenue [[dataset]()]
* Traffic-Belleview [[dataset]()]
* Traffic-Train [[dataset]()]
* IDIAP [[dataset]()]
* AIC [[dataset]()]


## Wandb ML ops
* Before run training
```bash
wandb login --cloud [KEY]
```

## Training and Eval 
* Belleview
```bash
!cp /content/drive/MyDrive/Dataset/Traffic-Belleview-2/train.zip /content
!cp /content/drive/MyDrive/Dataset/Traffic-Belleview-2/test.zip /content
!unzip train.zip 
!unzip test.zip 
!cp "/content/drive/MyDrive/Dataset/Traffic-Belleview-2/Traffic-Belleview.zip" . 
!unzip Traffic-Belleview.zip
!git clone https://github.com/Khang-9966/Traffic-video-anomaly-detection.git
%cd Traffic-video-anomaly-detection/
!python train_and_eval.py --wandb_run_name "test_wandb_1" --dataset_path .. --data_type "belleview" --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path "../Traffic-Belleview/"
```

* Train
```bash
!cp /content/drive/MyDrive/Dataset/Traffic-Train/train.zip /content
!cp /content/drive/MyDrive/Dataset/Traffic-Train/test.zip /content
!unzip train.zip 
!unzip test.zip 
!cp "/content/drive/MyDrive/Dataset/Traffic-Train/Traffic-Train.zip" . 
!unzip Traffic-Train.zip
!git clone https://github.com/Khang-9966/Traffic-video-anomaly-detection.git
%cd Traffic-video-anomaly-detection/
!python train_and_eval.py  --wandb_run_name "test_wandb_1" --dataset_path ".." --data_type "train" --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path "../Traffic-Train/"
```

* Ped2
```bash
!cp /content/drive/MyDrive/Dataset/UCSD/ped2_flows.zip /content
!unzip ped2_flows.zip 
!git clone https://github.com/Khang-9966/Traffic-video-anomaly-detection.git
%cd Traffic-video-anomaly-detection/
!python train_and_eval.py --wandb_run_name "test_wandb_1" --dataset_path "/content/content/flownet2pytorch/FlowNet2-pytorch/" --data_type "ped2" --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path ""
```

* Idiap
```bash
!cp  /content/drive/MyDrive/Dataset/DataRelease-TrafficJunction-idiap/idiap_dataset_10FPS_192_128.zip .
!unzip idiap_dataset_10FPS_192_128.zip
!git clone https://github.com/Khang-9966/Traffic-video-anomaly-detection.git
%cd Traffic-video-anomaly-detection/
!python train_and_eval.py --epochs 25 --wandb_run_name "idiap_test_wandb_1" --dataset_path "../dataset/idiap/" --data_type "idiap" --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path "../dataset/idiap/"
```

## Testing
* Belleview
```bash
python train.py --dataset_path .. --data_type belleview --exp_dir "../save_model"
python eval.py  --dataset_path .. --data_type belleview --exp_dir "../save_model" --exp_dir "../save_model" --groundtruth_path "../Traffic-Belleview/"
```
* Ped2


