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
!python train_and_eval.py --wandb_run_name "test_wandb_1" --dataset_path .. --data_type belleview --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path "../Traffic-Belleview/"
```

* Ped2
```bash
!cp /content/drive/MyDrive/Dataset/Traffic-Belleview-2/train.zip /content
!cp /content/drive/MyDrive/Dataset/Traffic-Belleview-2/test.zip /content
!unzip train.zip 
!unzip test.zip 
!cp "/content/drive/MyDrive/Dataset/Traffic-Belleview-2/Traffic-Belleview.zip" . 
!unzip Traffic-Belleview.zip
!git clone https://github.com/Khang-9966/Traffic-video-anomaly-detection.git
%cd Traffic-video-anomaly-detection/
!python train_and_eval.py --wandb_run_name "test_wandb_1" --dataset_path .. --data_type belleview --exp_dir "/content/drive/MyDrive/checkpoint/" --wandb_log True  --groundtruth_path "../Traffic-Belleview/"
```

## Testing
* Belleview
```bash
python train.py --dataset_path .. --data_type belleview --exp_dir "../save_model"
python eval.py  --dataset_path .. --data_type belleview --exp_dir "../save_model" --exp_dir "../save_model" --groundtruth_path "../Traffic-Belleview/"
```
* Ped2


