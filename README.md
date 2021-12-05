# How to Run

## Dependencies
* pip install barbar

## Datasets
* USCD Ped2 [[dataset]()]
* CUHK Avenue [[dataset]()]
* Traffic-Belleview [[dataset]()]
* Traffic-Train [[dataset]()]
* IDIAP [[dataset]()]
* AIC [[dataset]()]


## Training

```bash
python train.py --dataset_path .. --data_type belleview --exp_dir "../save_model"
```

## Testing
```bash
python eval.py  --dataset_path .. --data_type belleview --exp_dir "../save_model" --exp_dir "../save_model" --groundtruth_path "../Traffic-Belleview/"
```