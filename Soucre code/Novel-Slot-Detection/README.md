- Các thành viên: Võ Minh Trí - 19522395, Trần Triệu Vũ - 19522539, Phạm Đức Thể - 19522253
- Đồ Án Khoa Học Dữ Liệu Và Ứng Dụng


## Introduction
The Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System.




## Dependencies

We use anaconda to create python environment:
```
conda create --name python=3.6
```
Install all required libraries:
```
pip install -r requirements.txt
```

## How to run
#### 1. Train (only):
```
python --mode train --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 
```
#### 2. Predict (only):
```
python --mode test --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 
```
#### 1. Train and predict (Both):
```
python --mode both --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 
```
## Parameters
- `mode`, optional, Specify running mode, only `train`,only`test` or `both`.
- `dataset`, required, The dataset to use, `SnipsNSD5%` or `SnipsNSD15%` or `SnipsNSD30%`.
- `threshold`, required, The specified threshold value.
- `output_dir`, default="./output"
- `batch_size`, default=256
- `cuda`, default=1





