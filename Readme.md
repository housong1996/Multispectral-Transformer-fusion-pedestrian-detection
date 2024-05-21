## Dependency package
Two way:

1.For environment configuration, please refer to the dependency package installation of yolov5:
https://github.com/ultralytics/yolov5

or

2.Install the installation packages required for the virtual environment directly according to the following files:
conda create --name mstf --file conda_mstf_env.txt

## Datasets
The KAIST and LLVIP dataset in the matched format can be downloaded directly from the following cloud link:
https://pan.baidu.com/s/1jh6B88BtZQrRmIUWsjS_aA?pwd=dgct

## Download trained model
The "MSTF_train_onkaist.pt" and "MSTF_train_onllvip.pt" in "weights" folder are the trained model on kaist and llvip dataset, which can be download from the cloud link:
https://pan.baidu.com/s/1wj5iBRA0KmCRdU7sJlmdag?pwd=3yqv

## Screen recording video of code demonstration
Available from the following cloud link:
https://pan.baidu.com/s/1jBZ3_d4SBxa55TwxQJ9HrQ?pwd=dyf7


## Train:
### on kaist:  
nohup python train.py --cfg ./models/fusion_mstf_model.yaml --batch-size 64 --epochs 50 --data ./data/kaist_sanitized.yaml --name mstf_train_onkaist --device 0,1,2,3 > ./mstf_train_onkaist.log 2>&1 &

### on llvip:
python train.py --cfg ./models/fusion_mstf_model.yaml --batch-size 64 --epochs 50 --data ./data/llvip.yaml --name mstf_train_onllvip --device 0,1,2,3


## Test
### on kaist:
python test.py --weights ./weights/MSTF_train_onkaist.pt --batch-size 32 --data ./data/kaist_sanitized.yaml --device 0

### on llvip:
python test.py --weights ./weights/MSTF_train_onllvip.pt --batch-size 32 --data ./data/llvip.yaml --device 0

