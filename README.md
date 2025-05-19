# EIF6D
![image](https://github.com/user-attachments/assets/a017f9e3-4bb2-4b46-a472-14073909696f)

## Getting startted
#### ***Prepare the environment***

``` shell
conda env create -f environment.yml

```

#### ***Compiling***
```shell
# Compile pointnet2
cd model/pointnet2
python setup.py install
```

#### ***Prepare the datasets***
For REAL275 and CAMERA25 datasets, please follow the [instruction](https://github.com/JiehongLin/Self-DPDN) in DPDN 
and [instruction](https://github.com/mentian/object-deformnet) in SPD.


### Training from scartch
```shell
# gpus refers to the ids of gpu. For single gpu, please set it as 0
python train_PT2.py --gpus 0,1 --config config/PT2Net.yaml
```


### Evaluation
```shell
python test_PT2.py --config config/PT2Net.yaml

```

## Acknowledgement
- Our code is developed upon [DPDN](https://github.com/JiehongLin/Self-DPDN) and [IST-Net](https://github.com/CVMI-Lab/IST-Net).
- The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019). 


