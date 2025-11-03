You need to **strictly** follow our instruction to prepare the environment of UniLION.

## Create environment
```conda create -n unilion python==3.9```

## Install torch & torchvision
Our codebase is validated on CUDA 12.4. 

```torch==2.5.0+cu124, torchvision==0.20.0+cu124```

## Install mmcv
We modify mmcv codebase to make it suitable for higher versions of torch and CUDA.

```cd mmv```

```MMCV_WITH_OPS=1 pip setup.py install```

## Install mmdetection3d
We modify mmdetection3d codebase to make it suitable for higher versions of torch and CUDA.

```cd mmdetection3d```

```python setup.py install```

## Install package
```pip install -r requirement.txt```

## Install mamba
```projects/mmdet3d_plugin/models/ops/mamba```

## Install unilion

```cd  projects```

```python setup.py develop```