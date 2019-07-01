# DREML
A PyTorch implementation of DREML based on ECCV 2018 paper [Deep Randomized Ensembles for Metric Learning](https://arxiv.org/abs/1808.04469).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```

## Datasets
[cars196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [cub200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
and [sop](http://cvgl.stanford.edu/projects/lifted_struct/) are used in this repo.

You should download these datasets by yourself, and extract them into `data` directory, make sure the dir names are 
`car`, `cub` and `sop`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --num_epochs 12
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop'])
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 128]
--num_epochs                  train epochs number [default value is 12]
--ensemble_size               ensemble model size [default value is 12]
--meta_class_size             meta class size [default value is 12]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size of 128 on one 
NVIDIA Tesla V100 (32G) GPU.

The images are preprocessed with random resize, random crop, random horizontal flip, and normalize.

For `Cars196` and `CUB200` datasets, `20` epochs, ensemble size `48` and meta class size `12` are used. For `SOP` dataset,
`50` epochs, ensemble size `16` and meta class size `500` is used.

The pretrained model and test images' features can be download from [BaiduYun](https://pan.baidu.com/s/14CE3GaDN1dxnPcGPXT4YEg) (access code:2n9m).

Here is the recall details:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Cars196</th>
      <th>CUB200</th>
      <th>SOP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">84.96%</td>
      <td align="center">63.52%</td>
      <td align="center">73.70%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">90.74%</td>
      <td align="center">74.46%</td>
      <td align="center">78.57%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">94.27%</td>
      <td align="center">82.88%</td>
      <td align="center">82.46%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">96.70%</td>
      <td align="center">89.01%</td>
      <td align="center">85.56%</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">97.25%</td>
      <td align="center">90.56%</td>
      <td align="center">86.39%</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.79%</td>
      <td align="center">98.85%</td>
      <td align="center">92.76%</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.98%</td>
      <td align="center">96.43%</td>
    </tr>
  </tbody>
</table>

