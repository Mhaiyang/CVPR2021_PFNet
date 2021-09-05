# CVPR2021_PFNet

## Camouflaged Object Segmentation With Distraction Mining
[Haiyang Mei](https://mhaiyang.github.io/), Ge-Peng Ji, Ziqi Wei, Xin Yang, Xiaopeng Wei, [Deng-Ping Fan](http://dpfan.net/)

[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Camouflaged_Object_Segmentation_With_Distraction_Mining_CVPR_2021_paper.pdf)] [[Project Page](https://mhaiyang.github.io/CVPR2021_PFNet/index.html)]

### Abstract
Camouflaged object segmentation (COS) aims to identify objects that are "perfectly" assimilate into their surroundings, which has a wide range of valuable applications. The key challenge of COS is that there exist high intrinsic similarities between the candidate objects and noise background. In this paper, we strive to embrace challenges towards effective and efficient COS. To this end, we develop a bio-inspired framework, termed Positioning and Focus Network (PFNet), which mimics the process of predation in nature. Specifically, our PFNet contains two key modules, i.e., the positioning module (PM) and the focus module (FM). The PM is designed to mimic the detection process in predation for positioning the potential target objects from a global perspective and the FM is then used to perform the identification process in predation for progressively refining the coarse prediction via focusing on the ambiguous regions. Notably, in the FM, we develop a novel distraction mining strategy for the distraction region discovery and removal, to benefit the performance of estimation. Extensive experiments demonstrate that our PFNet runs in real-time (72 FPS) and significantly outperforms 18 cutting-edge models on three challenging benchmark datasets under four standard metrics.

### Citation
If you use this code, please cite:

```
@InProceedings{Mei_2021_CVPR,
    author    = {Mei, Haiyang and Ji, Ge-Peng and Wei, Ziqi and Yang, Xin and Wei, Xiaopeng and Fan, Deng-Ping},
    title     = {Camouflaged Object Segmentation With Distraction Mining},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8772-8781}
}
```

### Requirements
* PyTorch == 1.0.0
* TorchVision == 0.2.1
* CUDA 10.0  cudnn 7.2

### Train
Download 'resnet50-19c8e357.pth' at [here](https://download.pytorch.org/models/resnet50-19c8e357.pth), then run `train.py`.


### Test
Download trained model 'PFNet.pth' at [here](https://mhaiyang.github.io/CVPR2021_PFNet/index.html), then run `infer.py`.

### License
Please see `license.txt`

### Contact
E-Mail: mhy666@mail.dlut.edu.cn
