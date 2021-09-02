# [On Universal Black-Box Domain Adaptation]

This repository provides code for the paper ---- [On Universal Black-Box Domain Adaptation](https://arxiv.org/pdf/2104.04665.pdf).

## Environment

Python 3.8.5, Pytorch 1.4.0 with CUDA, Torch Vision 0.5.0, and the Nvidia apex library [Apex](https://github.com/NVIDIA/apex). For details of our running environment , please see requirement.txt.

## Data preparation

[Office31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

[Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)

[DomainNet](http://ai.bu.edu/M3SDA/)

Prepare dataset in data directory as follows.

'''

./data/Office31/amazon/...

./data/Office31/dslr/...

./data/Office31/webcam/...



./data/OfficeHome/Art/...

./data/OfficeHome/Clipart/...

./data/OfficeHome/Product/...

./data/OfficeHome/Real_World/...



./data/DomainNet/painting/...

./data/DomainNet/sketch/...

./data/DomainNet/real/...

'''

## Train

'''

sh script/run_office31.sh $gpu-id

sh script/run_officehome.sh $gpu-id

sh script/run_domainNet.sh $gpu-id

'''


## Reference

This repository is contributed by [Bin Deng](https://bindeng.xyz/).
If you consider using this code or its derivatives, please consider citing:

```
@misc{deng2021universal,
      title={On Universal Black-Box Domain Adaptation}, 
      author={Bin Deng and Yabin Zhang and Hui Tang and Changxing Ding and Kui Jia},
      year={2021},
      eprint={2104.04665},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

