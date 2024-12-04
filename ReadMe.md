# FedPSD (Federated Progressive Self-Distillation with Logits Calibration for Personalized IIoT)

This repository is the official PyTorch implementation of:

"Federated Progressive Self-Distillation with Logits Calibration for Personalized IIoT " 

<img src="assets/fedPSD.png" width="800"/>

Our code is based on the [FedNTD](https://github.com/Lee-Gihun/FedNTD/)  library.



## Requirements
- Our experimental environment is based on PyTorch version 2.0.1 and CUDA versions 11.8 and 12.0.

- Install and configure the necessary environment by using: `pip install requirements.txt`


## How to Run Codes?

The usage of our proposed algorithm is as follows:
```
cd FEDPSD/
python ./main.py --config_path ./config/fedpsd.json
``` 
The usage of other algorithms can refer to the following format:
```
cd FEDPSD/
python ./main.py --config_path ./config/${algorithm_name}.json
```
## Reference Github

We are referring to the following repositories:
- https://github.com/Lee-Gihun/FedNTD
- https://github.com/Lee-Gihun/FedLMD

## Citing this work

```
@ARTICLE{2024arXiv241200410W,
       author = {{Wang}, Yingchao and {Niu}, Wenqi},
        title = "{Federated Progressive Self-Distillation with Logits Calibration for Personalized IIoT Edge Intelligence}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Artificial Intelligence},
         year = 2024,
        month = nov,
          eid = {arXiv:2412.00410},
        pages = {arXiv:2412.00410},
archivePrefix = {arXiv},
       eprint = {2412.00410},
 primaryClass = {cs.AI},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241200410W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
