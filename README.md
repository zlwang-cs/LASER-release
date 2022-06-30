# LASER

This is the pytorch implemention for the ACL '22 Finding paper [Towards Few-shot Entity Recognition in Document Images: A Label-aware Sequence-to-Sequence Framework](https://arxiv.org/abs/2204.05819).

## Quick Start

*The details about the running environment will be released later. If you encounter any problems about it, please refer to the environment for [LayoutReader](https://github.com/microsoft/unilm/tree/master/layoutreader) or contact me directly.*

### Pretrained Weight

Please download the pre-trained weight of LayoutReader from [here](https://github.com/microsoft/unilm/tree/master/layoutreader) and copy `pytorch_model.bin` into `./weights/layoutreader/`.

### Train & Decode & Evaluate
```
git clone https://github.com/zlwang-cs/LASER-release.git
cd LASER-release
mkdir outputs
cd shell_scripts
sh run_few_shot_FUNSD.sh
```

### Collect Data
see `collect_results.ipynb`

## Citation
If you find the code useful, please cite our paper:

```
@inproceedings{wang2022towards,
  title={Towards Few-shot Entity Recognition in Document Images: A Label-aware Sequence-to-Sequence Framework},
  author={Wang, Zilong and Shang, Jingbo},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={4174--4186},
  year={2022}
}
```
