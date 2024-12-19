# Distractors-Immune Representation Learning with Cross-modal Contrastive Regularization for Change Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Distractors-Immune Representation Learning with Cross-modal Contrastive Regularization for Change Captioning"](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05989.pdf), which has appeared as a regular paper in ECCV 2024. 

## We illustrate the training and testing details as follows:

## Installation

1. Make virtual environment with miniconda (`conda create -n card python=3.8`)
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/tylin/coco-caption)) (Since the repo only supports Python 2.7, either create a separate virtual environment with Python 2.7 or modify the code to be compatible with Python 3.5).

## Data
1. Download image data from here: [viewpoint-agnostic change captioning with cycle consistency (ICCV'21)](https://github.com/hsgkim/clevr-dc)
2. You need to split them as bef-change images and aft-change images, and put two kinds of images into two directories, namely `images` and `sc_images`. 
I have also uploaded my downloaded images into the baidu drive [clevr-dc.zip](https://pan.baidu.com/s/1VK6dH7BQ7rYaIVYOYLVZGg?pwd=dc24), where the extraction code is `dc24`.
3. After obtaining the image pairs and captions, you should rename them first by using the following commands:
```
python pad_img.py
python rename_dc_caption.py
```   


5. Preprocess data

 Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
``
python scripts/extract_features.py --input_image_dir ./spot-the-diff/images --output_dir ./spot-the-diff/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./spot-the-diff/sc_images --output_dir ./spot-the-diff/sc_features --batch_size 128


* Build vocab and label files using caption annotations (Both files have been in the `spot-the-diff'' directory, so you can skip this step.)
``
python scripts/preprocess_captions_multi_spot.py
```

## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train_card_spot.py --cfg configs/dynamic/transformer_multi_spot.yaml
```

## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test_card_spot.py --cfg configs/dynamic/transformer_multi_spot.yaml --snapshot 8000 --gpu 1
```
The command above will take the model snapshot at 8000th iteration and run inference using GPU ID 1.

## Evaluation
* Caption evaluation

To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools ([github](https://github.com/tylin/coco-caption)), make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils_spot.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate_spot.py --results_dir ./experiments/card_spot/eval_sents --anno ./spot-the-diff/change_multi_captions_reformat.json 
```

Once the best model is found on the validation set, you can run inference on test set:
```
python evaluate_spot.py --results_dir ./experiments/card_spot/test_output/captions --anno ./spot-the-diff/change_multi_captions_reformat.json 
```
The results are saved in `./experiments/card_spot/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@inproceedings{tu2024context,
  title={Context-aware Difference Distilling for Multi-change Captioning},
  author={Tu, Yunbin and Li, Liang and Su, Li and Zha, Zheng-Jun and Yan, Chenggang and Huang, Qingming},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7941--7956},
  year={2024}
}
```

## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!



