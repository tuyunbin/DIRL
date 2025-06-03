# Distractors-Immune Representation Learning with Cross-modal Contrastive Regularization for Change Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Distractors-Immune Representation Learning with Cross-modal Contrastive Regularization for Change Captioning"](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05989.pdf), which has appeared as a regular paper in ECCV 2024. The arxiv version is [here.](https://arxiv.org/pdf/2407.11683)

## Upadte
The training and testing code files for CLEVR-Change dataset have been uploaded. The data files and checkpoint file are in  the baidu drive [dirl_clevr_change]
(https://pan.baidu.com/s/1WolB0B2rbV377I1B2hzlwA?pwd=dirl), where the extration code is dirl 

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
# rename image pairs 
python pad_img.py

# rename captions
python rename_dc_caption.py
```   


5. Preprocess data

 Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
``
python scripts/extract_features.py --input_image_dir ./clevr_dc/images --output_dir ./clevr_dc/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./clevr_dc/sc_images --output_dir ./clevr_dc/sc_features --batch_size 128


* Build vocab and label files using caption annotations.
``
python scripts/preprocess_captions_dc.py
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
python train.py --cfg configs/dynamic/transformer_dc.yaml
```

## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test.py --cfg configs/dynamic/transformer_dc.yaml --snapshot 12000 --gpu 1
```
The command above will take the model snapshot at 12000th iteration and run inference using GPU ID 1.

## Evaluation
* Caption evaluation

To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools ([github](https://github.com/tylin/coco-caption)), make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils_dc.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate_dc.py --results_dir ./experiments/DIRL+CCR/eval_sents --anno ./clevr_dc/change_captions_reformat.json 
```

Once the best model is found on the validation set, you can run inference on test set:
```
python evaluate_dc.py --results_dir ./experiments/DIRL+CCR/test_output/captions --anno ./clevr_dc/change_captions_reformat.json 
```
The results are saved in `./experiments/DIRL+CCR/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@inproceedings{tu2024distractors,
  title={Distractors-Immune Representation Learning with Cross-modal Contrastive Regularization for Change Captioning},
  author={Tu, Yunbin and Li, Liang and Su, Li and Yan, Chenggang and Huang, Qingming},
  booktitle={ECCV},
  pages={311--328},
  year={2024},
}
```

## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!



