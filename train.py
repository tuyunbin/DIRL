import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random

from configs.config_transformer import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.DIRL import DIRL, AddSpatialInfo
from models.CCR import CCR
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_sequence_transformer, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss, LabelSmoothingLoss

from utils.vis_utils import visualize_att
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
# parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# print(os.path.basename(args.cfg).replace('.yaml', ''))
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')



# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
torch.backends.cudnn.enabled = False
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

random.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)

# Create model
change_detector = DIRL(cfg)
change_detector.to(device)

speaker = CCR(cfg)
speaker.to(device)

spatial_info = AddSpatialInfo()


print(change_detector)
print(speaker)


with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0

set_mode('train', [change_detector, speaker])
# ss_prob = speaker.ss_prob

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    # lr_scheduler.step()
    speaker_loss_avg = AverageMeter()
    cdcr_loss_avg = AverageMeter()
    sim_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()

    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, sc_feats, \
        labels, labels_with_ignore, masks, d_img_paths, sc_img_paths = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)

        masks = masks.squeeze(1).float()

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)

        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)

        optimizer.zero_grad()

        diff_bef_pos, diff_aft_pos, dirl_loss = change_detector(d_feats, sc_feats)


        loss_pos, att_pos, ccr_loss = speaker._forward(diff_bef_pos, diff_aft_pos,
                                              labels, masks, labels_with_ignore=labels_with_ignore)


        speaker_loss = loss_pos
        speaker_loss_val = speaker_loss.item()


        dirl_loss_val = dirl_loss.item()

        ccr_loss_val = ccr_loss.item()

        total_loss = speaker_loss + 0.03 * dirl_loss + 0.05 * ccr_loss

        total_loss_val = total_loss.item()

        speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
        cdcr_loss_avg.update(dirl_loss_val, 2 * batch_size)
        sim_loss_avg.update(ccr_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}

        stats['speaker_loss'] = speaker_loss_val
        stats['avg_speaker_loss'] = speaker_loss_avg.avg
        stats['cdcr_loss'] = dirl_loss_val
        stats['avg_cdcr_loss'] = cdcr_loss_avg.avg
        stats['sim_loss'] = ccr_loss_val
        stats['avg_sim_loss'] = sim_loss_avg.avg
        stats['total_loss'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg

        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        total_loss.backward()
        if cfg.train.grad_clip != -1.0:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(change_detector.parameters(), cfg.train.grad_clip)
            nn.utils.clip_grad_norm_(speaker.parameters(), cfg.train.grad_clip)

        optimizer.step()
        # lr_scheduler.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)


                result_sents_pos = {}

                for val_i, val_batch in enumerate(val_loader):
                    d_feats, sc_feats, \
                    labels, labels_with_ignore, masks, \
                    d_img_paths, sc_img_paths = val_batch

                    val_batch_size = d_feats.size(0)

                    d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)

                    labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(
                        device)

                    diff_bef_pos, diff_aft_pos, dirl_loss = change_detector(d_feats, sc_feats)


                    speaker_output_pos, _ = speaker.sample(diff_bef_pos, diff_aft_pos)


                    gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:]) # no start

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence_transformer(idx_to_word, labels[val_j][:, 1:])

                        sent_pos = gen_sents_pos[val_j]

                        image_id = d_img_paths[val_j].split('/')[-1]
                        result_sents_pos[image_id] = sent_pos

                        message = '%s results:\n' % d_img_paths[val_j]
                        message += '\t' + sent_pos + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'

                        print(message)


                test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')

                coco_gen_format_save(result_sents_pos, result_save_path_pos)


            set_mode('train', [change_detector, speaker])
    lr_scheduler.step()