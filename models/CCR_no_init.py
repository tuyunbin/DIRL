from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class CrossTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        self.self_att = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.cross_att = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2, dec_mask, diagonal_mask=True):
        tgt_length = dec_mask.size(1)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        pad_mask = 1 - dec_mask
        pad_mask = pad_mask.bool()
        attn_output, self_weight = self.self_att(input1, input1, input1, attn_mask=mask,key_padding_mask=pad_mask)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)

        text = output.mean(0)

        attn_output, cross_weight = self.cross_att(output, input2, input2)
        output = output + self.dropout2(attn_output)
        output = self.norm2(output)

        visual = output.mean(0)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm3(output)

        #############################
        logit1 = (text @ visual.T) / 0.5
        logit2 = (visual @ text.T) / 0.5
        logit = (logit1 + logit2) / 2
        logpt = F.log_softmax(logit, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        ############################

        return output, cross_weight, sim_loss


class DynamicCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.model.transformer_decoder.vocab_size
        self.word_embed_size = cfg.model.transformer_decoder.word_dim
        self.hidden_size = cfg.model.transformer_decoder.att_dim
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_size, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(self.word_embed_size, self.hidden_size),
            nn.Dropout(0.1))
        self.embed_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.n_head = cfg.model.transformer_decoder.att_head

        self.position_enc = PositionEncoding(n_filters=self.word_embed_size,
                                                    max_len=cfg.model.transformer_decoder.seq_length)



        self.num_hidden_layers = cfg.model.transformer_decoder.att_layer
        self.layer = nn.ModuleList([CrossTransformer(self.hidden_size, self.n_head)
                                    for _ in range(self.num_hidden_layers)])

    def forward(self, seq, dec_mask, diff_bef, diff_aft,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        dec_hidden_states = self.position_enc(self.embed(seq))
        dec_hidden_states = self.fc(dec_hidden_states)
        dec_hidden_states = dec_hidden_states.transpose(0,1)
        enc_outputs = torch.cat([diff_bef, diff_aft], -1)
        enc_outputs = self.embed_fc(enc_outputs).transpose(0, 1)
        all_encoder_layers = []
        all_att_vis = []
        all_sim_loss = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states, attention_weight, sim_loss = layer_module(
                dec_hidden_states, enc_outputs, dec_mask, diagonal_mask=diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
                all_att_vis.append(attention_weight)
                all_sim_loss.append(sim_loss)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
            all_att_vis.append(attention_weight)
            all_sim_loss.append(sim_loss)
        return all_encoder_layers[-1].transpose(0, 1), all_att_vis[-1], all_sim_loss[-1]



class CCR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.model.transformer_decoder.vocab_size
        self.word_embed_size = cfg.model.transformer_decoder.word_dim
        self.hidden_size = cfg.model.transformer_decoder.att_dim
        self.label_smoothing = cfg.model.transformer_decoder.label_smoothing

        self.seq_length = cfg.model.transformer_decoder.seq_length

        self.core = DynamicCore(cfg)

        self.share_wd_cls_weight = cfg.model.transformer_decoder.share_wd_cls_weight


        if self.share_wd_cls_weight:
            logit_weight = self.core.embed.weight

            self.logit = LMPredictionHead(cfg, logit_weight)
        else:
            self.logit = nn.Linear(self.hidden_size, self.vocab_size)


        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        # self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def _forward(self,
                 diff_bef, diff_aft, seq, mask, labels_with_ignore=None):


        decoder_outputs, attention_weight, sim_loss = self.core(
            seq, mask, diff_bef, diff_aft, diagonal_mask=True)  # [:,1:,:]  # (N, Lt, D)
        prediction_scores = self.logit(decoder_outputs)
        caption_loss = 0.
        if labels_with_ignore is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.vocab_size),
                                          labels_with_ignore.view(-1))
        return caption_loss, attention_weight, sim_loss

    def sample(
            self,
            diff_bef, diff_aft, start_idx=2, unk_idx=1, sample_max=0):
        """The first few args are the same to the input to the forward_step func
        Note:
            1, Copy the prev_ms each word generation step, as the func will modify this value,
            which will cause discrepancy between training and inference
            2, After finish the current sentence generation step, replace the words generated
            after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
            next memory state tensor.
        """

        bsz = diff_bef.size(0)
        max_cap_len = self.seq_length
        text_input_ids = diff_bef.new_zeros((bsz, max_cap_len), dtype=torch.long)
        text_masks = diff_bef.new_zeros(bsz, max_cap_len).float()  # zeros
        next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
        for dec_idx in range(max_cap_len):
            text_input_ids[:, dec_idx] = next_symbols
            text_masks[:, dec_idx] = 1
            decoder_outputs, attention_weight, sim_loss = self.core(text_input_ids, text_masks, diff_bef, diff_aft, diagonal_mask=True)
            pred_scores = F.log_softmax(self.logit(decoder_outputs), -1)
            # suppress unk token; (N, L, vocab_size)
            pred_scores[:, :, unk_idx] = -1e10
            if sample_max:
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words
            else:
                prob_prev = torch.exp(pred_scores[:, dec_idx])
                next_words = torch.multinomial(prob_prev, 1)
                next_symbols = next_words.view(-1).long()

            if dec_idx == 0:
                unfinished = next_symbols != 3 # 3 is the end sign
            else:
                unfinished = unfinished * (next_symbols != 3)
            next_symbols = next_symbols * unfinished.type_as(next_symbols)

            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return text_input_ids, attention_weight



