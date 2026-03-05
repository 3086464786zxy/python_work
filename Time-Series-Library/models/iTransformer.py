#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from layers.Transformer_EncDec import Encoder, EncoderLayer
#from layers.SelfAttention_Family import FullAttention, AttentionLayer
#from layers.Embed import DataEmbedding_inverted
#import numpy as np
#
#
#class Model(nn.Module):
#    """
#    Paper link: https://arxiv.org/abs/2310.06625
#    """
#
#    def __init__(self, configs):
#        super(Model, self).__init__()
#        #新增
#        # ===================
#        self.configs = configs
#        self.enc_in = configs.enc_in
#        self.c_out = configs.c_out
#        # =================
#
#        self.task_name = configs.task_name
#        self.seq_len = configs.seq_len
#        self.pred_len = configs.pred_len
#        # Embedding
#        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                    configs.dropout)
#        # Encoder
#        self.encoder = Encoder(
#            [
#                EncoderLayer(
#                    AttentionLayer(
#                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                      output_attention=False), configs.d_model, configs.n_heads),
#                    configs.d_model,
#                    configs.d_ff,
#                    dropout=configs.dropout,
#                    activation=configs.activation
#                ) for l in range(configs.e_layers)
#            ],
#            norm_layer=torch.nn.LayerNorm(configs.d_model)
#        )
#        # Decoder
#        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
#        if self.task_name == 'imputation':
#            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#        if self.task_name == 'anomaly_detection':
#            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#        if self.task_name == 'classification':
#            self.act = F.gelu
#            self.dropout = nn.Dropout(configs.dropout)
#            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
#
#    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#        # Normalization from Non-stationary Transformer
#        means = x_enc.mean(1, keepdim=True).detach()
#        x_enc = x_enc - means
#        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#        x_enc /= stdev
#
#        _, _, N = x_enc.shape
#
#        # Embedding
#        enc_out = self.enc_embedding(x_enc, x_mark_enc)
#        enc_out, attns = self.encoder(enc_out, attn_mask=None)
#
#        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#        # De-Normalization from Non-stationary Transformer
#        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#        return dec_out
#
#
#    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
#        # Normalization from Non-stationary Transformer
#        means = x_enc.mean(1, keepdim=True).detach()
#        x_enc = x_enc - means
#        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#        x_enc /= stdev
#
#        _, L, N = x_enc.shape
#
#        # Embedding
#        enc_out = self.enc_embedding(x_enc, x_mark_enc)
#        enc_out, attns = self.encoder(enc_out, attn_mask=None)
#
#        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#        # De-Normalization from Non-stationary Transformer
#        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#        return dec_out
#
#    def anomaly_detection(self, x_enc):
#        # Normalization from Non-stationary Transformer
#        means = x_enc.mean(1, keepdim=True).detach()
#        x_enc = x_enc - means
#        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#        x_enc /= stdev
#
#        _, L, N = x_enc.shape
#
#        # Embedding
#        enc_out = self.enc_embedding(x_enc, None)
#        enc_out, attns = self.encoder(enc_out, attn_mask=None)
#
#        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#        # De-Normalization from Non-stationary Transformer
#        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#        return dec_out
#
#    def classification(self, x_enc, x_mark_enc):
#        # Embedding
#        enc_out = self.enc_embedding(x_enc, None)
#        enc_out, attns = self.encoder(enc_out, attn_mask=None)
#
#        # Output
#        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
#        output = self.dropout(output)
#        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
#        output = self.projection(output)  # (batch_size, num_classes)
#        return output
#
#    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#        if self.task_name == 'imputation':
#            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
#            return dec_out  # [B, L, D]
#        if self.task_name == 'anomaly_detection':
#            dec_out = self.anomaly_detection(x_enc)
#            return dec_out  # [B, L, D]
#        if self.task_name == 'classification':
#            dec_out = self.classification(x_enc, x_mark_enc)
#            return dec_out  # [B, N]
#        return None





#第一次修改
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


# ===============================
# ⭐ 频率增强信道注意力模块
# ===============================
class FrequencyEnhancedChannelAttention(nn.Module):
    def __init__(self, seq_len, channels, reduction=16):
        super(FrequencyEnhancedChannelAttention, self).__init__()

        self.seq_len = seq_len
        self.channels = channels

        hidden = max(1, channels // reduction)

        # 通道注意力 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, L, N]
        """

        # ===== 1️⃣ 频率增强（沿时间维）=====
        fft_x = torch.fft.rfft(x, dim=1)  # [B, L//2+1, N]

        magnitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 构造高频增强权重
        freq_len = magnitude.shape[1]
        freq_weight = torch.linspace(
            0, 1, freq_len, device=x.device
        ).unsqueeze(0).unsqueeze(-1)  # [1, F, 1]

        # 高频增强
        magnitude = magnitude * (1 + freq_weight)

        # 复数重建
        enhanced_fft = magnitude * torch.exp(1j * phase)

        x_enhanced = torch.fft.irfft(
            enhanced_fft,
            n=self.seq_len,
            dim=1
        )

        # ===== 2️⃣ 通道注意力 =====
        channel_stat = x_enhanced.mean(dim=1)  # [B, N]

        attention = self.mlp(channel_stat)  # [B, N]

        attention = attention.unsqueeze(1)  # [B,1,N]

        out = x_enhanced * attention

        return out


# ===============================
# ⭐ iTransformer 主模型
# ===============================
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ⭐ 新增：频率增强信道注意力
        self.freq_channel_attn = FrequencyEnhancedChannelAttention(
            seq_len=configs.seq_len,
            channels=configs.enc_in
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder / Projection
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len)

        if self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.seq_len)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.enc_in,
                configs.num_class
            )

    # ==================================
    # Forecast
    # ==================================
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        _, _, N = x_enc.shape

        # ⭐ 加入频率增强信道注意力
        x_enc = self.freq_channel_attn(x_enc)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-normalization
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )

        return dec_out

    # ==================================
    # Imputation
    # ==================================
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        _, L, N = x_enc.shape

        # ⭐ 加入频率增强信道注意力
        x_enc = self.freq_channel_attn(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        )

        return dec_out

    # ==================================
    # Anomaly Detection
    # ==================================
    def anomaly_detection(self, x_enc):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        _, L, N = x_enc.shape

        # ⭐ 加入频率增强信道注意力
        x_enc = self.freq_channel_attn(x_enc)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        )

        return dec_out

    # ==================================
    # Classification
    # ==================================
    def classification(self, x_enc, x_mark_enc):

        # ⭐ 加入频率增强信道注意力
        x_enc = self.freq_channel_attn(x_enc)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)

        output = output.reshape(output.shape[0], -1)

        output = self.projection(output)

        return output

    # ==================================
    # Forward
    # ==================================
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec
            )
            return dec_out[:, -self.pred_len:, :]

        if self.task_name == 'imputation':
            return self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask
            )

        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)

        return None

