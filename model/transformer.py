import torch
import numpy as np
import torch.nn as nn
import math


# some code adapted from https://wmathor.com/index.php/archives/1455/


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_q]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)  # Linear only change the last dimension

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]  [512, 1, 5]  --> Spatial info
        input_K: [batch_size, len_k, d_model]  [512, 30, 5]  --> Temporal info
        input_V: [batch_size, len_v(=len_k), d_model]  [512, 30, 5]  --> Temporal info
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn  # All batch size dimensions are reserved.


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]  [512, 1, 5]  --> Spatial info
        enc_outputs: [batch_size, src_len, d_model]  [512, 30, 5]  --> Temporal info
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]  [512, 1, 5]
        enc_intpus: [batch_size, src_len, d_model]  [512, 30, 5]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = dec_inputs  # self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs


# d_model,   Embedding Size
# d_ff, FeedForward dimension
# d_k = d_v,   dimension of K(=Q), V
# n_layers,   number of Encoder of Decoder Layer
# n_heads,   number of heads in Multi-Head Attention

class Transformer2_3_1(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Transformer2_3_1, self).__init__()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q).cuda()
        self.decoder = Decoder(d_model, d_ff, d_k, d_v, 1, n_heads, len_q).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [Frames, src_len, d_model]  [512, 30, 5]
        dec_inputs: [Frames, 1, d_model]  [512, 1, 5]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)  # Self-attention for temporal features
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q, d_model=None):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q
        self.d_model = out_features if d_model is None else d_model

        self.spatial_encoder = EncoderLayer(self.d_model, mstcn_f_maps, mstcn_f_maps, mstcn_f_maps, 8, 5)
        self.transformer = Transformer2_3_1(d_model=self.d_model, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                            d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q=len_q)
        self.fc = nn.Linear(mstcn_f_dim, d_model, bias=False)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_model, out_features, bias=False)
        )


    def forward(self, x, long_feature):
        """
        Fusion of current temporal features and long-range spatial features
        :param x: Shifted frame-wise predictions  [1, 5, 512]
        :param long_feature: Long-range spatial features  [1, 512, 2048]
        :return: frame-wise predictions.
        """
        out_features = x.transpose(1, 2)  # [1, 512, 5]
        bs = out_features.size(0)  # 1
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input0 = torch.zeros((bs, self.len_q-1-i, self.d_model)).cuda()
                input0 = torch.cat([input0, out_features[:, 0:i+1]], dim=1)
            else:
                input0 = out_features[:, i-self.len_q+1:i+1]  # Collect all previous features
            inputs.append(input0)
        inputs = torch.stack(inputs, dim=0).squeeze(1)

        feas = torch.tanh(self.fc(long_feature))  # .transpose(0, 1))  # Project the input to desired dimension
        out_feas = []
        spa_len = 10
        for i in range(feas.size(1)):
            if i < spa_len - 1:
                input0 = torch.zeros((bs, spa_len - 1 - i, 32)).cuda()
                input0 = torch.cat([input0, feas[:, 0:i + 1]], dim=1)
            else:
                input0 = out_features[:, i - spa_len + 1:i + 1]  # Collect all previous features
            out_feas.append(input0)
        out_feas = torch.stack(out_feas, dim=0).squeeze(1)
        # inputs: [512, 30, 5],
        # feas: [512, 1, 5]  --> Spatial features
        out_feas, _ = self.spatial_encoder(out_feas)
        output = self.transformer(inputs, out_feas)  # Feature fusion between  temporal and spatial features
        output = self.out(output)
        return output[:, -1]
