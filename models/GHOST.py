import torch
import torch.nn as nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding, DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.GatingMechanism import HierarchicalSentimentGating  # Import gating mechanism


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.num_stocks = configs.num_stocks
        self.d_model = configs.d_model
        self.num_features = configs.enc_in
        self.output_attention = configs.output_attention

        # Embedding for Mamba model
        self.enc_embedding = DataEmbedding(
            self.num_features, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Add sentiment gating mechanism
        self.sentiment_gating = HierarchicalSentimentGating(
            d_model=configs.d_model,
            sentiment_dim=configs.sentiment_dim,  # Need to add in configs
            num_levels=configs.gating_levels,     # Need to add in configs
            dropout=configs.dropout
        )


        # Mamba model
        self.mamba = Mamba(
            d_model=self.d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Here we use a linear layer to replace the embedding layer
        self.linear_embedding = nn.Linear(self.seq_len * self.d_model, self.d_model)

        # Encoder for iTransformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.projection = nn.Linear(configs.d_model, self.pred_len * self.num_features, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, sentiment_features):
        batch_size, seq_len, num_stocks, num_features = x_enc.shape
        x_mark_enc = x_mark_enc.unsqueeze(2).expand(-1, -1, num_stocks, -1)

        # Calculate mean and standard deviation
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        # Embedding for Mamba
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [Batch, seq_len, num_stocks, d_model]

        # Apply sentiment gating mechanism
        gated_enc_out, gate_weights = self.sentiment_gating(enc_out, sentiment_features)

        # Reshape for Mamba processing
        gated_enc_out = gated_enc_out.reshape(batch_size * num_stocks, seq_len, self.d_model)

        # Mamba processing
        mamba_out = self.mamba(gated_enc_out)
        mamba_out = mamba_out.view(batch_size, num_stocks, seq_len * self.d_model)

        # Linear embedding layer
        emb_out = self.linear_embedding(mamba_out)

        # Encoder
        enc_out, attns = self.encoder(emb_out, attn_mask=None)

        # Projection layer
        dec_out = self.projection(enc_out)
        dec_out = dec_out.view(batch_size, num_stocks, self.pred_len, self.num_features)
        dec_out = dec_out.permute(0, 2, 1, 3)

        # Denormalization
        means = means.expand(-1, self.pred_len, -1, -1)
        stdev = stdev.expand(-1, self.pred_len, -1, -1)
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, gate_weights, attns
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, sentiment_features):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.output_attention:
                dec_out, gate_weights, attns = self.forecast(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, sentiment_features)
                return dec_out, gate_weights, attns
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, sentiment_features)
            return dec_out
        else:
            raise NotImplementedError("Only forecasting task is implemented")