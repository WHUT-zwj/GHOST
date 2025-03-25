# layers/GatingMechanism.py

import torch
import torch.nn as nn


class SentimentGating(nn.Module):
    """Sentiment gating mechanism module"""

    def __init__(self, d_model, sentiment_dim, dropout=0.1):
        super(SentimentGating, self).__init__()
        self.sentiment_embedding = nn.Linear(sentiment_dim, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sentiment):
        # x: [batch_size, seq_len, num_stocks, d_model]
        # sentiment: [batch_size, seq_len, num_stocks, sentiment_dim]

        # Sentiment feature embedding
        sentiment_emb = self.sentiment_embedding(sentiment)  # [batch_size, seq_len, num_stocks, d_model]

        # Calculate gating weights
        gate_weights = torch.sigmoid(self.gate(sentiment_emb))

        # Apply gating
        gated_output = x * gate_weights

        # Add residual connection and layer normalization
        gated_output = self.layer_norm(gated_output + x)

        return gated_output, gate_weights


class HierarchicalSentimentGating(nn.Module):
    """Hierarchical sentiment gating mechanism"""

    def __init__(self, d_model, sentiment_dim, num_levels=2, dropout=0.1):
        super(HierarchicalSentimentGating, self).__init__()
        self.levels = nn.ModuleList([
            SentimentGating(d_model, sentiment_dim if i == 0 else d_model, dropout)
            for i in range(num_levels)
        ])

    def forward(self, x, sentiment):
        gate_weights_list = []
        current_x = x

        for level in self.levels:
            current_x, gate_weights = level(current_x,
                                            sentiment if len(gate_weights_list) == 0
                                            else current_x)
            gate_weights_list.append(gate_weights)

        return current_x, gate_weights_list