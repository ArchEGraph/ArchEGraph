from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, GATv2Conv, TransformerConv
from torch_geometric.utils import to_dense_batch


class WeatherEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, w):
        return self.net(w)


class FaceEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x_geom = x[:, :-1]
        is_transparent = x[:, -1:]
        return self.net(x_geom), is_transparent


class SpaceEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentionWeights(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.attention(x))


class FaceToSpaceConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, weather_dim):
        super().__init__(aggr="add", flow="source_to_target")

        in_dim = hidden_dim + edge_dim + weather_dim

        self.mlp_opaque = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mlp_trans = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attention = AttentionWeights(in_dim)

    def forward(self, x_face, is_transparent, x_space, edge_index, edge_attr, weather_space):
        self._edge_trans = is_transparent[edge_index[0]].detach()
        return self.propagate(
            edge_index=edge_index,
            x=(x_face, x_space),
            edge_attr=edge_attr,
            weather_space=weather_space,
        )

    def message(self, x_j, edge_attr, weather_space_i):
        is_ext = edge_attr[:, 0:1]
        weather_msg = weather_space_i * is_ext

        msg_input = torch.cat([x_j, edge_attr, weather_msg], dim=-1)
        msg_o = self.mlp_opaque(msg_input)
        msg_t = self.mlp_trans(msg_input)

        msg = msg_t * self._edge_trans + msg_o * (1.0 - self._edge_trans)
        msg = msg * self.attention(msg_input)
        return msg


class WeatherEncoderOriginal(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, w):
        return self.net(w)


class FaceEncoderOriginal(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x_geom = x[:, :-1]
        is_transparent = x[:, -1:]
        return self.net(x_geom), is_transparent


class SpaceEncoderOriginal(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class FaceToSpaceConvOriginal(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, weather_dim):
        super().__init__(aggr="add", flow="source_to_target")

        in_dim = hidden_dim + edge_dim + weather_dim

        self.mlp_opaque = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mlp_trans = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_face, is_transparent, x_space, edge_index, edge_attr, weather_space):
        self._edge_trans = is_transparent[edge_index[0]].detach()
        return self.propagate(
            edge_index=edge_index,
            x=(x_face, x_space),
            edge_attr=edge_attr,
            weather_space=weather_space,
        )

    def message(self, x_j, edge_attr, weather_space_i):
        is_ext = edge_attr[:, 0:1]
        weather_msg = weather_space_i * is_ext

        msg_input = torch.cat([x_j, edge_attr, weather_msg], dim=-1)
        msg_o = self.mlp_opaque(msg_input)
        msg_t = self.mlp_trans(msg_input)

        return msg_t * self._edge_trans + msg_o * (1.0 - self._edge_trans)


class F2S(nn.Module):
    def __init__(self, hidden_dim=32, edge_dim=1, weather_dim=16, layers=3, weather_in_dim=9):
        super().__init__()

        self.face_enc = FaceEncoderOriginal(hidden_dim)
        self.space_enc = SpaceEncoderOriginal(hidden_dim)

        self.face_type = nn.Embedding(1, hidden_dim)
        self.space_type = nn.Embedding(1, hidden_dim)

        self.weather_enc = WeatherEncoderOriginal(weather_in_dim, weather_dim)

        self.convs = nn.ModuleList([FaceToSpaceConvOriginal(hidden_dim, edge_dim, weather_dim) for _ in range(layers)])

        self.temporal = nn.GRU(
            input_size=hidden_dim + weather_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Batch):
        weather_seq = data.weather
        if weather_seq.dim() != 3:
            raise ValueError(
                f"F2S expects weather tensor [B, 8760, F], got {tuple(weather_seq.shape)}."
            )

        bsz, seq_len, weather_in = weather_seq.shape
        if seq_len != 8760:
            raise ValueError(f"F2S expects sequence length 8760, got {seq_len}.")

        x_face, is_transparent = self.face_enc(data["face"].x)
        x_space = self.space_enc(data["space"].x)

        x_face = x_face + self.face_type(torch.zeros(x_face.size(0), dtype=torch.long, device=x_face.device))
        x_space = x_space + self.space_type(torch.zeros(x_space.size(0), dtype=torch.long, device=x_space.device))

        weather_flat = weather_seq.reshape(bsz * seq_len, weather_in)
        weather_emb = self.weather_enc(weather_flat).reshape(bsz, seq_len, -1)

        space_batch = data["space"].batch
        weather_year = weather_emb.mean(dim=1)
        weather_space = weather_year[space_batch]

        edge_index = data["face", "to", "space"].edge_index
        edge_attr = data["face", "to", "space"].edge_attr

        for conv in self.convs:
            delta = conv(
                x_face=x_face,
                is_transparent=is_transparent,
                x_space=x_space,
                edge_index=edge_index,
                edge_attr=edge_attr,
                weather_space=weather_space,
            )
            x_space = x_space + delta

        space_seq = x_space.unsqueeze(1).expand(-1, seq_len, -1)
        weather_space_seq = weather_emb[space_batch]
        temporal_input = torch.cat([space_seq, weather_space_seq], dim=-1)
        temporal_out, _ = self.temporal(temporal_input)
        return self.energy_head(temporal_out).squeeze(-1)

class F2SAttr(nn.Module):
    def __init__(self, hidden_dim=64, edge_dim=1, weather_dim=32, layers=7, weather_in_dim=9):
        super().__init__()

        self.face_enc = FaceEncoder(hidden_dim)
        self.space_enc = SpaceEncoder(hidden_dim)

        self.face_type = nn.Embedding(1, hidden_dim)
        self.space_type = nn.Embedding(1, hidden_dim)

        self.weather_enc = WeatherEncoder(weather_in_dim, weather_dim)

        self.convs = nn.ModuleList([FaceToSpaceConv(hidden_dim, edge_dim, weather_dim) for _ in range(layers)])

        self.temporal = nn.GRU(
            input_size=hidden_dim + weather_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Batch):
        x_face, is_transparent = self.face_enc(data["face"].x)
        x_space_base = self.space_enc(data["space"].x)

        x_face = x_face + self.face_type(torch.zeros(x_face.size(0), dtype=torch.long, device=x_face.device))
        x_space_base = x_space_base + self.space_type(
            torch.zeros(x_space_base.size(0), dtype=torch.long, device=x_space_base.device)
        )

        weather_seq = data.weather
        if weather_seq.dim() == 2:
            weather_seq = weather_seq.unsqueeze(1)

        bsz, seq_len, weather_in = weather_seq.shape
        weather_flat = weather_seq.reshape(bsz * seq_len, weather_in)
        weather_emb = self.weather_enc(weather_flat).reshape(bsz, seq_len, -1)

        edge_index = data["face", "to", "space"].edge_index
        edge_attr = data["face", "to", "space"].edge_attr
        space_batch = data["space"].batch

        weather_year = weather_emb.mean(dim=1)
        weather_space_year = weather_year[space_batch]

        x_space = x_space_base
        for conv in self.convs:
            delta = conv(
                x_face=x_face,
                is_transparent=is_transparent,
                x_space=x_space,
                edge_index=edge_index,
                edge_attr=edge_attr,
                weather_space=weather_space_year,
            )
            x_space = x_space + delta

        space_seq = x_space.unsqueeze(1).expand(-1, seq_len, -1)
        weather_space_seq = weather_emb[space_batch]
        temporal_input = torch.cat([space_seq, weather_space_seq], dim=-1)
        temporal_out, _ = self.temporal(temporal_input)
        return self.energy_head(temporal_out).squeeze(-1)


class GATv2(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        edge_dim=1,
        weather_dim=32,
        layers=6,
        weather_in_dim=9,
        heads=4,
    ):
        super().__init__()

        self.face_enc = FaceEncoder(hidden_dim)
        self.space_enc = SpaceEncoder(hidden_dim)
        self.weather_enc = WeatherEncoder(weather_in_dim, weather_dim)

        self.face_type = nn.Embedding(1, hidden_dim)
        self.space_type = nn.Embedding(1, hidden_dim)

        self.weather_to_space = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=0.1,
                    add_self_loops=False,
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

        self.temporal = nn.GRU(
            input_size=hidden_dim + weather_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Batch):
        x_face, _ = self.face_enc(data["face"].x)
        x_space = self.space_enc(data["space"].x)

        x_face = x_face + self.face_type(torch.zeros(x_face.size(0), dtype=torch.long, device=x_face.device))
        x_space = x_space + self.space_type(torch.zeros(x_space.size(0), dtype=torch.long, device=x_space.device))

        weather_seq = data.weather
        if weather_seq.dim() == 2:
            weather_seq = weather_seq.unsqueeze(1)

        bsz, seq_len, weather_in = weather_seq.shape
        weather_flat = weather_seq.reshape(bsz * seq_len, weather_in)
        weather_emb = self.weather_enc(weather_flat).reshape(bsz, seq_len, -1)

        edge_index = data["face", "to", "space"].edge_index
        edge_attr = data["face", "to", "space"].edge_attr
        space_batch = data["space"].batch

        weather_year = weather_emb.mean(dim=1)
        weather_space_year = weather_year[space_batch]
        x_space = x_space + self.weather_to_space(weather_space_year)

        for conv, norm in zip(self.convs, self.norms):
            delta = conv((x_face, x_space), edge_index, edge_attr)
            x_space = norm(x_space + delta)

        space_seq = x_space.unsqueeze(1).expand(-1, seq_len, -1)
        weather_space_seq = weather_emb[space_batch]
        temporal_input = torch.cat([space_seq, weather_space_seq], dim=-1)
        temporal_out, _ = self.temporal(temporal_input)
        return self.energy_head(temporal_out).squeeze(-1)


class TransformerConv(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        edge_dim=1,
        weather_dim=32,
        layers=6,
        weather_in_dim=9,
        heads=4,
    ):
        super().__init__()

        self.face_enc = FaceEncoder(hidden_dim)
        self.space_enc = SpaceEncoder(hidden_dim)
        self.weather_enc = WeatherEncoder(weather_in_dim, weather_dim)

        self.face_type = nn.Embedding(1, hidden_dim)
        self.space_type = nn.Embedding(1, hidden_dim)

        self.weather_to_space = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=0.1,
                    beta=True,
                )
                for _ in range(layers)
            ]
        )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(layers)
            ]
        )

        self.temporal = nn.GRU(
            input_size=hidden_dim + weather_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Batch):
        x_face, _ = self.face_enc(data["face"].x)
        x_space = self.space_enc(data["space"].x)

        x_face = x_face + self.face_type(torch.zeros(x_face.size(0), dtype=torch.long, device=x_face.device))
        x_space = x_space + self.space_type(torch.zeros(x_space.size(0), dtype=torch.long, device=x_space.device))

        weather_seq = data.weather
        if weather_seq.dim() == 2:
            weather_seq = weather_seq.unsqueeze(1)

        bsz, seq_len, weather_in = weather_seq.shape
        weather_flat = weather_seq.reshape(bsz * seq_len, weather_in)
        weather_emb = self.weather_enc(weather_flat).reshape(bsz, seq_len, -1)

        edge_index = data["face", "to", "space"].edge_index
        edge_attr = data["face", "to", "space"].edge_attr
        space_batch = data["space"].batch

        weather_year = weather_emb.mean(dim=1)
        weather_space_year = weather_year[space_batch]
        x_space = x_space + self.weather_to_space(weather_space_year)

        for conv, norm, ffn in zip(self.convs, self.norms, self.ffns):
            delta = conv((x_face, x_space), edge_index, edge_attr)
            x_space = norm(x_space + delta)
            x_space = x_space + ffn(x_space)

        space_seq = x_space.unsqueeze(1).expand(-1, seq_len, -1)
        weather_space_seq = weather_emb[space_batch]
        temporal_input = torch.cat([space_seq, weather_space_seq], dim=-1)
        temporal_out, _ = self.temporal(temporal_input)
        return self.energy_head(temporal_out).squeeze(-1)


class _GPSBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim, heads=4):
        super().__init__()

        self.local_conv = GATv2Conv(
            (hidden_dim, hidden_dim),
            hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=0.1,
            add_self_loops=False,
        )
        self.local_norm = nn.LayerNorm(hidden_dim)

        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=0.1,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_face, x_space, edge_index, edge_attr, space_batch):
        local_delta = self.local_conv((x_face, x_space), edge_index, edge_attr)
        x_space = self.local_norm(x_space + local_delta)

        dense_x, dense_mask = to_dense_batch(x_space, space_batch)
        attn_out, _ = self.global_attn(
            dense_x,
            dense_x,
            dense_x,
            key_padding_mask=~dense_mask,
            need_weights=False,
        )
        x_space_global = attn_out[dense_mask]
        x_space = self.global_norm(x_space + x_space_global)

        x_space = self.ffn_norm(x_space + self.ffn(x_space))
        return x_space


class GraphGPS(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        edge_dim=1,
        weather_dim=32,
        layers=4,
        weather_in_dim=9,
        heads=4,
    ):
        super().__init__()

        self.face_enc = FaceEncoder(hidden_dim)
        self.space_enc = SpaceEncoder(hidden_dim)
        self.weather_enc = WeatherEncoder(weather_in_dim, weather_dim)

        self.face_type = nn.Embedding(1, hidden_dim)
        self.space_type = nn.Embedding(1, hidden_dim)

        self.weather_to_space = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([_GPSBlock(hidden_dim, edge_dim=edge_dim, heads=heads) for _ in range(layers)])

        self.temporal = nn.GRU(
            input_size=hidden_dim + weather_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Batch):
        x_face, _ = self.face_enc(data["face"].x)
        x_space = self.space_enc(data["space"].x)

        x_face = x_face + self.face_type(torch.zeros(x_face.size(0), dtype=torch.long, device=x_face.device))
        x_space = x_space + self.space_type(torch.zeros(x_space.size(0), dtype=torch.long, device=x_space.device))

        weather_seq = data.weather
        if weather_seq.dim() == 2:
            weather_seq = weather_seq.unsqueeze(1)

        bsz, seq_len, weather_in = weather_seq.shape
        weather_flat = weather_seq.reshape(bsz * seq_len, weather_in)
        weather_emb = self.weather_enc(weather_flat).reshape(bsz, seq_len, -1)

        edge_index = data["face", "to", "space"].edge_index
        edge_attr = data["face", "to", "space"].edge_attr
        space_batch = data["space"].batch

        weather_year = weather_emb.mean(dim=1)
        weather_space_year = weather_year[space_batch]
        x_space = x_space + self.weather_to_space(weather_space_year)

        for block in self.blocks:
            x_space = block(x_face, x_space, edge_index, edge_attr, space_batch)

        space_seq = x_space.unsqueeze(1).expand(-1, seq_len, -1)
        weather_space_seq = weather_emb[space_batch]
        temporal_input = torch.cat([space_seq, weather_space_seq], dim=-1)
        temporal_out, _ = self.temporal(temporal_input)
        return self.energy_head(temporal_out).squeeze(-1)


class WeatherMLP(nn.Module):
    def __init__(self, weather_dim=6, hidden_dim=32):
        super().__init__()

        self.weather_enc = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Batch):
        weather = data.weather
        space_batch = data["space"].batch

        bsz, seq_len, feat_dim = weather.shape
        flat_weather = weather.reshape(bsz * seq_len, feat_dim)
        weather_emb = self.weather_enc(flat_weather).reshape(bsz, seq_len, -1)
        weather_space = weather_emb[space_batch]
        return self.energy_head(weather_space).squeeze(-1)




def build_model(model_name: str, weather_input_dim: int, device: torch.device):
    if model_name == "F2S":
        model = F2S(weather_in_dim=weather_input_dim)
    elif model_name == "F2SAttr":
        model = F2SAttr(weather_in_dim=weather_input_dim)
    elif model_name == "GATv2":
        model = GATv2(weather_in_dim=weather_input_dim)
    elif model_name == "TransformerConv":
        model = TransformerConv(weather_in_dim=weather_input_dim)
    elif model_name == "GraphGPS":
        model = GraphGPS(weather_in_dim=weather_input_dim)
    elif model_name == "WeatherMLP":
        model = WeatherMLP(weather_dim=weather_input_dim)

    else:
        raise ValueError(
            "Unsupported model name: "
            f"{model_name}. Expected one of: "
            "F2S, F2SAttr, "
            "GATv2, TransformerConv, GraphGPS, "
            "WeatherMLP"
        )
    return model.to(device)
