from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    csv_path: str = "data/cse_cic_ids2018_processed.csv"
    label_col: str = "Label"
    timestamp_col: str = "Timestamp"
    protocol_col: str = "Protocol"
    # Columns that must NEVER be used as model inputs (Sec. 77, item 6 & 3).
    excluded_input_cols: List[str] = field(
        default_factory=lambda: [
            "Label",
            "Timestamp",
            "Src IP",
            "Dst IP",
            "Source IP",
            "Destination IP",
        ]
    )
    # Skew threshold for log1p transform (Sec. 7): x_min >= 0 and skew(x) > 1
    skew_threshold: float = 1.0
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15  # chronological, non-overlapping (Sec. 10/11)
    # Has the raw data got Src/Dst IP columns? Controls whether R4 (graph) is
    # even reachable. Defaults to False per Sec. 19 ("current processed CSV
    # does not contain Source IP or Destination IP").
    has_ip_columns: bool = False


@dataclass
class WindowConfig:
    window_seconds: float = 10.0  # Sec. 12, initial candidate = 10s
    candidate_window_seconds: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 30.0, 60.0]
    )
    max_flows_per_window: int = 512  # hard cap -> truncate + mask (Sec. 13/14)
    stride_seconds: Optional[float] = None  # None => non-overlapping windows


@dataclass
class FlowEncoderConfig:
    input_dim: int = 80  # 77 numeric (59 log1p'd) + 3 one-hot protocol dims
    hidden_dim: int = 128  # d = 128 (Sec. 15)
    dropout: float = 0.1
    time_variant: str = (
        "T2"  # "T1": relative-only, "T2": relative+global (Sec. 16)
    )
    time_embed_dim: int = 128  # combined additively with flow embedding


@dataclass
class StateConfig:
    variant: str = (
        "R2"  # R1 mean | R2 attention | R3 set-transformer | R4 graph | R5 multi-token
    )
    state_dim: int = 128
    num_tokens: int = 4  # only used by R5
    set_transformer_heads: int = 4
    set_transformer_layers: int = 2


@dataclass
class DynamicsConfig:
    variant: str = (
        "D2"  # D1 LSTM/GRU seq2seq | D2 Transformer-Encoder | D3 Transformer Enc-Dec
    )
    history_len: int = 20  # L, number of past states fed in
    horizon: int = 5  # K, initial candidate horizon (Sec. 30)
    candidate_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    forecast_mode: str = (
        "direct"  # "direct" (F2) | "autoregressive" (F3); trained direct, eval'd both (Sec. 29)
    )
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class RiskConfig:
    variant: str = (
        "RISK-2"  # RISK-1 per-step | RISK-2 horizon | RISK-3 current-state | RISK-4 history-only
    )
    hidden_dim: int = 128


@dataclass
class StageConfig:
    variant: str = "S2"  # S1 independent | S2 sequence decoder
    num_stages: int = (
        5  # placeholder count of ATT&CK-aligned stage buckets; NOT ground truth (Sec. 42)
    )
    hidden_dim: int = 128


@dataclass
class ThreatConfig:
    variant: str = "T5"  # T1 R only | T2 R+S | T3.. see Sec. 47 / Sec. 66
    use_surprise: bool = True
    use_stage_consistency: bool = True


@dataclass
class LossWeights:
    lambda_dyn: float = 1.0  # z-space forecasting loss
    lambda_state: float = 0.5  # decoded observable-state loss
    lambda_risk: float = 1.0
    lambda_stage: float = 0.5
    lambda_ssl: float = 0.0  # 0 unless a self-supervised phase is active


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs_phase1: int = (
        10  # representation (Stage O SSL or supervised pretrain of encoder+state)
    )
    epochs_phase2: int = 15  # dynamics
    epochs_phase3: int = 15  # risk/stage
    epochs_phase4: int = 10  # end-to-end joint fine-tune
    device: str = "cuda"  # falls back to "cpu" automatically if unavailable
    seed: int = 42


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    flow_encoder: FlowEncoderConfig = field(default_factory=FlowEncoderConfig)
    state: StateConfig = field(default_factory=StateConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    train: TrainConfig = field(default_factory=TrainConfig)
