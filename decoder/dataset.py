from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

import soundfile

# PyTorchのスレッド数を1に設定
torch.set_num_threads(1)


@dataclass
class DataConfig:
    # 訓練・検証データの設定を保持するクラス
    filelist_path: str  # ファイルリストのパス
    sampling_rate: int  # サンプリングレート
    num_samples: int  # 音声データのサンプル数（切り出し長）
    batch_size: int  # バッチサイズ
    num_workers: int  # DataLoaderのワーカースレッド数


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        # 訓練と検証の設定を保持
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        # データセットとDataLoaderの作成
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        # 訓練用DataLoaderを返す
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        # 検証用DataLoaderを返す
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        # データセットの初期化
        # ファイルリストを読み込む
        with open(cfg['filelist_path']) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg['sampling_rate']  # サンプリングレート
        self.num_samples = cfg['num_samples']  # サンプル数
        self.train = train  # 訓練か検証か

    def __len__(self) -> int:
        # データセットの長さ（ファイルリストの数）
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        # データの取得
        audio_path = self.filelist[index]
        # 音声ファイルを読み込み
        y1, sr = soundfile.read(audio_path)
        y = torch.tensor(y1).float().unsqueeze(0)  # Tensor型に変換し、チャネル次元を追加
        if y.ndim > 2:
            # 多次元の場合、モノラルに変換
            y = y.mean(dim=-1, keepdim=False)
        # 訓練時はランダムに音量を調整
        gain = np.random.uniform(-1, -6) if self.train else -3
        if sr != self.sampling_rate:
            # サンプリングレートが異なる場合はリサンプリング
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            # サンプル数が足りない場合、パディングを追加
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            # 訓練時はランダムに切り出し
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # 検証時は常に最初の部分を使用（決定的にするため）
            y = y[:, : self.num_samples]

        return y[0]
