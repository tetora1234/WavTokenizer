import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from decoder.dataset import VocosDataModule
from decoder.experiment import WavTokenizer
from decoder.feature_extractors import EncodecFeatures
from decoder.models import VocosBackbone
from decoder.heads import ISTFTHead
from decoder.helpers import GradNormCallback
import subprocess
import webbrowser

if __name__ == '__main__':

    # データモジュールの設定
    data_module = VocosDataModule(
        train_params={
            "filelist_path": r"C:\Users\user\Desktop\git\WavTokenizer\data\file_list.txt",  # 学習データのファイルリストパス
            "sampling_rate": 48000,      # サンプリングレート(Hz)
            "num_samples": 144000,        # 1つのオーディオサンプルの長さ（3秒分）
            "batch_size": 6,             # バッチサイズ
            "num_workers": 1             # データローダーのワーカー数
        },
        val_params={
            "filelist_path": r"C:\Users\user\Desktop\git\WavTokenizer\data\file_list.txt",  # 検証データのファイルリストパス
            "sampling_rate": 48000,      # サンプリングレート(Hz)
            "num_samples": 144000,        # 1つのオーディオサンプルの長さ（3秒分）
            "batch_size": 6,             # バッチサイズ
            "num_workers": 1             # データローダーのワーカー数
        }
    )

    # 特徴量抽出器の設定
    feature_extractor = EncodecFeatures(
        encodec_model="encodec_48khz",   # 使用するEncodecモデル
        bandwidths=[6.6, 6.6, 6.6, 6.6], # 各レイヤーの帯域幅
        train_codebooks=True,            # コードブックを学習させるかどうか
        num_quantizers=1,                # 量子化器の数
        dowmsamples=[8, 5, 4, 2],        # 各レイヤーのダウンサンプリング率
        vq_bins=4096,                    # Vector Quantizationのビン数
        vq_kmeans=200                    # k-meansクラスタリングの数
    )

    # バックボーンネットワークの設定
    backbone = VocosBackbone(
        input_channels=512,              # 入力チャンネル数
        dim=768,                         # モデルの基本次元数
        intermediate_dim=2304,           # 中間層の次元数
        num_layers=12,                   # トランスフォーマーレイヤー数
        adanorm_num_embeddings=4         # AdaNormの埋め込み数
    )

    # 出力ヘッドの設定
    head = ISTFTHead(
        dim=768,                         # 入力次元数
        n_fft=1280,                      # FFTサイズ
        hop_length=320,                  # ホップ長（フレーム間の移動量）
        padding="same"                   # パディングタイプ
    )

    # モデル全体の設定
    model = WavTokenizer(
        sample_rate=48000,               # サンプリングレート(Hz)
        initial_learning_rate=0.0001,      # 初期学習率
        mel_loss_coeff=45,               # メルスペクトログラム損失の係数
        mrd_loss_coeff=1.0,              # Multi-Resolution Discriminator損失の係数
        num_warmup_steps=0,              # ウォームアップステップ数
        pretrain_mel_steps=0,            # メル事前学習のステップ数
        evaluate_utmos=True,             # UTMOSスコアを評価するかどうか
        evaluate_pesq=True,              # PESQスコアを評価するかどうか
        evaluate_periodicty=True,        # 周期性を評価するかどうか
        resume=False,                    # 学習を再開するかどうか
        resume_config = r"C:\Users\user\Desktop\git\WavTokenizer\configs\48hz.yaml",  # 再開用の設定ファイル
        resume_model = r"C:\Users\user\Desktop\git\WavTokenizer\models\last.ckpt",  # 再開用のモデルチェックポイント
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head
    )

    # 現在の時刻を取得して、hhmmss形式でフォルダ名に追加
    current_time = datetime.datetime.now().strftime("%H%M%S")
    output_dir = f"./result/train/{current_time}/"  # hhmmss形式の時刻をフォルダ名に組み込む

    # トレーニング用コールバックの設定
    callbacks = [
        LearningRateMonitor(),           # 学習率の監視
        ModelSummary(max_depth=2),       # モデル構造のサマリー出力
        ModelCheckpoint(
            monitor="val_loss",          # 監視する評価指標
            dirpath=f"{output_dir}/models",
            filename="checkpoint_{epoch}_{step}_{val_loss:.4f}",  # チェックポイントのファイル名フォーマット
            save_top_k=10,               # 保存する上位k個のモデル数
            save_last=True,              # 最後のモデルを保存するかどうか
            every_n_epochs=1,            # エポックごとに保存
        ),
        GradNormCallback()               # 勾配正規化のコールバック
    ]

    # TensorBoard用ロガーの設定
    logger = TensorBoardLogger(
        save_dir=output_dir  # フォルダ名に時刻が追加されたものを指定
    )

    # トレーナーの設定
    trainer = pl.Trainer(
        logger=logger,                   
        callbacks=callbacks,              
        max_steps=20000000,             # 最大トレーニングステップ数
        limit_val_batches=1,           # 検証時のバッチ数制限
        accelerator="gpu",              # 使用するアクセラレータ（GPU）
        gpus = [0],                      # 使用するGPUの数
        log_every_n_steps=1             # ログを出力するステップ間隔
    )

    # TensorBoardをバックグラウンドで起動
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", output_dir])

    # ブラウザでTensorBoardを開く
    webbrowser.open("http://localhost:6006")

    # モデルの学習を開始
    trainer.fit(model, data_module)

    # トレーニング終了後、TensorBoardのプロセスを終了
    tensorboard_process.terminate()
