import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning import seed_everything
from decoder.dataset import VocosDataModule
from decoder.experiment import WavTokenizer
from decoder.feature_extractors import EncodecFeatures
from decoder.models import VocosBackbone
from decoder.heads import ISTFTHead
from decoder.helpers import GradNormCallback
import subprocess
import webbrowser

# 実験の再現性のために乱数シードを固定
seed_everything(3407)

if __name__ == '__main__':

    # データモジュールの設定
    data_module = VocosDataModule(
        train_params={
            "filelist_path": r"C:\Users\user\Desktop\git\WavTokenizer\data\file_list.txt",  # 学習データのファイルリストパス
            "sampling_rate": 24000,      # サンプリングレート(Hz)
            "num_samples": 72000,        # 1つのオーディオサンプルの長さ（3秒分）
            "batch_size": 1,             # バッチサイズ
            "num_workers": 1             # データローダーのワーカー数
        },
        val_params={
            "filelist_path": r"C:\Users\user\Desktop\git\WavTokenizer\data\file_list.txt",  # 検証データのファイルリストパス
            "sampling_rate": 24000,      # サンプリングレート(Hz)
            "num_samples": 72000,        # 1つのオーディオサンプルの長さ（3秒分）
            "batch_size": 1,             # バッチサイズ
            "num_workers": 1             # データローダーのワーカー数
        }
    )

    # 特徴量抽出器の設定
    feature_extractor = EncodecFeatures(
        encodec_model="encodec_24khz",   # 使用するEncodecモデル
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
        sample_rate=24000,               # サンプリングレート(Hz)
        initial_learning_rate=2e-4,      # 初期学習率
        mel_loss_coeff=45,               # メルスペクトログラム損失の係数
        mrd_loss_coeff=1.0,              # Multi-Resolution Discriminator損失の係数
        num_warmup_steps=0,              # ウォームアップステップ数
        pretrain_mel_steps=0,            # メル事前学習のステップ数
        evaluate_utmos=True,             # UTMOSスコアを評価するかどうか
        evaluate_pesq=True,              # PESQスコアを評価するかどうか
        evaluate_periodicty=True,        # 周期性を評価するかどうか
        resume=False,                    # 学習を再開するかどうか
        resume_config = r"C:\Users\user\Desktop\git\WavTokenizer\configs\x.yaml",  # 再開用の設定ファイル
        resume_model = r"C:\Users\user\Desktop\git\WavTokenizer\models\x.ckpt",  # 再開用のモデルチェックポイント
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head
    )

    # トレーニング用コールバックの設定
    callbacks = [
        LearningRateMonitor(),           # 学習率の監視
        ModelSummary(max_depth=2),       # モデル構造のサマリー出力
        ModelCheckpoint(
            monitor="val_loss",          # 監視する評価指標
            dirpath="./result/models",
            filename="wavtokenizer_checkpoint_{epoch}_{step}_{val_loss:.4f}",  # チェックポイントのファイル名フォーマット
            save_top_k=10,               # 保存する上位k個のモデル数
            save_last=True,               # 最後のモデルを保存するかどうか
            every_n_epochs=1,        # エポックごとに保存
        ),
        GradNormCallback()               # 勾配正規化のコールバック
    ]

    # TensorBoard用ロガーの設定
    logger = TensorBoardLogger(
        save_dir="./result/train/test/"  # ログの保存ディレクトリ
    )

    # トレーナーの設定
    trainer = pl.Trainer(
        logger=logger,                   
        callbacks=callbacks,              
        max_steps=20000000,             # 最大トレーニングステップ数
        limit_val_batches=100,          # 検証時のバッチ数制限
        accelerator="gpu",              # 使用するアクセラレータ（GPU）
        devices=1,                      # 使用するGPUの数
        log_every_n_steps=1          # ログを出力するステップ間隔
    )

    # TensorBoardをバックグラウンドで起動
    log_dir = "./result/train/test/"
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir])

    # ブラウザでTensorBoardを開く
    webbrowser.open("http://localhost:6006")

    # モデルの学習を開始
    trainer.fit(model, data_module)

    # トレーニング終了後、TensorBoardのプロセスを終了
    tensorboard_process.terminate()
