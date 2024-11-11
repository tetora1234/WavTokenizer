import os
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device = torch.device('cpu')

config_path = r"C:\Users\user\Desktop\git\WavTokenizer\configs\wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = r"C:\Users\user\Desktop\git\WavTokenizer\result\models\wavtokenizer_checkpoint_epoch=0_step=4_val_loss=8.0469.ckpt"
input_dir = r"C:\Users\user\Desktop\git\WavTokenizer\inputs"  # 入力音声が保存されているディレクトリ
output_dir = r"C:\Users\user\Desktop\git\WavTokenizer\outputs"  # 出力音声を保存するディレクトリ

# 出力ディレクトリが存在しない場合、作成する
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

# 入力ディレクトリ内のすべての.wavファイルを処理
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_filepath = os.path.join(input_dir, filename)
        wav, sr = torchaudio.load(input_filepath)

        # 音声の変換
        wav = convert_audio(wav, sr, 24000, 1)
        bandwidth_id = torch.tensor([0])
        wav = wav.to(device)

        # エンコードとデコード
        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

        # 出力ファイル名を入力ファイル名に「_processed」を付けて決定
        output_filename = os.path.splitext(filename)[0] + "_processed.wav"  # 拡張子の前に「_processed」を追加
        output_filepath = os.path.join(output_dir, output_filename)

        # 音声を保存
        torchaudio.save(output_filepath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

        print(f"Processed and saved: {output_filename}")
