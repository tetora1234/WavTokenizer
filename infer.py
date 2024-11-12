import os
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device = torch.device('cpu')

config_path = r"C:\Users\user\Desktop\git\WavTokenizer\configs\48hz.yaml"
model_path = r"C:\Users\user\Desktop\git\WavTokenizer\result\train\180638\models\last.ckpt"
input_dir = r"C:\Users\user\Desktop\git\WavTokenizer\inputs"  # 入力音声が保存されているディレクトリ
output_dir = r"C:\Users\user\Desktop\git\WavTokenizer\outputs"  # 出力音声を保存するディレクトリ

# 出力ディレクトリが存在しない場合、作成する
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

# 入力ディレクトリ内のすべての.wavファイルを処理
for filename in os.listdir(input_dir):
    input_filepath = os.path.join(input_dir, filename)
    wav, sr = torchaudio.load(input_filepath)

    # 音声の変換
    wav = convert_audio(wav, sr, 48000, 1)
    bandwidth_id = torch.tensor([0])
    wav = wav.to(device)

    # エンコードとデコード
    features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

    # 出力ファイル名を入力ファイル名に「_processed」を付けて決定
    output_filename = os.path.splitext(filename)[0] + "_processed.wav"  # 拡張子の前に「_processed」を追加
    output_filepath = os.path.join(output_dir, output_filename)

    # 音声を保存
    torchaudio.save(output_filepath, audio_out, sample_rate=48000, encoding='PCM_S', bits_per_sample=32)

    print(f"output_filename: {output_filename}\ndiscrete_code.size:{discrete_code.size(-1)}\ndiscrete_code:{discrete_code}\n\n")