WavTokenizerを自分用に使いやすくした

メモ

トークン数 = sampling_rate / hop_length
24000 / 320 = 75
24000 / 600 = 40
n_fft = hop_length * 4しただけだと思う

とりあえず48000テストする

24000の訓練テスト結果
5,6epochまでlossが下がらないこともあるので10epochぐらいまで様子を見てから異常か判断する

所感
また、アーキテクチャが分かりやすい反面、訓練が長い。
また、モデルによってエンコーダーデコーダー結果が変わるためLLMへトークンを作成させる場合融通が利きにくい。