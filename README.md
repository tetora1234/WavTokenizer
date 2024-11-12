WavTokenizerを自分用に使いやすくした

メモ

トークン数 = sampling_rate / hop_length
24000 / 320 = 75
24000 / 600 = 40
n_fft = hop_length * 4しただけだと思う
→1秒でテストしたところ4800で150が出たので確定

とりあえず48000テストする
-48000でも訓練できているみたい。
-lrを2e-4から上げるとlossの計算がおかしくなったので2e-4固定みたい。(内部の損失計算をいじるのは面倒)
-2e3はlossが落ち着いて計算できているが2e4はNan連発。(活性化関数周りだと思うが)

24000の訓練テスト結果
5,6epochまでlossが下がらないこともあるので10epochぐらいまで様子を見てから異常か判断する

所感
アーキテクチャが分かりやすい反面、訓練が長い。
また、モデルによってエンコーダーデコーダー結果が変わるためLLMへトークンを作成させる場合融通が利きにくい。