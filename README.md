# Engine Simulator

音声から推定した回転数（RPM）をもとに、8気筒エンジンを想定した排気音を合成する実験用Pythonリポジトリです。`librosa` による基本周波数・RMS解析、RPM補間、左右バンクのインパルス生成、フィルタリング、排気反射エフェクトを組み合わせています。

## 必要環境

- Python 3.10 以上
- NumPy
- SciPy
- librosa
- SoundFile
- Matplotlib
- Pygame（リアルタイムRPMシミュレーションを使う場合）

依存パッケージは `pyproject.toml` からインストールできます。

```bash
python -m pip install -e .
```

開発用ツールもインストールする場合は、次を実行します。

```bash
python -m pip install -e ".[dev]"
```

## 使い方

### 音声からエンジン音を生成する

`engine.py` は、リポジトリ直下の `silen2.mp3` を入力として読み込みます。別の音声を使う場合は、`engine.py` の入力ファイル名を変更してください。

```bash
python engine.py
```

実行すると入力音声のRPM推定グラフを表示し、次の音声ファイルを生成します。

- `engine.wav`: 合成されたエンジン音
- `left.wav`: 左バンクの中間音声
- `right.wav`: 右バンクの中間音声

### キー操作でRPMをシミュレーションする

`simulator.py` はPygameウィンドウを開きます。スペースキーを押している間だけスロットルを開き、終了後にエンジン音とスペクトログラムを生成します。

```bash
python simulator.py
```

生成物:

- `simulator.wav`: シミュレーションされたエンジン音

生成されたWAVファイルはリポジトリ直下に出力されます。これらは `.gitignore` の対象です。

`rpm_simulator.py` はRPM変化だけを確認する簡易シミュレーターです。

```bash
python rpm_simulator.py
```

### STFTの実験

`stft.py` は `voice.mp3` を読み込み、STFT上で高調波成分を追加して `engine.wav` を生成します。

```bash
python stft.py
```

## モジュール構成

| ファイル | 役割 |
| --- | --- |
| `engine.py` | 音声解析からエンジン音生成までの実行スクリプト |
| `simulator.py` | Pygameによるスロットル操作とエンジン音生成 |
| `rpm_simulator.py` | キー操作によるRPM値の簡易シミュレーション |
| `engine_simulator.py` | 時間変化するスロットルからRPMカーブを生成 |
| `signal_processor.py` | 点火パルス、左右バンク、フィルタ、反射音の処理 |
| `audio_utils.py` | RPM計算、平滑化、フィルタなどの音声処理ユーティリティ |
| `effects.py` | 排気反射エフェクト |
| `stft.py` | STFTを使った高調波付加の実験 |
| `main.py` | 現在は未使用のエントリーポイント |

## 処理の概要

```text
入力音声
  ↓
基本周波数・RMS解析（librosa）
  ↓
RPM推定と音声長への補間
  ↓
左右バンクの点火パルス生成
  ↓
ローパス・バンドパス・排気反射
  ↓
正規化
  ↓
WAV出力（SoundFile）
```

## 注意事項

- 実行時にグラフウィンドウが表示されるスクリプトがあります。ウィンドウを閉じると処理が進みます。
- `engine.py` と `stft.py` は入力ファイル名をソースコード内に直接指定しています。
- 生成されたWAVファイルは既存ファイルを上書きします。
- `pyproject.toml` に実行用依存関係と開発用ツールを定義しています。
- 自動テストは今後追加予定です。
