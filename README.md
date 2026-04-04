# 日本語BERT感情分析ツール

日本語テキストの感情（ポジティブ/ネガティブ）を判定するCLIツールです。

## プロジェクト概要

**BERT（Bidirectional Encoder Representations from Transformers）** は、Googleが開発した自然言語処理の事前学習モデルです。テキストを双方向に読み取ることで、文脈を深く理解できます。

このプロジェクトでは、東北大学が公開した日本語BERTをベースに感情分析用にfine-tuningされたモデルを使用して、入力された日本語テキストがポジティブかネガティブかを判定します。

### できること

- 日本語テキストのポジティブ/ネガティブ判定
- 確信度スコアの表示
- 単一テキスト分析、バッチ分析、インタラクティブモードに対応

## 使用モデル

[`koheiduck/bert-japanese-finetuned-sentiment`](https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)

- ベースモデル: 東北大学 `cl-tohoku/bert-base-japanese-whole-word-masking`
- タスク: 感情分析（2クラス: POSITIVE / NEGATIVE）
- トークナイザー: MeCab + unidic-lite

## 環境構築

Python 3.10以上が必要です。

```bash
pip install -r requirements.txt
```

> 初回実行時にHugging Faceからモデルが自動ダウンロードされます（約440MB）。

## 使い方

### 単一テキストの分析

```bash
python -m src.cli analyze "この映画は最高に面白かった！"
```

### インタラクティブモード

```bash
python -m src.cli interactive
```

テキストを入力するたびに感情分析が実行されます。`q` で終了します。

### デモ実行

```bash
python -m src.cli demo
```

7件のサンプルテキストで動作確認できます。

### デモスクリプト

```bash
python examples/demo.py
```

## プロジェクト構成

```
bert-japanese-sentiment/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── __main__.py       # python -m src で実行可能にする
│   ├── model.py           # モデルのロード・推論ロジック
│   └── cli.py             # CLIインターフェース
├── examples/
│   └── demo.py            # サンプルテキストで動作確認
└── tests/
    └── test_model.py      # 基本的なテスト
```

## 技術的な補足

### BERTのアーキテクチャ

BERTはTransformerのEncoder部分を使用したモデルで、以下の特徴があります：

- **双方向性**: 従来の左から右（または右から左）の一方向モデルと異なり、文の両方向から文脈を学習
- **事前学習**: 大量のテキストデータでMasked Language Model（MLM）とNext Sentence Prediction（NSP）の2つのタスクで事前学習
- **Self-Attention**: 入力トークン間の関係性を動的に学習するAttentionメカニズムにより、長距離の依存関係を捉える

### Fine-tuning

事前学習済みBERTモデルに感情分析用の分類層を追加し、感情ラベル付きデータセットで追加学習（fine-tuning）を行うことで、汎用的な言語理解能力を感情分析タスクに特化させています。この転移学習アプローチにより、少量のラベル付きデータでも高い精度を実現できます。

## テスト

```bash
pytest tests/
```

## License

MIT
