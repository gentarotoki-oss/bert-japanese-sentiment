"""サンプルテキストで感情分析の動作確認を行うスクリプト"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import SentimentAnalyzer

samples = [
    "この映画は最高に面白かった！また観たい",
    "今日の天気は最悪だ。電車も遅延してるし",
    "新しいプロジェクトが始まって、とてもワクワクしている",
    "期待していたレストランだったが、味も接客もひどかった",
    "まあまあかな、悪くはないけど特に良くもない",
    "数学の問題が解けた時の達成感は格別だ",
    "締め切りに間に合わなくて落ち込んでいる",
]


def main() -> None:
    print("モデルをロード中...")
    analyzer = SentimentAnalyzer()
    print("ロード完了!\n")

    for text in samples:
        result = analyzer.analyze(text)
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        print(f"{emoji} [{result['label']}] ({result['confidence']:.1%}) {text}")

    print(f"\n--- バッチ推論テスト ---")
    batch_results = analyzer.analyze_batch(samples)
    print(f"{len(batch_results)} 件の分析が完了しました。")


if __name__ == "__main__":
    main()
