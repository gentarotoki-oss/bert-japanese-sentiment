"""日本語BERT感情分析のCLIインターフェース"""

import argparse
import sys

from src.model import SentimentAnalyzer


DEMO_SAMPLES = [
    "この映画は最高に面白かった！また観たい",
    "今日の天気は最悪だ。電車も遅延してるし",
    "新しいプロジェクトが始まって、とてもワクワクしている",
    "期待していたレストランだったが、味も接客もひどかった",
    "まあまあかな、悪くはないけど特に良くもない",
    "数学の問題が解けた時の達成感は格別だ",
    "締め切りに間に合わなくて落ち込んでいる",
]


def format_result(result: dict) -> str:
    """分析結果を見やすくフォーマットする"""
    label = result["label"]
    confidence = result["confidence"]
    pos_score = result["scores"].get("POSITIVE", 0)
    neg_score = result["scores"].get("NEGATIVE", 0)

    emoji = "😊" if label == "POSITIVE" else "😞"
    bar_len = 20
    pos_filled = round(pos_score * bar_len)
    neg_filled = round(neg_score * bar_len)

    lines = [
        f"  テキスト: {result['text']}",
        f"  判定:     {emoji} {label} ({confidence:.1%})",
        f"  POS: [{'█' * pos_filled}{'░' * (bar_len - pos_filled)}] {pos_score:.1%}",
        f"  NEG: [{'█' * neg_filled}{'░' * (bar_len - neg_filled)}] {neg_score:.1%}",
    ]
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> None:
    """単一テキストの分析を実行する"""
    print("モデルをロード中...")
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(args.text)
    print()
    print(format_result(result))


def cmd_interactive(args: argparse.Namespace) -> None:
    """インタラクティブモードで分析を実行する"""
    print("モデルをロード中...")
    analyzer = SentimentAnalyzer()
    print("日本語感情分析 インタラクティブモード")
    print("テキストを入力してください（'q' で終了）")
    print("-" * 50)

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if text.lower() == "q":
            print("終了します。")
            break
        if not text:
            continue

        result = analyzer.analyze(text)
        print()
        print(format_result(result))


def cmd_demo(args: argparse.Namespace) -> None:
    """デモ用サンプルテキストの分析を実行する"""
    print("モデルをロード中...")
    analyzer = SentimentAnalyzer()
    print(f"\n{'='*60}")
    print("  日本語BERT感情分析 デモ")
    print(f"{'='*60}\n")

    results = analyzer.analyze_batch(DEMO_SAMPLES)
    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}]")
        print(format_result(result))
        print()


def main() -> None:
    """CLIのエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="日本語BERT感情分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="単一テキストを分析")
    p_analyze.add_argument("text", type=str, help="分析対象のテキスト")
    p_analyze.set_defaults(func=cmd_analyze)

    # interactive
    p_interactive = subparsers.add_parser("interactive", help="インタラクティブモード")
    p_interactive.set_defaults(func=cmd_interactive)

    # demo
    p_demo = subparsers.add_parser("demo", help="サンプルテキストでデモ実行")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
