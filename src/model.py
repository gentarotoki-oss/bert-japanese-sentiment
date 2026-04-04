"""日本語BERT感情分析モデルのロード・推論ロジック"""

from transformers import pipeline


MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"


class SentimentAnalyzer:
    """日本語テキストの感情分析を行うクラス

    東北大学の日本語BERTをベースにfine-tuningされたモデルを使用して、
    テキストのポジティブ/ネガティブ判定を行う。
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """モデルとトークナイザーをロードする

        Args:
            model_name: Hugging Face上のモデル名
        """
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True,
        )

    def analyze(self, text: str) -> dict:
        """単一テキストの感情分析を行う

        Args:
            text: 分析対象の日本語テキスト

        Returns:
            分析結果の辞書。label, confidence, scoresを含む。
        """
        results = self.pipe(text)
        scores = {item["label"]: item["score"] for item in results[0]}
        top = max(results[0], key=lambda x: x["score"])
        return {
            "text": text,
            "label": top["label"],
            "confidence": top["score"],
            "scores": scores,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """複数テキストをバッチで感情分析する

        Args:
            texts: 分析対象の日本語テキストのリスト

        Returns:
            各テキストの分析結果の辞書のリスト
        """
        results = self.pipe(texts)
        output = []
        for text, result in zip(texts, results):
            scores = {item["label"]: item["score"] for item in result}
            top = max(result, key=lambda x: x["score"])
            output.append({
                "text": text,
                "label": top["label"],
                "confidence": top["score"],
                "scores": scores,
            })
        return output
