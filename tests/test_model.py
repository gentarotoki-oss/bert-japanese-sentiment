"""SentimentAnalyzerの基本テスト"""

import pytest

from src.model import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    """モジュールスコープでモデルを一度だけロードする"""
    return SentimentAnalyzer()


def test_model_loads(analyzer: SentimentAnalyzer) -> None:
    """モデルが正常にロードできること"""
    assert analyzer.pipe is not None


def test_positive_text(analyzer: SentimentAnalyzer) -> None:
    """ポジティブなテキストでPOSITIVEが返ること"""
    result = analyzer.analyze("この映画は最高に面白かった！また観たい")
    assert result["label"] == "POSITIVE"


def test_negative_text(analyzer: SentimentAnalyzer) -> None:
    """ネガティブなテキストでNEGATIVEが返ること"""
    result = analyzer.analyze("今日の天気は最悪だ。電車も遅延してるし")
    assert result["label"] == "NEGATIVE"


def test_analyze_schema(analyzer: SentimentAnalyzer) -> None:
    """analyzeの戻り値が正しいスキーマであること"""
    result = analyzer.analyze("テスト文です")
    assert "text" in result
    assert "label" in result
    assert "confidence" in result
    assert "scores" in result
    assert isinstance(result["text"], str)
    assert result["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result["confidence"] <= 1.0
    assert "POSITIVE" in result["scores"]
    assert "NEGATIVE" in result["scores"]


def test_analyze_batch(analyzer: SentimentAnalyzer) -> None:
    """analyze_batchが正しい件数を返すこと"""
    texts = ["楽しい", "悲しい", "普通"]
    results = analyzer.analyze_batch(texts)
    assert len(results) == len(texts)
    for result in results:
        assert "label" in result
        assert "confidence" in result
