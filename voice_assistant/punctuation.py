#voice_assistant/punctuation.py
import logging
import os
import sherpa_onnx

log = logging.getLogger(__name__)


class PunctuationEngine:
    """
    Sherpa_onnx streaming punctuation engine
    """
    def __init__(self, punctuation_cfg: dict) -> None:
        punctuation_model = os.path.expanduser(punctuation_cfg.get("punctuation_model", ""))
        bpe = os.path.expanduser(punctuation_cfg.get("bpe", ""))

        model_config = sherpa_onnx.OnlinePunctuationModelConfig(
            cnn_bilstm=punctuation_model,
            bpe_vocab=bpe
        )
        config = sherpa_onnx.OnlinePunctuationConfig(model_config)
        self.punctuation = sherpa_onnx.OnlinePunctuation(config)

    def add_punctuation(self, phrase: str) -> str:
        return self.punctuation.add_punctuation_with_case(phrase.lower())
