from abc import ABC, abstractmethod

import numpy as np
import torch
import transformers

from .tokenizer import TransformerTokenizer


def batch_model_predict(model_predict, inputs, batch_size=64):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.
    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i: i + batch_size]
        batch_preds = model_predict(batch)

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)


class ModelWrapper(ABC):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based victim_model return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text victim_model return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    def __init__(self, model, tokenizer, batch_size=64):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, text_list):
        raise NotImplementedError()

    def encode(self, inputs):
        """
        Args:
            inputs (list[str]): list of input strings

        Returns:
            tokens (list[list[int]]): List of list of ids
        """
        return self.tokenizer.batch_encode(inputs)


class PyTorchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, batch_size=64):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )
        super(PyTorchModelWrapper, self).__init__(model, tokenizer, batch_size)

    def __call__(self, text_input_list):
        ids = self.encode(text_input_list)

        with torch.no_grad():
            outputs = batch_model_predict(
                self.model, ids, batch_size=self.batch_size
            )

        return outputs


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, batch_size=64):
        super(HuggingFaceModelWrapper, self).__init__(model, tokenizer, batch_size)
        if isinstance(self.tokenizer, transformers.PreTrainedTokenizer):
            self.tokenizer = TransformerTokenizer(tokenizer=self.tokenizer)

    def _model_predict(self, inputs):
        """Turn a list of dicts into a dict of lists.

        Then make lists (values of dict) into tensors.
        """
        model_device = next(self.model.parameters()).device
        input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
        input_dict = {
            k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
        }
        outputs = self.model(**input_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence victim_model return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification victim_model return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs[0]

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace victim_model as keyword arguments.
        (Regular PyTorch ``nn.Module`` victim_model typically take inputs as
        positional arguments.)
        """
        ids = self.encode(text_input_list)

        with torch.no_grad():
            outputs = batch_model_predict(
                self._model_predict, ids, batch_size=self.batch_size
            )

        return outputs
