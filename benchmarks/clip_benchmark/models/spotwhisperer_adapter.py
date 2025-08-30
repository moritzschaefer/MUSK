from pathlib import Path
import os
import torch
import torch.nn as nn
from torchvision import transforms

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning


class SpotWhispererMUSKAdapter(nn.Module):
    """Adapter that exposes a minimal MUSK-compatible forward interface around
    an existing CellWhisperer LightningModule or model object.

    The forward signature matches what the MUSK benchmark code expects:
    model(image=..., text_description=..., padding_mask=..., out_norm=..., with_head=...)
    returning (vision_cls, language_cls)

    Caveats:

    """

    def __init__(self, pl_model):
        super().__init__()
        # Accept either the LightningModule or the inner model
        self.pl_model = pl_model
        # If a Lightning wrapper was passed, try to extract the raw model
        self.model = getattr(pl_model, "model", pl_model)
        self.model.eval()

    def forward(self, image=None, text_description=None, padding_mask=None, return_global=True, with_head=True, out_norm=True, ms_aug=False, **kwargs):
        vision_cls = None
        language_cls = None

        # Image path: try common cellwhisperer model methods
        if image is not None:
            # Prefer get_image_features if available
            if hasattr(self.model, "get_image_features"):
                try:
                    # Some implementations return (features, embeds)
                    img_res = self.model.get_image_features(patches=image, normalize_embeds=out_norm)
                    if isinstance(img_res, tuple) or isinstance(img_res, list):
                        # prefer embeds if provided
                        vision_cls = img_res[-1]
                    else:
                        vision_cls = img_res
                except TypeError:
                    # fallback: call with different arg name
                    img_res = self.model.get_image_features(image=image, normalize_embeds=out_norm)
                    vision_cls = img_res[-1] if isinstance(img_res, (tuple, list)) else img_res
            elif hasattr(self.model, "encode_image"):
                # Some CLIP-like models expose encode_image
                try:
                    vision_cls = self.model.encode_image(image, proj_contrast=False, normalize=out_norm)
                except TypeError:
                    vision_cls = self.model.encode_image(image)
            else:
                raise RuntimeError("Underlying model does not expose an image encoding API compatible with the MUSK adapter")

        # Text path
        if text_description is not None:
            # If token ids + padding provided by MUSK tokenizer, pass through
            if hasattr(self.model, "get_text_features"):
                try:
                    txt_res = self.model.get_text_features(input_ids=text_description, attention_mask=padding_mask, normalize_embeds=out_norm)
                    language_cls = txt_res[-1] if isinstance(txt_res, (tuple, list)) else txt_res
                except Exception:
                    # best-effort fallback
                    txt_res = self.model.get_text_features(text_description)
                    language_cls = txt_res[-1] if isinstance(txt_res, (tuple, list)) else txt_res
            elif hasattr(self.model, "encode_text"):
                language_cls = self.model.encode_text(text_description)
            else:
                # no text API found
                language_cls = None

        # Ensure outputs are tensors or None
        return vision_cls, language_cls

    def encode_text(self, text_tokens):
        _, text_embeds = self.model.get_text_features(**text_tokens)
        return text_embeds


class TokenizedBatch:
    """Wrapper around tokenized output dict that implements .to(device)."""

    def __init__(self, data: dict):
        self.data = data

    def to(self, device):
        for k, v in list(self.data.items()):
            if hasattr(v, "to"):
                self.data[k] = v.to(device)
        return self

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()


class TokenizerWrapper:
    """Wrap a HF tokenizer so tokenizer(texts) returns TokenizedBatch with .to(device)."""

    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", **kwargs):
        inputs = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
            **kwargs,
        )
        return TokenizedBatch(inputs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def load_spotwhisperer_adapter(pretrained_path: str, device: str = "cuda"):
    """Load a CellWhisperer checkpoint and return a MUSK-compatible adapter,
    a basic image transform and a tokenizer placeholder (None).

    pretrained_path may be a path to a checkpoint file understood by
    load_cellwhisperer_model (preferred) or TranscriptomeTextDualEncoderLightning.load_from_checkpoint.
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)

    tokenizer = None

    # Prefer the repository helper which may also return a tokenizer

    pl_model, tokenizer, transcriptome_processor, image_processor = load_cellwhisperer_model(pretrained_path)

    pl_model.eval()

    adapter = SpotWhispererMUSKAdapter(pl_model)
    adapter.to(device)

    tokenizer = TokenizerWrapper(tokenizer)

    return adapter, image_processor.transform, tokenizer
