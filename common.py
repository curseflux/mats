#!/usr/bin/env python3
"""Shared utilities for the Gemma 4 conflict-awareness experiments.

The central invariant is that generation, candidate scoring, and activation
collection all condition on the *same* canonical chat-templated prefix.  The
prefix includes Gemma 4's model-turn generation prompt and, with thinking
disabled, its empty thought channel.  We never hand-write control tokens.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


CODE_VERSION = "1.0.0"
SUPPORTED_ACTIVATION_POSITIONS = {
    "claim_answer_end",
    "claim_end",
    "context_end",
    "policy_end",
    "query_subject_end",
    "question_end",
    "prompt_end",
    "assistant_start",
}


# ---------------------------------------------------------------------------
# Configuration and files
# ---------------------------------------------------------------------------


def _require(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing required config key: {dotted_key}")
        value = value[part]
    return value


def _resolve_optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, minimally validate, and resolve paths relative to the YAML file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    for key in (
        "project.seed",
        "paths.dataset_dir",
        "paths.output_dir",
        "model.id",
        "model.revision",
        "model.transformers_version",
        "model.dtype",
        "chat.enable_thinking",
        "chat.add_generation_prompt",
        "chat.max_input_tokens",
        "generation.do_sample",
        "generation.max_new_tokens",
        "screening.batch_size",
        "screening.scoring_batch_size",
        "screening.results_file",
        "screening.eligible_facts_file",
        "screening.run_metadata_file",
        "screening.require_all_template_bundles",
        "screening.min_true_minus_false_logprob",
        "collection.generation_batch_size",
        "collection.scoring_batch_size",
        "collection.activation_batch_size",
        "collection.behavior_results_file",
        "collection.run_metadata_file",
        "collection.require_complete_factorial_groups",
        "collection.activations.enabled",
        "collection.activations.directory",
        "collection.activations.manifest_file",
        "collection.activations.shard_size",
        "collection.activations.storage_dtype",
        "collection.activations.layers",
        "collection.activations.positions",
    ):
        _require(config, key)

    if config["chat"]["enable_thinking"] is not False:
        raise ValueError("This experiment requires chat.enable_thinking=false")
    if config["chat"]["add_generation_prompt"] is not True:
        raise ValueError("Gemma IT inference requires chat.add_generation_prompt=true")
    if config["generation"]["do_sample"] is not False:
        raise ValueError("Screening and behavior collection require deterministic greedy decoding")
    if config["screening"]["require_all_template_bundles"] is not True:
        raise ValueError("This design requires every screening template bundle to pass")
    if float(config["screening"]["min_true_minus_false_logprob"]) < 0:
        raise ValueError("screening.min_true_minus_false_logprob cannot be negative")
    if int(config["chat"]["max_input_tokens"]) < 1:
        raise ValueError("chat.max_input_tokens must be positive")
    for dotted_key in (
        "generation.max_new_tokens",
        "screening.batch_size",
        "screening.scoring_batch_size",
        "collection.generation_batch_size",
        "collection.scoring_batch_size",
        "collection.activation_batch_size",
        "collection.activations.shard_size",
    ):
        if int(_require(config, dotted_key)) < 1:
            raise ValueError(f"{dotted_key} must be positive")
    if config["collection"]["require_complete_factorial_groups"] is not True:
        raise ValueError("This design requires complete factorial groups")
    if not isinstance(config["collection"]["activations"]["enabled"], bool):
        raise ValueError("collection.activations.enabled must be true or false")
    if config["collection"]["activations"]["storage_dtype"] not in {
        "float16",
        "float32",
    }:
        raise ValueError("Activation storage_dtype must be float16 or float32")

    base_dir = config_path.parent
    config["_config_path"] = config_path
    config["paths"]["dataset_dir"] = _resolve_optional_path(
        config["paths"]["dataset_dir"], base_dir
    )
    config["paths"]["output_dir"] = _resolve_optional_path(
        config["paths"]["output_dir"], base_dir
    )
    config["paths"]["hf_cache_dir"] = _resolve_optional_path(
        config["paths"].get("hf_cache_dir"), base_dir
    )

    activation_cfg = _require(config, "collection.activations")
    if not isinstance(activation_cfg, Mapping):
        raise ValueError("collection.activations must be a mapping")
    positions = list(_require(config, "collection.activations.positions"))
    unknown_positions = sorted(set(positions).difference(SUPPORTED_ACTIVATION_POSITIONS))
    if unknown_positions:
        raise ValueError(f"Unknown activation positions: {unknown_positions}")
    if len(positions) != len(set(positions)):
        raise ValueError("Activation positions must be unique")
    if not positions:
        raise ValueError("At least one activation position is required")

    return config


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "_config_path"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text_sha256(payload)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _jsonable(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _jsonable(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize_jsonl(partial_path: str | Path, final_path: str | Path) -> None:
    partial_path, final_path = Path(partial_path), Path(final_path)
    if not partial_path.exists():
        raise FileNotFoundError(f"Missing partial JSONL file: {partial_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, final_path)


def unique_by(records: Sequence[Mapping[str, Any]], field: str, source: str) -> None:
    values = [str(record.get(field, "")) for record in records]
    if any(not value for value in values):
        raise ValueError(f"Missing {field} in {source}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {field} values in {source}")


def validate_manifest_file(
    manifest: Mapping[str, Any],
    filename: str,
    path: str | Path,
    record_count: int,
) -> None:
    entry = manifest.get("files", {}).get(filename)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Dataset manifest has no entry for {filename}")
    if entry.get("records") is not None and int(entry["records"]) != record_count:
        raise ValueError(f"Dataset manifest record count disagrees for {filename}")
    if str(entry.get("sha256", "")) != file_sha256(path):
        raise ValueError(f"Dataset manifest hash disagrees for {filename}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------


_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_OUTER_LEFT = " \t\r\n\"'`([{<“‘«"
_OUTER_RIGHT = " \t\r\n\"'`)]}>.,!?;:”“’»"


def normalize_answer(text: Any) -> str:
    """Match the dataset's casefold/outer-punctuation normalization rule."""
    value = unicodedata.normalize("NFKC", str(text))
    value = _ZERO_WIDTH.sub("", value).strip()
    value = value.lstrip(_OUTER_LEFT).rstrip(_OUTER_RIGHT)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def answer_matches(generated: str, acceptable_answers: Sequence[str]) -> bool:
    normalized = normalize_answer(generated)
    return bool(normalized) and normalized in {
        normalize_answer(answer) for answer in acceptable_answers
    }


def is_one_orthographic_word(text: str) -> bool:
    normalized = normalize_answer(text)
    return bool(normalized) and not bool(re.search(r"\s", normalized))


# ---------------------------------------------------------------------------
# Chat formatting and tokenization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPrompt:
    sample_id: str
    raw_prompt: str
    rendered_text: str
    raw_start: int
    semantic_positions: Mapping[str, int | None]


@dataclass(frozen=True)
class ContinuationScore:
    key: str
    continuation: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    sum_logprob: float
    mean_logprob: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.continuation,
            "normalized_text": normalize_answer(self.continuation),
            "token_ids": list(self.token_ids),
            "tokens": list(self.tokens),
            "token_logprobs": list(self.token_logprobs),
            "num_tokens": len(self.token_ids),
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
        }


def get_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("The loaded AutoProcessor does not expose a text tokenizer")
    return tokenizer


def render_messages(processor: Any, messages: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    rendered = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=bool(config["chat"]["add_generation_prompt"]),
        enable_thinking=bool(config["chat"]["enable_thinking"]),
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("apply_chat_template did not return a non-empty string")
    return rendered


def validate_text_chat_tokenization(
    processor: Any,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> None:
    """Ensure render-then-tokenize exactly matches the documented processor path."""
    messages = [{"role": "user", "content": "Gemma tokenization preflight."}]
    rendered = render_messages(processor, messages, config)
    direct = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=bool(config["chat"]["add_generation_prompt"]),
        enable_thinking=bool(config["chat"]["enable_thinking"]),
    )
    direct_ids = direct["input_ids"]
    if hasattr(direct_ids, "tolist"):
        direct_ids = direct_ids.tolist()
    if direct_ids and isinstance(direct_ids[0], list):
        direct_ids = direct_ids[0]
    rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if rendered_ids and isinstance(rendered_ids[0], list):
        rendered_ids = rendered_ids[0]
    if [int(value) for value in direct_ids] != [int(value) for value in rendered_ids]:
        raise RuntimeError(
            "Render-then-tokenize differs from processor.apply_chat_template(tokenize=True)"
        )


def render_dataset_record(
    processor: Any,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RenderedPrompt:
    sample_id = str(record["sample_id"])
    raw_prompt = str(record["raw_prompt"])
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Sample {sample_id} has no chat messages")
    if (
        len(messages) != 1
        or messages[0].get("role") != "user"
        or messages[0].get("content") != raw_prompt
    ):
        raise ValueError(
            f"Sample {sample_id} must contain exactly one unmodified user message"
        )

    rendered = render_messages(processor, messages, config)
    raw_start = rendered.find(raw_prompt)
    if raw_start < 0 or raw_start != rendered.rfind(raw_prompt):
        raise ValueError(
            f"Sample {sample_id}: raw_prompt must occur exactly once in the templated prefix"
        )
    semantic_positions = record.get("semantic_positions", {})
    if not isinstance(semantic_positions, Mapping):
        raise ValueError(f"Sample {sample_id} has invalid semantic_positions")
    return RenderedPrompt(
        sample_id=sample_id,
        raw_prompt=raw_prompt,
        rendered_text=rendered,
        raw_start=raw_start,
        semantic_positions=semantic_positions,
    )


@contextlib.contextmanager
def tokenizer_padding_side(tokenizer: Any, side: str):
    previous = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = previous


def encode_rendered_batch(
    tokenizer: Any,
    rendered_texts: Sequence[str],
    max_input_tokens: int,
    *,
    return_offsets_mapping: bool = False,
) -> Any:
    """Tokenize canonical rendered strings without adding control tokens again."""
    with tokenizer_padding_side(tokenizer, "left"):
        encoded = tokenizer(
            list(rendered_texts),
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=return_offsets_mapping,
            return_tensors="pt",
        )
    lengths = encoded["attention_mask"].sum(dim=1)
    maximum = int(lengths.max().item())
    if maximum > int(max_input_tokens):
        raise ValueError(
            f"A templated prompt has {maximum} tokens, exceeding max_input_tokens={max_input_tokens}. "
            "Prompts are never silently truncated."
        )
    return encoded


def build_position_ids(attention_mask: Any) -> Any:
    """Assign real tokens positions 0..length-1 despite left padding."""
    import torch

    position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    return position_ids.clamp_min(0)


def _model_input_device(model: Any) -> Any:
    try:
        device = model.get_input_embeddings().weight.device
        if str(device) != "meta":
            return device
    except (AttributeError, TypeError, NotImplementedError):
        pass
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("Could not identify the model's input device")


def _to_model_device(values: Mapping[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = _model_input_device(model)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def char_end_to_token_index(
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int],
    char_end: int,
) -> int:
    """Map an exclusive character endpoint to the token containing its last char."""
    real_tokens = [
        (index, int(start), int(end))
        for index, ((start, end), mask) in enumerate(zip(offsets, attention_mask))
        if int(mask) == 1 and int(end) > int(start)
    ]
    containing = [index for index, start, end in real_tokens if start < char_end <= end]
    if containing:
        return containing[-1]
    left = [(end, index) for index, _, end in real_tokens if end <= char_end]
    if left:
        return max(left)[1]
    raise ValueError(f"No non-padding token exists at or before character endpoint {char_end}")


def locate_activation_positions(
    tokenizer: Any,
    prompts: Sequence[RenderedPrompt],
    position_names: Sequence[str],
    max_input_tokens: int,
) -> tuple[Any, Any, Any, Any]:
    """Return encoding, token indices, token IDs, and unpadded prompt lengths."""
    import torch

    encoded = encode_rendered_batch(
        tokenizer,
        [prompt.rendered_text for prompt in prompts],
        max_input_tokens,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    attention_mask = encoded["attention_mask"]
    indices: list[list[int]] = []

    for row_index, prompt in enumerate(prompts):
        row_offsets = offsets[row_index].tolist()
        row_mask = attention_mask[row_index].tolist()
        non_padding = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
        if non_padding.numel() == 0:
            raise ValueError(f"Sample {prompt.sample_id} tokenized to no real tokens")
        row_indices: list[int] = []
        for name in position_names:
            if name == "assistant_start":
                token_index = int(non_padding[-1].item())
            else:
                endpoint = prompt.semantic_positions.get(name)
                if endpoint is None:
                    raise ValueError(
                        f"Sample {prompt.sample_id} has no endpoint for requested position {name!r}"
                    )
                endpoint = int(endpoint)
                if not 0 < endpoint <= len(prompt.raw_prompt):
                    raise ValueError(
                        f"Sample {prompt.sample_id} has invalid {name} endpoint {endpoint}"
                    )
                formatted_endpoint = prompt.raw_start + endpoint
                token_index = char_end_to_token_index(row_offsets, row_mask, formatted_endpoint)
            if int(attention_mask[row_index, token_index].item()) != 1:
                raise AssertionError(f"Sample {prompt.sample_id}: {name} resolved to padding")
            row_indices.append(token_index)
        indices.append(row_indices)

    position_indices = torch.tensor(indices, dtype=torch.long)
    position_token_ids = encoded["input_ids"].gather(1, position_indices)
    lengths = attention_mask.sum(dim=1).to(dtype=torch.int32)
    return encoded, position_indices, position_token_ids, lengths


# ---------------------------------------------------------------------------
# Model loading, generation, and likelihoods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    processor: Any
    tokenizer: Any


def load_model_bundle(config: Mapping[str, Any]) -> ModelBundle:
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    required_version = str(config["model"]["transformers_version"])
    if transformers.__version__ != required_version:
        raise RuntimeError(
            f"transformers=={required_version} is required for reproducible Gemma 4 runs; "
            f"found {transformers.__version__}"
        )

    dtype_name = str(config["model"]["dtype"]).casefold()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")

    common_kwargs: dict[str, Any] = {
        "revision": str(config["model"]["revision"]),
        "trust_remote_code": bool(config["model"].get("trust_remote_code", False)),
        "local_files_only": bool(config["model"].get("local_files_only", False)),
    }
    cache_dir = config["paths"].get("hf_cache_dir")
    if cache_dir is not None:
        common_kwargs["cache_dir"] = str(cache_dir)

    processor = AutoProcessor.from_pretrained(
        str(config["model"]["id"]),
        padding_side="left",
        **common_kwargs,
    )
    tokenizer = get_tokenizer(processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        raise ValueError("The Gemma tokenizer has no pad token; do not substitute EOS silently")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Gemma's pad and primary EOS tokens should remain distinct")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Activation positions require the checkpoint's fast tokenizer")
    validate_text_chat_tokenization(processor, tokenizer, config)

    model_kwargs = dict(common_kwargs)
    model_kwargs.update(
        {
            "dtype": dtype_map[dtype_name],
            "device_map": config["model"].get("device_map", "auto"),
            "attn_implementation": config["model"].get("attention_implementation", "sdpa"),
            "low_cpu_mem_usage": True,
        }
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        str(config["model"]["id"]),
        **model_kwargs,
    )
    model.eval()
    if type(model).__name__ != "Gemma4UnifiedForConditionalGeneration":
        raise TypeError(
            "Expected Gemma4UnifiedForConditionalGeneration, found "
            f"{type(model).__name__}"
        )
    return ModelBundle(model=model, processor=processor, tokenizer=tokenizer)


def _eos_token_ids(model: Any) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(value) for value in eos}


def _trim_generated_ids(
    token_ids: Sequence[int],
    pad_token_id: int,
    eos_token_ids: set[int],
) -> tuple[list[int], int | None, str]:
    content: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in eos_token_ids:
            return content, token_id, "eos"
        if token_id == pad_token_id:
            return content, token_id, "pad_after_eos"
        content.append(token_id)
    return content, None, "max_new_tokens"


def generate_batch(
    model: Any,
    tokenizer: Any,
    rendered_texts: Sequence[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    encoded = encode_rendered_batch(
        tokenizer,
        rendered_texts,
        int(config["chat"]["max_input_tokens"]),
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    model_inputs = _to_model_device(dict(encoded), model)
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "use_cache": bool(config["generation"].get("use_cache", True)),
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    with torch.inference_mode():
        sequences = model.generate(**model_inputs, **generation_kwargs)
    generated = sequences[:, prompt_width:].detach().cpu().tolist()
    eos_ids = _eos_token_ids(model)
    outputs: list[dict[str, Any]] = []
    for row in generated:
        content_ids, stop_id, finish_reason = _trim_generated_ids(
            row,
            int(tokenizer.pad_token_id),
            eos_ids,
        )
        decode_kwargs = {"clean_up_tokenization_spaces": False}
        raw_text = tokenizer.decode(content_ids, skip_special_tokens=False, **decode_kwargs)
        clean_text = tokenizer.decode(content_ids, skip_special_tokens=True, **decode_kwargs).strip()
        outputs.append(
            {
                "text": clean_text,
                "raw_text": raw_text,
                "normalized_text": normalize_answer(clean_text),
                "token_ids": content_ids,
                "num_tokens": len(content_ids),
                "finish_reason": finish_reason,
                "stop_token_id": stop_id,
            }
        )
    return outputs


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _prepare_continuation(tokenizer: Any, rendered_text: str, continuation: str) -> tuple[list[int], list[int]]:
    if not continuation:
        raise ValueError("Candidate continuations may not be empty")
    prompt_ids = _encode_without_special_tokens(tokenizer, rendered_text)
    combined_ids = _encode_without_special_tokens(tokenizer, rendered_text + continuation)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer continuation is not prefix-stable at the chat-template boundary. "
            "Refusing to score a silently altered prompt."
        )
    candidate_ids = combined_ids[len(prompt_ids) :]
    if not candidate_ids:
        raise ValueError(f"Candidate {continuation!r} tokenized to no continuation tokens")
    return prompt_ids, candidate_ids


def score_continuations(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, str]],
    batch_size: int,
    max_input_tokens: int,
) -> list[ContinuationScore]:
    """Teacher-force exact answer strings after the canonical generation prefix.

    ``sum_logprob`` is the primary joint sequence score. ``mean_logprob`` is
    retained as a tokenization-length diagnostic. EOS is intentionally not
    included: this measures answer-string preference, not stopping behavior.
    """
    import torch

    prepared: list[dict[str, Any]] = []
    for request in requests:
        key = str(request["key"])
        continuation = str(request["continuation"])
        prompt_ids, candidate_ids = _prepare_continuation(
            tokenizer,
            str(request["rendered_text"]),
            continuation,
        )
        if len(prompt_ids) + len(candidate_ids) > int(max_input_tokens):
            raise ValueError(f"Scored sequence {key} exceeds max_input_tokens={max_input_tokens}")
        prepared.append(
            {
                "key": key,
                "continuation": continuation,
                "prompt_ids": prompt_ids,
                "candidate_ids": candidate_ids,
                "combined_ids": prompt_ids + candidate_ids,
            }
        )

    results: list[ContinuationScore] = []
    for start in range(0, len(prepared), int(batch_size)):
        batch = prepared[start : start + int(batch_size)]
        width = max(len(item["combined_ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        candidate_starts: list[int] = []
        for row_index, item in enumerate(batch):
            combined = torch.tensor(item["combined_ids"], dtype=torch.long)
            pad_length = width - len(combined)
            input_ids[row_index, pad_length:] = combined
            attention_mask[row_index, pad_length:] = 1
            candidate_starts.append(pad_length + len(item["prompt_ids"]))

        keep = max(len(item["candidate_ids"]) + 1 for item in batch)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": build_position_ids(attention_mask),
        }
        inputs = _to_model_device(inputs, model)
        with torch.inference_mode():
            outputs = model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=keep,
            )
        logits = outputs.logits
        logits_start = width - int(logits.shape[1])

        for row_index, item in enumerate(batch):
            per_token: list[float] = []
            for offset, target_id in enumerate(item["candidate_ids"]):
                absolute_prediction_index = candidate_starts[row_index] - 1 + offset
                local_prediction_index = absolute_prediction_index - logits_start
                if not 0 <= local_prediction_index < logits.shape[1]:
                    raise AssertionError("logits_to_keep omitted a required candidate prediction")
                token_logits = logits[row_index, local_prediction_index].float()
                logprob = token_logits[int(target_id)] - torch.logsumexp(token_logits, dim=-1)
                per_token.append(float(logprob.item()))
            token_ids = tuple(int(value) for value in item["candidate_ids"])
            token_strings = tuple(tokenizer.convert_ids_to_tokens(list(token_ids)))
            total = float(sum(per_token))
            results.append(
                ContinuationScore(
                    key=str(item["key"]),
                    continuation=str(item["continuation"]),
                    token_ids=token_ids,
                    tokens=token_strings,
                    token_logprobs=tuple(per_token),
                    sum_logprob=total,
                    mean_logprob=total / len(per_token),
                )
            )
        del outputs, logits, inputs
    return results


# ---------------------------------------------------------------------------
# Decoder-layer activation capture
# ---------------------------------------------------------------------------


def get_decoder_layers(model: Any) -> Sequence[Any]:
    candidate_paths = (
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    )
    for path in candidate_paths:
        value = model
        try:
            for part in path:
                value = getattr(value, part)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and len(value) > 0:
            expected = getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", None)
            if expected is not None and len(value) != int(expected):
                continue
            return value
    raise TypeError(
        "Could not locate the decoder block list. The pinned Gemma/Transformers architecture may have changed."
    )


def resolve_layer_indices(specification: Any, num_layers: int) -> list[int]:
    if isinstance(specification, str):
        if specification.casefold() != "all":
            raise ValueError("Activation layers must be 'all' or an explicit integer list")
        return list(range(num_layers))
    if not isinstance(specification, Sequence) or isinstance(specification, (str, bytes)):
        raise ValueError("Activation layers must be 'all' or an explicit integer list")
    values = [int(value) for value in specification]
    if not values or len(values) != len(set(values)):
        raise ValueError("Explicit activation layer indices must be non-empty and unique")
    if values != sorted(values) or values[0] < 0 or values[-1] >= num_layers:
        raise ValueError(f"Activation layers must be sorted and within [0, {num_layers - 1}]")
    return values


def _first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    hidden = getattr(value, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise TypeError(f"Decoder block returned unsupported output type: {type(value).__name__}")


class LayerActivationCapture:
    """Capture post-block residual vectors at precomputed token positions."""

    def __init__(
        self,
        model: Any,
        layer_indices: Sequence[int],
        storage_dtype: str,
    ) -> None:
        import torch

        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        if storage_dtype not in dtype_map:
            raise ValueError("Activation storage_dtype must be float16 or float32")
        self.model = model
        self.layers = get_decoder_layers(model)
        self.layer_indices = list(layer_indices)
        self.storage_dtype = dtype_map[storage_dtype]
        self.position_indices: Any = None
        self.captured: dict[int, Any] = {}
        self.handles = [
            self.layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            for layer_index in self.layer_indices
        ]

    def _make_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            if self.position_indices is None:
                return
            hidden = _first_tensor(output)
            positions = self.position_indices.to(hidden.device)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            selected = hidden[batch_indices, positions, :]
            self.captured[layer_index] = selected.detach().to(
                device="cpu",
                dtype=self.storage_dtype,
            )

        return hook

    def capture(self, encoded: Mapping[str, Any], position_indices: Any) -> Any:
        import torch

        self.position_indices = position_indices
        self.captured = {}
        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "position_ids": build_position_ids(encoded["attention_mask"]),
        }
        model_inputs = _to_model_device(model_inputs, self.model)
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        missing = [index for index in self.layer_indices if index not in self.captured]
        if missing:
            raise RuntimeError(f"Activation hooks did not fire for layers: {missing}")
        activations = torch.stack(
            [self.captured[index] for index in self.layer_indices],
            dim=1,
        )
        self.position_indices = None
        self.captured = {}
        del outputs, model_inputs
        return activations

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> "LayerActivationCapture":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_fingerprint(
    config: Mapping[str, Any],
    bundle: ModelBundle,
    dataset_files: Mapping[str, str | Path],
) -> dict[str, Any]:
    import torch

    model, processor, tokenizer = bundle.model, bundle.processor, bundle.tokenizer
    chat_template = getattr(processor, "chat_template", None) or getattr(
        tokenizer, "chat_template", ""
    )
    text_config = getattr(model.config, "text_config", model.config)
    return {
        "code_version": CODE_VERSION,
        "config_sha256": config_sha256(config),
        "model": {
            "id": config["model"]["id"],
            "requested_revision": config["model"]["revision"],
            "loaded_commit_hash": getattr(model.config, "_commit_hash", None),
            "class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "dtype": str(getattr(model, "dtype", config["model"]["dtype"])),
            "num_hidden_layers": int(text_config.num_hidden_layers),
            "hidden_size": int(text_config.hidden_size),
            "attention_implementation": config["model"]["attention_implementation"],
            "device_map": _jsonable(getattr(model, "hf_device_map", None)),
        },
        "chat": {
            "add_generation_prompt": config["chat"]["add_generation_prompt"],
            "enable_thinking": config["chat"]["enable_thinking"],
            "chat_template_sha256": text_sha256(str(chat_template)),
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "generation_eos_token_ids": sorted(_eos_token_ids(model)),
        },
        "packages": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "tokenizers": package_version("tokenizers"),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "datasets": {
            name: {"path": str(Path(path)), "sha256": file_sha256(path)}
            for name, path in dataset_files.items()
        },
    }
