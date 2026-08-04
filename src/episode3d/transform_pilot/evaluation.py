"""Zero-shot InternVL/SenseNova evaluation for Transform Pilot records."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

DIRECTION_TEXT = {
    "前方": "front",
    "右侧": "right",
    "后方": "back",
    "左侧": "left",
}


def parse_multiple_choice(response: str, choices: list[str] | None = None) -> str | None:
    """Extract a single A-D choice without treating arbitrary prose as an answer."""

    answer_tags = re.findall(r"<answer>\s*([A-D])(?:\.|\b)", response, re.IGNORECASE)
    if answer_tags:
        return answer_tags[-1].upper()
    explicit = re.findall(
        r"(?:答案|answer|选项)\s*(?:是|为|:)?\s*([A-D])(?:\.|\b)",
        response,
        re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    bare = re.fullmatch(r"\s*([A-D])(?:\..*)?\s*", final_line, re.IGNORECASE)
    if bare:
        return bare.group(1).upper()
    if choices:
        normalized = response.strip().lower()
        matches = [
            chr(ord("A") + index)
            for index, choice in enumerate(choices)
            if choice.lower() in normalized
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def trace_format(response: str) -> dict[str, bool]:
    return {
        "has_cue": bool(re.search(r"<cue>.*?</cue>", response, re.DOTALL)),
        "has_transform": bool(
            re.search(r"<transform>.*?</transform>", response, re.DOTALL)
        ),
        "has_answer": bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL)),
    }


def parse_direction_label(response: str) -> str | None:
    """Extract a direction word from the answer span, never from the trace."""

    answer_spans = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    scope = answer_spans[-1] if answer_spans else (
        response.strip().splitlines()[-1] if response.strip() else ""
    )
    matches = re.findall(r"前方|右侧|后方|左侧", scope)
    return DIRECTION_TEXT[matches[-1]] if matches else None


def parse_choice_labels(question: str) -> dict[str, str]:
    return {
        key: DIRECTION_TEXT[text]
        for key, text in re.findall(r"([A-D])\.\s*(前方|右侧|后方|左侧)", question)
    }


def _score_response(
    *, response: str, item: dict[str, Any], oracle: dict[str, Any]
) -> dict[str, Any]:
    prediction = parse_multiple_choice(response)
    choice_labels = parse_choice_labels(str(item["question"]))
    option_label = choice_labels.get(prediction) if prediction else None
    text_label = parse_direction_label(response)
    semantic_label = text_label or option_label
    option_text_consistent = (
        option_label == text_label
        if option_label is not None and text_label is not None
        else None
    )
    return {
        "prediction": prediction,
        "prediction_option_label": option_label,
        "prediction_text_label": text_label,
        "prediction_semantic_label": semantic_label,
        "ground_truth": oracle["answer"]["key"],
        "ground_truth_label": oracle["answer"]["label"],
        "correct": prediction == oracle["answer"]["key"],
        "semantic_correct": semantic_label == oracle["answer"]["label"],
        "option_text_consistent": option_text_consistent,
        "trace_format": trace_format(response),
    }


def _dynamic_preprocess(image: Any, *, image_size: int, max_num: int) -> list[Any]:
    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    ratio = min(
        ratios,
        key=lambda item: (
            abs(aspect - item[0] / item[1]),
            -int(width * height > 0.5 * image_size**2 * item[0] * item[1]),
        ),
    )
    target_width = image_size * ratio[0]
    target_height = image_size * ratio[1]
    resized = image.resize((target_width, target_height))
    patches = []
    for index in range(ratio[0] * ratio[1]):
        x = (index % ratio[0]) * image_size
        y = (index // ratio[0]) * image_size
        patches.append(resized.crop((x, y, x + image_size, y + image_size)))
    if len(patches) != 1:
        patches.append(image.resize((image_size, image_size)))
    return patches


def _local_internvl_classes(model_path: Path) -> tuple[type[Any], type[Any]]:
    """Import a local InternVL snapshot without consulting remote ``auto_map``.

    SenseNova 1.1 stores repository-qualified entries containing a dot in its
    ``auto_map``.  Older Transformers versions split that repository name into
    an invalid Python package (``SenseNova-SI-1``).  Loading the checked local
    source as a synthetic package both avoids that bug and pins evaluation to
    the code shipped beside the weights.
    """

    model_path = model_path.resolve()
    required = (
        "configuration_intern_vit.py",
        "configuration_internvl_chat.py",
        "modeling_intern_vit.py",
        "modeling_internvl_chat.py",
        "conversation.py",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"local InternVL snapshot misses code: {missing}")
    digest = hashlib.sha256(str(model_path).encode()).hexdigest()[:12]
    package_name = f"_epispace_internvl_{digest}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(model_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    config_module = importlib.import_module(
        f"{package_name}.configuration_internvl_chat"
    )
    model_module = importlib.import_module(f"{package_name}.modeling_internvl_chat")
    return config_module.InternVLChatConfig, model_module.InternVLChatModel


class InternVLBackend:
    """Minimal local adapter matching SenseNova-SI's official InternVL example."""

    def __init__(self, model_path: Path) -> None:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoTokenizer

        self.torch = torch
        self.Image = Image
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda image: image.convert("RGB")
                    if image.mode != "RGB"
                    else image
                ),
                transforms.Resize(
                    (448, 448), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        config_class, model_class = _local_internvl_classes(model_path)
        config = config_class.from_pretrained(model_path)
        num_layers = int(config.llm_config.num_hidden_layers)
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("SenseNova evaluation requires at least one visible CUDA GPU")
        per_gpu = math.ceil(num_layers / (world_size - 0.5))
        allocations = [per_gpu] * world_size
        allocations[0] = math.ceil(allocations[0] * 0.5)
        device_map: dict[str, int] = {}
        layer = 0
        for gpu_index, count in enumerate(allocations):
            for _ in range(count):
                if layer < num_layers:
                    device_map[f"language_model.model.layers.{layer}"] = gpu_index
                    layer += 1
        for name in (
            "vision_model",
            "mlp1",
            "language_model.model.tok_embeddings",
            "language_model.model.embed_tokens",
            "language_model.output",
            "language_model.model.norm",
            "language_model.model.rotary_emb",
            "language_model.lm_head",
        ):
            device_map[name] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        self.model = model_class.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            device_map=device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

    def _pixels(self, images: list[Path]) -> tuple[Any, list[int]]:
        max_num = max(1, min(6, 64 // len(images)))
        tensors = []
        counts = []
        for path in images:
            image = self.Image.open(path).convert("RGB")
            patches = _dynamic_preprocess(image, image_size=448, max_num=max_num)
            tensor = self.torch.stack([self.transform(patch) for patch in patches])
            tensors.append(tensor)
            counts.append(tensor.shape[0])
        return self.torch.cat(tensors).to(dtype=self.torch.bfloat16, device="cuda:0"), counts

    def generate(
        self, *, images: list[Path], prompt: str, maximum_new_tokens: int
    ) -> str:
        pixel_values, patch_counts = self._pixels(images)
        image_prefix = "".join(
            f"Image-{index + 1}: <image>\n" for index in range(len(images))
        )
        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=patch_counts,
            question=image_prefix + prompt,
            generation_config={
                "max_new_tokens": maximum_new_tokens,
                "do_sample": False,
                "pad_token_id": (
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                ),
            },
            history=None,
        )
        return str(response)


def evaluate_sensenova(
    *,
    dataset_root: Path,
    model_path: Path,
    output_path: Path,
    prompt_mode: str,
    evaluation_scope: str = "test",
    start_index: int = 0,
    stop_index: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run resumable zero-shot evaluation and write prediction-level evidence."""

    if prompt_mode not in {"answer_only", "grounded_cot"}:
        raise ValueError("prompt_mode must be answer_only or grounded_cot")
    allowed_scopes = {"test", "full_test", "balanced_all", "all"}
    if evaluation_scope not in allowed_scopes:
        raise ValueError(
            f"evaluation_scope must be one of {sorted(allowed_scopes)}"
        )
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if stop_index is not None and stop_index < start_index:
        raise ValueError("stop_index must not precede start_index")
    dataset_root = dataset_root.resolve()
    inputs_path = dataset_root / f"eval_inputs_{evaluation_scope}.jsonl"
    oracle_path = dataset_root / f"eval_oracle_{evaluation_scope}.jsonl"
    inputs = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in inputs_path
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    oracle = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in oracle_path
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    if set(inputs) != set(oracle):
        raise ValueError("evaluation inputs and oracle IDs disagree")
    done: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["record_id"])] = row
    wanted = sorted(inputs)[start_index:stop_index]
    if limit is not None:
        wanted = wanted[:limit]
    pending = [record_id for record_id in wanted if record_id not in done]
    backend = InternVLBackend(model_path) if pending else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        for record_id in pending:
            item = inputs[record_id]
            instruction = (
                "只输出一个选项，格式为 <answer>A. 选项文本</answer>。"
                if prompt_mode == "answer_only"
                else (
                    "先给出图像线索，再明确执行视角变换，最后作答。严格使用"
                    "<cue>...</cue><transform>...</transform><answer>A. 选项文本</answer>。"
                )
            )
            prompt = f"{item['system']}\n{instruction}\n{item['question']}"
            images = [dataset_root / path for path in item["images"]]
            assert backend is not None
            response = backend.generate(
                images=images,
                prompt=prompt,
                maximum_new_tokens=96 if prompt_mode == "answer_only" else 384,
            )
            scores = _score_response(
                response=response,
                item=item,
                oracle=oracle[record_id],
            )
            row = {
                "record_id": record_id,
                "task_id": item["task_id"],
                "scene_id": item["scene_id"],
                "split": item.get("split", oracle[record_id].get("split", "test")),
                "evaluation_scope": evaluation_scope,
                "prompt_mode": prompt_mode,
                **scores,
                "response": response,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            done[record_id] = row
    for record_id, row in done.items():
        if record_id not in inputs:
            continue
        row.setdefault("task_id", inputs[record_id]["task_id"])
        row.setdefault("scene_id", inputs[record_id]["scene_id"])
        row.setdefault(
            "split", inputs[record_id].get("split", oracle[record_id].get("split", "test"))
        )
        row["evaluation_scope"] = evaluation_scope
        row.update(
            _score_response(
                response=str(row["response"]),
                item=inputs[record_id],
                oracle=oracle[record_id],
            )
        )
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as stream:
        for record_id in sorted(set(done) & set(inputs)):
            stream.write(json.dumps(done[record_id], ensure_ascii=False) + "\n")
    temporary_output.replace(output_path)
    selected = [done[record_id] for record_id in wanted if record_id in done]
    counts = Counter(row["task_id"] for row in selected)
    correct = Counter(row["task_id"] for row in selected if row["correct"])
    semantic_correct = Counter(
        row["task_id"] for row in selected if row["semantic_correct"]
    )
    trace = Counter(
        row["task_id"]
        for row in selected
        if all(row["trace_format"].values())
    )
    consistent = Counter(
        row["task_id"]
        for row in selected
        if row["option_text_consistent"] is True
    )
    consistency_denominator = Counter(
        row["task_id"]
        for row in selected
        if row["option_text_consistent"] is not None
    )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "count": count,
            "accuracy": sum(row["correct"] for row in rows) / count,
            "semantic_accuracy": sum(row["semantic_correct"] for row in rows) / count,
            "trace_format_rate": (
                sum(all(row["trace_format"].values()) for row in rows) / count
            ),
            "joint_correct_trace_rate": (
                sum(
                    row["correct"] and all(row["trace_format"].values())
                    for row in rows
                )
                / count
            ),
        }

    split_names = sorted({str(row.get("split", "unknown")) for row in selected})
    summary = {
        "schema_version": "epispace.sensenova_zeroshot_eval.v1",
        "model_path": str(model_path.resolve()),
        "prompt_mode": prompt_mode,
        "evaluation_scope": evaluation_scope,
        "record_index_range": {"start": start_index, "stop": stop_index},
        "evaluation_inputs": str(inputs_path),
        "evaluation_oracle": str(oracle_path),
        "evaluated": len(selected),
        "accuracy": (
            sum(row["correct"] for row in selected) / len(selected) if selected else 0.0
        ),
        "semantic_accuracy": (
            sum(row["semantic_correct"] for row in selected) / len(selected)
            if selected
            else 0.0
        ),
        "option_text_consistency_rate": (
            sum(row["option_text_consistent"] is True for row in selected)
            / sum(row["option_text_consistent"] is not None for row in selected)
            if any(row["option_text_consistent"] is not None for row in selected)
            else None
        ),
        "option_text_consistency_count": sum(
            row["option_text_consistent"] is True for row in selected
        ),
        "option_text_comparable_count": sum(
            row["option_text_consistent"] is not None for row in selected
        ),
        "trace_format_rate": (
            sum(all(row["trace_format"].values()) for row in selected) / len(selected)
            if selected
            else 0.0
        ),
        "joint_correct_trace_rate": (
            sum(
                row["correct"] and all(row["trace_format"].values())
                for row in selected
            )
            / len(selected)
            if selected
            else 0.0
        ),
        "by_task": {
            task: {
                "count": count,
                "accuracy": correct[task] / count,
                "semantic_accuracy": semantic_correct[task] / count,
                "option_text_consistency_rate": (
                    consistent[task] / consistency_denominator[task]
                    if consistency_denominator[task]
                    else None
                ),
                "trace_format_rate": trace[task] / count,
                "joint_correct_trace_rate": sum(
                    row["correct"] and all(row["trace_format"].values())
                    for row in selected
                    if row["task_id"] == task
                )
                / count,
            }
            for task, count in sorted(counts.items())
        },
        "by_split": {
            split: aggregate(
                [row for row in selected if str(row.get("split", "unknown")) == split]
            )
            for split in split_names
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
