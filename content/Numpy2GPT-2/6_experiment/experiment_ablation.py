"""Run three isolated GPT-2 ablations for the experiment blog post.

Run from this directory: .venv/bin/python experiment_ablation.py
Uses local weights/tokenizer only; the original model files are not edited.
"""
import hashlib
import inspect
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import GPT2Tokenizer

import notebook_model
from gpt2 import load_weights

ROOT = Path(__file__).resolve().parent
PROMPT = "Hello! My Name Is"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence: {old!r}")
    return source.replace(old, new, 1)


def make_variant(name):
    attention_source = inspect.getsource(notebook_model.multi_head_self_attention)
    forward_source = inspect.getsource(notebook_model.forward)
    if name == "no_scaling":
        attention_source = replace_once(
            attention_source,
            "    head_scores = head_scores / np.sqrt(d_head)",
            "    # head_scores = head_scores / np.sqrt(d_head)",
        )
    elif name == "no_mask":
        attention_source = replace_once(
            attention_source,
            "    future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)\n"
            "    head_masked_scores = np.where(future_mask, -np.inf, head_scores)",
            "    # future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)\n"
            "    # head_masked_scores = np.where(future_mask, -np.inf, head_scores)\n"
            "    head_masked_scores = head_scores",
        )
    elif name == "no_residual":
        forward_source = replace_once(
            forward_source, "        h = x + attention",
            "        # h = x + attention\n        h = attention",
        )
        forward_source = replace_once(
            forward_source, "        x = h + ffn_output",
            "        # x = h + ffn_output\n        x = ffn_output",
        )
    elif name != "baseline":
        raise ValueError(name)
    namespace = dict(notebook_model.__dict__)
    exec(compile(attention_source, f"<attention:{name}>", "exec"), namespace)
    exec(compile(forward_source, f"<forward:{name}>", "exec"), namespace)
    return namespace["forward"]


def first_attention_stats(ids, weights, config, *, scaling=True, mask=True):
    x = notebook_model.embedding(ids, weights["wte"], weights["wpe"])
    x = notebook_model.layer_norm(x, weights["blocks.0.ln_1.gamma"],
                                 weights["blocks.0.ln_1.beta"], config.layer_norm_eps)
    heads = []
    for letter in "QK":
        value = x @ weights[f"blocks.0.attn.W_{letter}"] + weights[f"blocks.0.attn.b_{letter}"]
        heads.append(value.reshape(len(ids), config.num_heads, -1).transpose(1, 0, 2))
    scores = heads[0] @ heads[1].swapaxes(-1, -2)
    if scaling:
        scores = scores / np.sqrt(config.d_model // config.num_heads)
    future = np.triu(np.ones((len(ids), len(ids)), dtype=bool), k=1)
    probabilities = notebook_model.softmax(np.where(future, -np.inf, scores) if mask else scores)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=-1)
    return dict(mean_entropy_nats=float(entropy.mean()),
                mean_row_max_probability=float(probabilities.max(axis=-1).mean()),
                mean_future_probability_mass=float((probabilities * future).sum(axis=-1).mean()),
                first_head_first_row=probabilities[0, 0].tolist())


def main():
    tokenizer = GPT2Tokenizer.from_pretrained(ROOT / "weights/gpt2", local_files_only=True)
    config, raw_weights = load_weights(ROOT / "weights/gpt2/gpt2.npz")
    # Preserve the checkpoint values while matching precision across all variants.
    weights = {key: value.astype(np.float64) for key, value in raw_weights.items()}
    del raw_weights
    ids = np.array(tokenizer.encode(PROMPT, add_special_tokens=False), dtype=np.int64)
    names = ("baseline", "no_scaling", "no_mask", "no_residual")
    outputs = {name: make_variant(name)(ids, weights, config, return_trace=True) for name in names}
    base = outputs["baseline"]["logits"]
    np.testing.assert_array_equal(base, notebook_model.forward(ids, weights, config)["logits"])
    all_top_ids = sorted({int(i) for output in outputs.values()
                          for i in np.argsort(output["logits"][-1])[-5:]})
    rows = {}
    for name, output in outputs.items():
        logits = output["logits"]
        assert logits.shape == (len(ids), config.vocab_size) and np.isfinite(logits).all()
        difference = logits[-1] - base[-1]
        probabilities = notebook_model.softmax(logits[-1])
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0)
        top_ids = np.argsort(logits[-1])[-5:][::-1]
        rows[name] = dict(
            top5=[dict(token_id=int(i), token=tokenizer.decode([int(i)]),
                       logit=float(logits[-1, i]), probability=float(probabilities[i])) for i in top_ids],
            last_position_max_absolute_difference=float(np.abs(difference).max()),
            last_position_mean_absolute_difference=float(np.abs(difference).mean()),
            last_position_centered_mean_absolute_difference=float(np.abs(difference - difference.mean()).mean()),
            last_position_probability_total_variation=float(np.abs(probabilities - notebook_model.softmax(base[-1])).sum() / 2),
            block_last_hidden_max_difference=[float(np.abs(a["output"][-1] - b["output"][-1]).max())
                                              for a, b in zip(output["trace"]["blocks"], outputs["baseline"]["trace"]["blocks"])],
            candidate_logits={str(i): float(logits[-1, i]) for i in all_top_ids},
        )
    baseline_prefix = make_variant("baseline")(ids[:1], weights, config)["logits"][0]
    maskless_prefix = make_variant("no_mask")(ids[:1], weights, config)["logits"][0]
    prefix_check = dict(
        baseline_first_position_max_difference=float(np.abs(base[0] - baseline_prefix).max()),
        no_mask_first_position_max_difference=float(np.abs(outputs["no_mask"]["logits"][0] - maskless_prefix).max()),
    )
    assert prefix_check["baseline_first_position_max_difference"] < 1e-9
    assert prefix_check["no_mask_first_position_max_difference"] > 1e-3
    assert rows["no_mask"]["block_last_hidden_max_difference"][0] < 1e-10
    assert rows["no_mask"]["block_last_hidden_max_difference"][1] > 1e-3
    import torch
    from transformers import GPT2LMHeadModel
    torch.set_num_threads(1)
    reference = GPT2LMHeadModel.from_pretrained(
        ROOT / "weights/gpt2", local_files_only=True, attn_implementation="eager",
    ).double().eval()
    with torch.inference_mode():
        expected = reference(torch.tensor(ids[None, :]), use_cache=False).logits[0].numpy()
    np.testing.assert_allclose(base, expected, atol=1e-9, rtol=1e-9)
    report = dict(
        prompt=PROMPT, token_ids=ids.tolist(),
        tokens=[tokenizer.decode([int(i)]) for i in ids],
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=json.loads((ROOT / "weights/gpt2/source.json").read_text()),
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        settings=dict(dtype="float64 (checkpoint float32 values converted without retraining)",
                      layers=config.num_layers, heads=config.num_heads, d_head=config.d_model // config.num_heads,
                      vocabulary_size=config.vocab_size, kv_cache=False, dropout=False,
                      logits_scope="last input position; full vocabulary", softmax_temperature=1.0,
                      residual_ablation="both additions in every block"),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ("notebook_model.py", "SelfAttention/multi_head_self_attention.py")},
        baseline_vs_hf_max_absolute_difference=float(np.abs(base - expected).max()),
        variants=rows,
        first_block_attention={
            "baseline": first_attention_stats(ids, weights, config),
            "no_scaling": first_attention_stats(ids, weights, config, scaling=False),
            "no_mask": first_attention_stats(ids, weights, config, mask=False),
        },
        prefix_invariance=prefix_check,
    )
    path = ROOT / "validation/gpt2_ablation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
