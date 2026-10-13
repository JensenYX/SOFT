import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fast_weights import FastWeights, install_hooks, remove_hooks, resolve_module


METHODS = ("short", "full", "memo", "soft", "soft_fc")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data", default="data/sessions.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--session", default="all")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", default="qwen3_omni_moe")
    parser.add_argument("--template-type", default="qwen3_omni")
    parser.add_argument("--forward-path", default="thinker")
    parser.add_argument("--layer-path", default="thinker.model.layers")
    parser.add_argument("--layers", default="38,40,42,44,46")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--lr-decay", type=float, default=0.92)
    parser.add_argument("--weight-decay", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--kl-mix", type=float, default=0.5)
    parser.add_argument("--residual-cap", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions", [])
    if not sessions:
        raise ValueError("The manifest contains no sessions")
    seen = set()
    total_turns = 0
    for session in sessions:
        session_id = session["session_id"]
        if not session_id or Path(session_id).name != session_id:
            raise ValueError(f"Invalid session ID: {session_id}")
        if session_id in seen:
            raise ValueError(f"Duplicate session: {session_id}")
        seen.add(session_id)
        turns = session.get("turns", [])
        if not turns:
            raise ValueError(f"Session has no turns: {session_id}")
        turn_ids = set()
        ordered_turns = []
        for turn in turns:
            turn_id = int(turn["turn"])
            if turn_id in turn_ids:
                raise ValueError(f"Duplicate turn in {session_id}: {turn_id}")
            turn_ids.add(turn_id)
            ordered_turns.append(turn_id)
            audio = Path(turn["audio"])
            if audio.is_absolute():
                raise ValueError(f"Absolute audio path in {session_id}: {audio}")
            audio_path = (path.parent / audio).resolve()
            if path.parent.resolve() not in audio_path.parents:
                raise ValueError(f"Audio path leaves the data directory in {session_id}: {audio}")
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            with wave.open(str(audio_path), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getframerate() <= 0 or stream.getnframes() == 0:
                    raise ValueError(f"Expected a nonempty mono WAV: {audio}")
            turn["_audio_path"] = str(audio_path)
            total_turns += 1
        if ordered_turns != sorted(ordered_turns):
            raise ValueError(f"Turns are not ordered in {session_id}")
        unknown_key_turns = set(map(int, session.get("key_turns", []))) - turn_ids
        if unknown_key_turns:
            raise ValueError(f"Unknown key turns in {session_id}: {sorted(unknown_key_turns)}")
    return sessions, total_turns


def selected_methods(value):
    methods = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {','.join(unknown)}")
    return methods


def stable_seed(base_seed, *parts):
    value = "::".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def move_value(value, key, device):
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            return None
    if key in {"input_ids", "attention_mask"} and tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if key == "position_ids" and tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    if key == "input_features":
        tensor = tensor.to(torch.bfloat16)
    return tensor


def encode_messages(engine, messages, audios, device):
    request = {"messages": messages}
    if audios:
        request["audios"] = audios
    encoded = engine.template.encode(request)
    inputs = {}
    keys = ("input_ids", "attention_mask", "input_features", "feature_attention_mask", "audio_feature_lengths")
    for key in keys:
        if key in encoded:
            value = move_value(encoded[key], key, device)
            if value is not None:
                inputs[key] = value
    if "input_ids" not in inputs:
        raise ValueError("The template did not return input_ids")
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def audio_context(engine, history, device, system_prompt):
    messages = []
    audios = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for entry in history:
        messages.append({"role": "user", "content": "<audio>"})
        audios.append(entry["audio"])
        if entry.get("response"):
            messages.append({"role": "assistant", "content": entry["response"]})
    return encode_messages(engine, messages, audios, device)


def text_context(engine, records, device, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for record in records:
        content = f"Transcript: {record['text']}\nState note: {record['note']}"
        messages.append({"role": "user", "content": content})
        if record.get("response"):
            messages.append({"role": "assistant", "content": record["response"]})
    return encode_messages(engine, messages, [], device)


def extract_response(text):
    text = (text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    elif "<think>" in text:
        return ""
    for marker in ("\nUser:", "\nHuman:", "\nAssistant:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def stop_token_ids(tokenizer):
    ids = []
    candidates = [tokenizer.eos_token_id]
    for token in ("<|im_end|>", "<|endoftext|>"):
        candidates.append(tokenizer.convert_tokens_to_ids(token))
    for token_id in candidates:
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids


def generate(model, tokenizer, inputs, args, stop_ids, fast_weights=None):
    handles = install_hooks(model, fast_weights, args.layer_path) if fast_weights is not None else []
    core = resolve_module(model, args.forward_path)
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "no_repeat_ngram_size": 8,
    }
    if stop_ids:
        kwargs["eos_token_id"] = stop_ids
        kwargs["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
    try:
        with torch.no_grad():
            output = core.generate(**inputs, **kwargs)
    finally:
        remove_hooks(handles)
    if hasattr(output, "sequences"):
        output = output.sequences
    prompt_length = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    return extract_response(text)


def common_suffix_length(teacher_ids, student_ids):
    length = min(teacher_ids.shape[-1], student_ids.shape[-1])
    equal = teacher_ids[0, -length:] == student_ids[0, -length:]
    mismatches = (~equal).nonzero(as_tuple=False).flatten()
    if mismatches.numel() == 0:
        return length
    return length - int(mismatches[-1].item()) - 1


def position_ids_for(core, inputs):
    attention_mask = inputs["attention_mask"]
    if not hasattr(core, "get_rope_index"):
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        return positions
    feature_mask = inputs.get("feature_attention_mask")
    audio_lengths = feature_mask.sum(dim=1) if feature_mask is not None else inputs.get("audio_feature_lengths")
    values = {
        "input_ids": inputs["input_ids"],
        "attention_mask": attention_mask,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "audio_seqlens": audio_lengths,
        "audio_feature_lengths": audio_lengths,
        "second_per_grids": inputs.get("video_second_per_grid"),
        "video_second_per_grid": inputs.get("video_second_per_grid"),
        "use_audio_in_video": False,
    }
    parameters = inspect.signature(core.get_rope_index).parameters
    kwargs = {key: value for key, value in values.items() if key in parameters and value is not None}
    positions, _ = core.get_rope_index(**kwargs)
    return positions


def align_position_ids(core, teacher_inputs, student_inputs, shared):
    teacher_positions = position_ids_for(core, teacher_inputs)
    student_positions = position_ids_for(core, student_inputs).clone()
    if teacher_positions.shape[-1] != teacher_inputs["input_ids"].shape[-1]:
        raise ValueError("Teacher position IDs do not match the encoded sequence")
    if student_positions.shape[-1] != student_inputs["input_ids"].shape[-1]:
        raise ValueError("Student position IDs do not match the encoded sequence")
    student_positions[..., -shared:] = teacher_positions[..., -shared:]
    teacher_inputs["position_ids"] = teacher_positions
    student_inputs["position_ids"] = student_positions


def symmetric_kl(teacher_logits, student_logits, temperature, mix):
    teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_prob = teacher_log.exp()
    student_prob = student_log.exp()
    forward = (teacher_prob * (teacher_log - student_log)).sum(dim=-1)
    reverse = (student_prob * (student_log - teacher_log)).sum(dim=-1)
    return (mix * forward + (1.0 - mix) * reverse).mean() * temperature ** 2


def adapt(model, teacher_inputs, student_inputs, fast_weights, optimizer, args):
    core = resolve_module(model, args.forward_path)
    shared = common_suffix_length(teacher_inputs["input_ids"], student_inputs["input_ids"])
    if shared <= 0:
        raise ValueError("Teacher and student inputs have no shared suffix")
    align_position_ids(core, teacher_inputs, student_inputs, shared)
    with torch.no_grad():
        teacher_logits = core(**teacher_inputs).logits.detach()
    final_loss = None
    for _ in range(args.steps):
        handles = install_hooks(model, fast_weights, args.layer_path)
        try:
            student_logits = core(**student_inputs).logits
        finally:
            remove_hooks(handles)
        loss = symmetric_kl(
            teacher_logits[:, -shared:],
            student_logits[:, -shared:],
            args.temperature,
            args.kl_mix,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast_weights.parameters(), args.grad_clip)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return final_loss, shared


def load_engine(args):
    swift_root = Path(__file__).resolve().parent / "ms-swift"
    sys.path.insert(0, str(swift_root))
    from swift.model import get_model_processor
    from swift.template import get_template

    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "false")
    model, processor = get_model_processor(
        args.model,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    template = get_template(
        processor,
        template_type=args.template_type,
        enable_thinking=False if args.disable_thinking else None,
        response_prefix="" if args.disable_thinking else None,
    )
    engine = SimpleNamespace(model=model, processor=processor, template=template)
    engine.model.eval()
    engine.template.model = engine.model
    for parameter in engine.model.parameters():
        parameter.requires_grad_(False)
    layers = resolve_module(engine.model, args.layer_path)
    selected = [int(value) for value in args.layers.split(",")]
    if not selected or min(selected) < 0 or max(selected) >= len(layers):
        raise ValueError(f"Layer selection is incompatible with a {len(layers)}-layer model")
    tokenizer = engine.processor.tokenizer
    return engine, tokenizer, stop_token_ids(tokenizer)


def make_history_entry(turn):
    return {
        "turn": int(turn["turn"]),
        "audio": turn["_audio_path"],
        "text": turn.get("text", ""),
    }


def run_baseline(engine, tokenizer, stop_ids, session, method, args):
    history = []
    turns = {}
    started = time.time()
    for turn in session["turns"]:
        history.append(make_history_entry(turn))
        context = history if method == "full" else history[-args.short_window:]
        inputs = audio_context(engine, context, args.device, args.system_prompt)
        response = generate(engine.model, tokenizer, inputs, args, stop_ids)
        history[-1]["response"] = response
        turns[str(turn["turn"])] = {
            "response": response,
            "context_tokens": int(inputs["input_ids"].shape[-1]),
        }
    return {"elapsed_seconds": round(time.time() - started, 2), "turns": turns}


def generate_note(engine, tokenizer, stop_ids, turn, args):
    messages = [
        {
            "role": "system",
            "content": "Write one short factual note about the user's affect, urgency, speaking style, stable preference, and active interaction constraints.",
        },
        {"role": "user", "content": "<audio>"},
    ]
    inputs = encode_messages(engine, messages, [turn["_audio_path"]], args.device)
    return generate(engine.model, tokenizer, inputs, args, stop_ids)


def run_memo(engine, tokenizer, stop_ids, session, args):
    records = []
    turns = {}
    started = time.time()
    for turn in session["turns"]:
        note = generate_note(engine, tokenizer, stop_ids, turn, args)
        records.append({"text": turn.get("text", ""), "note": note})
        inputs = text_context(engine, records, args.device, args.system_prompt)
        response = generate(engine.model, tokenizer, inputs, args, stop_ids)
        records[-1]["response"] = response
        turns[str(turn["turn"])] = {
            "response": response,
            "note": note,
            "context_tokens": int(inputs["input_ids"].shape[-1]),
        }
    return {"elapsed_seconds": round(time.time() - started, 2), "turns": turns}


def run_soft(engine, tokenizer, stop_ids, session, method, args, layers):
    torch.manual_seed(stable_seed(args.seed, session["session_id"], method))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stable_seed(args.seed, session["session_id"], method))
    fast_weights = FastWeights(args.hidden_size, args.rank, layers, args.residual_cap).to(args.device)
    history = []
    turns = {}
    started = time.time()
    for turn_index, turn in enumerate(session["turns"]):
        history.append(make_history_entry(turn))
        student_history = history[-args.short_window:]
        teacher_inputs = audio_context(engine, history, args.device, args.system_prompt)
        student_inputs = audio_context(engine, student_history, args.device, args.system_prompt)
        if turn_index:
            fast_weights.decay_(args.weight_decay)
        learning_rate = args.learning_rate * args.lr_decay ** turn_index
        optimizer = torch.optim.Adam(fast_weights.parameters(), lr=learning_rate)
        loss, shared = adapt(engine.model, teacher_inputs, student_inputs, fast_weights, optimizer, args)
        decode_history = history if method == "soft_fc" else student_history
        decode_inputs = audio_context(engine, decode_history, args.device, args.system_prompt)
        response = generate(engine.model, tokenizer, decode_inputs, args, stop_ids, fast_weights)
        history[-1]["response"] = response
        norm = sum(parameter.detach().float().pow(2).sum() for parameter in fast_weights.parameters()).sqrt()
        turns[str(turn["turn"])] = {
            "response": response,
            "loss": loss,
            "learning_rate": learning_rate,
            "fast_weight_norm": float(norm.cpu()),
            "shared_tokens": shared,
            "teacher_tokens": int(teacher_inputs["input_ids"].shape[-1]),
            "decode_tokens": int(decode_inputs["input_ids"].shape[-1]),
        }
    return {
        "elapsed_seconds": round(time.time() - started, 2),
        "fast_weight_parameters": fast_weights.parameter_count,
        "turns": turns,
    }


def session_record(session, methods, model_name, args, layers):
    metadata = {}
    for turn in session["turns"]:
        metadata[str(turn["turn"])] = {
            "text": turn.get("text", ""),
            "condition": turn.get("condition", ""),
            "state": turn.get("state", ""),
        }
    return {
        "session_id": session["session_id"],
        "scenario": session.get("scenario", ""),
        "axis": session.get("axis", ""),
        "key_turns": session.get("key_turns", []),
        "turn_metadata": metadata,
        "config": {
            "model": model_name,
            "methods": methods,
            "layers": layers,
            "hidden_size": args.hidden_size,
            "rank": args.rank,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "lr_decay": args.lr_decay,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "kl_mix": args.kl_mix,
            "residual_cap": args.residual_cap,
            "grad_clip": args.grad_clip,
            "short_window": args.short_window,
        },
        "methods": {},
    }


def main():
    args = parse_args()
    sessions, total_turns = load_manifest(args.data)
    methods = selected_methods(args.methods)
    if args.check:
        print(f"validated {len(sessions)} sessions and {total_turns} turns")
        return
    if not args.model:
        raise ValueError("--model is required unless --check is used")
    if args.session != "all":
        requested = {item.strip() for item in args.session.split(",") if item.strip()}
        sessions = [session for session in sessions if session["session_id"] in requested]
    if not sessions:
        raise ValueError("No sessions selected")
    layers = [int(value) for value in args.layers.split(",")]
    engine, tokenizer, stop_ids = load_engine(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).name
    for session in sessions:
        record = session_record(session, methods, model_name, args, layers)
        for method in methods:
            print(f"{session['session_id']}: {method}")
            if method in {"short", "full"}:
                result = run_baseline(engine, tokenizer, stop_ids, session, method, args)
            elif method == "memo":
                result = run_memo(engine, tokenizer, stop_ids, session, args)
            else:
                result = run_soft(engine, tokenizer, stop_ids, session, method, args, layers)
            record["methods"][method] = result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        path = output_dir / f"{session['session_id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
