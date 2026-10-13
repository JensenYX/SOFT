import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from statistics import mean


RUBRICS = {
    "emotion": "Reward a response that matches the user's current emotional state, avoids over-escalation and generic therapy language, and remains useful. Penalize responses that ignore fragile, angry, guarded, or softening voice cues.",
    "prosody": "Reward a compact response that gives the right immediate action when the voice is pressed or hurried. Penalize long analysis, hedging, delayed action, or a comprehensive tutorial under time pressure.",
    "preference": "Reward a response that follows the stable earlier preference and interaction style after intervening turns. Penalize generic comprehensive answers, over-explaining, or ignoring the user's preferred way of being helped.",
}


PROMPT = """You are evaluating one anonymous response from a spoken dialogue system. Do not infer or reward system identity.

Context:
Session: {session_id}
Scenario: {scenario}
Turn: {turn}
Total turns: {total_turns}
User transcript: {user_text}
Control condition: {condition}
Control state: {state}
Recent method-specific history: {history}

Anonymous response:
{response}

Focus rubric:
{rubric}

Use half-point increments from 1.0 to 5.0. A 5.0 is near-ideal, 4.0 is clearly good with a minor weakness, 3.0 is usable but ordinary, 2.0 misses an important requirement, and 1.0 is incoherent, off-topic, or contradictory. Use 3.0 as the ordinary baseline and reserve 4.5 or 5.0 for exceptional responses.

Return only JSON with numeric fields score, focus, memory, style and a short string field reason."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--methods", default="short,full,memo,soft,soft_fc")
    parser.add_argument("--focus", choices=("auto", *RUBRICS), default="auto")
    parser.add_argument("--turns", default="")
    parser.add_argument("--output", default="scores.json")
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_results(path):
    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.glob("*.json"))
    documents = []
    for result_path in paths:
        document = json.loads(result_path.read_text(encoding="utf-8"))
        if "session_id" in document and "methods" in document:
            documents.append(document)
    if not documents:
        raise ValueError("No session result files found")
    return documents


def selected_turns(document, override):
    if override:
        return [int(value) for value in override.split(",") if value.strip()]
    key_turns = document.get("key_turns", [])
    if key_turns:
        return [int(value) for value in key_turns]
    return sorted(int(value) for value in document.get("turn_metadata", {}))


def recent_history(document, method, turn):
    metadata = document.get("turn_metadata", {})
    method_turns = document["methods"][method].get("turns", {})
    parts = []
    prior = sorted(value for value in (int(key) for key in metadata) if value < turn)[-2:]
    for turn_id in prior:
        user = metadata[str(turn_id)].get("text", "")[:160]
        assistant = method_turns.get(str(turn_id), {}).get("response", "")[:240]
        parts.append(f"User: {user} Assistant: {assistant}")
    return " | ".join(parts) if parts else "None"


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def valid_score(value):
    value = float(value)
    if not math.isfinite(value) or value < 1.0 or value > 5.0 or abs(value * 2 - round(value * 2)) > 1e-6:
        raise ValueError(f"Invalid score: {value}")
    return value


def call_judge(client, model, prompt, retries):
    error = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=220,
            )
            payload = extract_json(completion.choices[0].message.content)
            return {
                "score": valid_score(payload["score"]),
                "focus": valid_score(payload["focus"]),
                "memory": valid_score(payload["memory"]),
                "style": valid_score(payload["style"]),
                "reason": str(payload.get("reason", ""))[:240],
            }
        except Exception as exception:
            error = exception
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge failed after {retries} attempts: {error}")


def build_prompt(document, method, turn, focus):
    metadata = document["turn_metadata"][str(turn)]
    response = document["methods"][method]["turns"][str(turn)].get("response", "")
    if not response:
        raise ValueError(f"Empty response for {document['session_id']} {method} turn {turn}")
    return PROMPT.format(
        session_id=document["session_id"],
        scenario=document.get("scenario", ""),
        turn=turn,
        total_turns=len(document.get("turn_metadata", {})),
        user_text=metadata.get("text", ""),
        condition=metadata.get("condition", ""),
        state=metadata.get("state", ""),
        history=recent_history(document, method, turn),
        response=response[:3000],
        rubric=RUBRICS[focus],
    )


def summarize(rows, methods):
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        summary[method] = {
            "count": len(selected),
            "overall": mean(row["score"] for row in selected),
            "control": mean(row["focus"] for row in selected),
            "memory": mean(row["memory"] for row in selected),
            "style": mean(row["style"] for row in selected),
        }
    return summary


def main():
    args = parse_args()
    documents = load_results(args.results)
    methods = list(dict.fromkeys(value.strip() for value in args.methods.split(",") if value.strip()))
    tasks = []
    for document in documents:
        available = document.get("methods", {})
        focus = args.focus if args.focus != "auto" else document.get("axis", "")
        if focus not in RUBRICS:
            raise ValueError(f"Unknown focus for {document['session_id']}: {focus}")
        for method in methods:
            if method not in available:
                continue
            for turn in selected_turns(document, args.turns):
                if str(turn) in available[method].get("turns", {}):
                    tasks.append((document, method, turn, focus))
    if args.check:
        for document, method, turn, focus in tasks:
            build_prompt(document, method, turn, focus)
        print(f"validated {len(documents)} result files and {len(tasks)} judge tasks")
        return
    if not tasks:
        raise ValueError("No judge tasks selected")
    if not args.base_url or not args.api_key or not args.model:
        raise ValueError("JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL are required")
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rows = []
    for document, method, turn, focus in tasks:
        prompt = build_prompt(document, method, turn, focus)
        scores = call_judge(client, args.model, prompt, args.retries)
        row = {
            "session_id": document["session_id"],
            "turn": turn,
            "method": method,
            "focus_name": focus,
            **scores,
        }
        rows.append(row)
        print(f"{document['session_id']} turn {turn}: {method} {scores['score']}/{scores['focus']}")
        time.sleep(args.sleep)
    payload = {
        "judge_model": args.model,
        "summary": summarize(rows, methods),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
