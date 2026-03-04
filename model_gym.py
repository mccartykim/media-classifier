#!/usr/bin/env python3
"""Model gym for benchmarking Ollama models as media classifier arbiters.

Tests multiple LLM models against curated classification cases with
pre-gathered evidence. Measures accuracy, latency, and consistency.

Usage:
    python3 hosts/historian/model_gym.py --host http://localhost:11434
    python3 hosts/historian/model_gym.py --host http://total-eclipse.nebula:11434
    python3 hosts/historian/model_gym.py --models qwen3:0.6b,qwen3:1.7b
    python3 hosts/historian/model_gym.py --pull  # Pull missing models first
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# =============================================================================
# Test dataset — each case includes filename, expected answer, and evidence
# =============================================================================

TEST_CASES = [
    {
        "filename": "Kingdom.S01E01.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Kingdom (JP anime, TV, 38 eps, popularity 52k)",
            "ffprobe": "24:12, audio: [jpn, eng], sub_formats: [ass]",
            "wikipedia": "Kingdom is a Japanese manga series written and illustrated by Yasuhisa Hara.",
        },
    },
    {
        "filename": "Riget.S01E01.DVDRip.mkv",
        "expected": "tv",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "57:00, audio: [dan], sub_formats: [subrip]",
            "wikipedia": "The Kingdom is a Danish television series created by Lars von Trier.",
        },
    },
    {
        "filename": "Castlevania.S04E08.1080p.NF.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Castlevania (JP anime, ONA, 32 eps, popularity 30k)",
            "ffprobe": "24:00, audio: [eng, jpn], sub_formats: [subrip]",
            "wikipedia": "Castlevania is an American adult animated dark fantasy action television series.",
        },
    },
    {
        "filename": "The.Office.S03E12.720p.mkv",
        "expected": "tv",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "22:00, audio: [eng], sub_formats: [subrip]",
            "wikipedia": "The Office is an American mockumentary sitcom television series.",
        },
    },
    {
        "filename": "One.Piece.Film.Red.2022.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "One Piece Film: Red (JP anime, MOVIE, popularity 80k)",
            "ffprobe": "112:00, audio: [jpn], sub_formats: [ass]",
            "wikipedia": "One Piece Film: Red is a 2022 Japanese animated fantasy action adventure film.",
        },
    },
    {
        "filename": "The.Matrix.1999.2160p.UHD.BluRay.mkv",
        "expected": "movie",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "136:00, audio: [eng], sub_formats: [subrip]",
            "wikipedia": "The Matrix is a 1999 science fiction action film.",
        },
    },
    {
        "filename": "Vinland.Saga.S02E10.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Vinland Saga (JP anime, TV, 24 eps, popularity 95k)",
            "ffprobe": "24:00, audio: [jpn, eng], sub_formats: [ass]",
            "wikipedia": "Vinland Saga is a Japanese historical manga series.",
        },
    },
    {
        "filename": "Vikings.S04E15.720p.mkv",
        "expected": "tv",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "44:00, audio: [eng], sub_formats: [subrip]",
            "wikipedia": "Vikings is a historical drama television series created for the History channel.",
        },
    },
    {
        "filename": "Blade.Runner.2049.2017.2160p.mkv",
        "expected": "movie",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "164:00, audio: [eng], sub_formats: [hdmv_pgs_subtitle]",
            "wikipedia": "Blade Runner 2049 is a 2017 American science fiction film directed by Denis Villeneuve.",
        },
    },
    {
        "filename": "Attack.on.Titan.S04E28.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Shingeki no Kyojin (JP anime, TV, 16 eps, popularity 500k)",
            "ffprobe": "24:00, audio: [jpn], sub_formats: [ass]",
            "wikipedia": "Attack on Titan is a Japanese manga series written and illustrated by Hajime Isayama.",
        },
    },
    {
        "filename": "Succession.S04E10.1080p.HMAX.mkv",
        "expected": "tv",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "58:00, audio: [eng], sub_formats: [subrip]",
            "wikipedia": "Succession is an American satirical comedy-drama television series created by Jesse Armstrong.",
        },
    },
    {
        "filename": "Spirited.Away.2001.1080p.BluRay.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Sen to Chihiro no Kamikakushi (JP anime, MOVIE, popularity 200k)",
            "ffprobe": "125:00, audio: [jpn, eng], sub_formats: [ass, subrip]",
            "wikipedia": "Spirited Away is a 2001 Japanese animated fantasy film written and directed by Hayao Miyazaki.",
        },
    },
    {
        "filename": "Parasite.2019.1080p.BluRay.mkv",
        "expected": "movie",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "132:00, audio: [kor], sub_formats: [subrip]",
            "wikipedia": "Parasite is a 2019 South Korean black comedy thriller film directed by Bong Joon-ho.",
        },
    },
    {
        "filename": "Mob.Psycho.100.S03E12.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Mob Psycho 100 III (JP anime, TV, 12 eps, popularity 120k)",
            "ffprobe": "24:00, audio: [jpn], sub_formats: [ass]",
            "wikipedia": "Mob Psycho 100 is a Japanese manga series written and illustrated by ONE.",
        },
    },
    {
        "filename": "Dark.S03E08.1080p.NF.mkv",
        "expected": "tv",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "55:00, audio: [deu, eng], sub_formats: [subrip]",
            "wikipedia": "Dark is a German science fiction thriller television series.",
        },
    },
    {
        "filename": "Cyberpunk.Edgerunners.S01E10.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Cyberpunk: Edgerunners (JP anime, ONA, 10 eps, popularity 150k)",
            "ffprobe": "24:00, audio: [jpn, eng], sub_formats: [ass, subrip]",
            "wikipedia": "Cyberpunk: Edgerunners is a 2022 anime series based on the video game Cyberpunk 2077.",
        },
    },
    {
        "filename": "Dune.Part.Two.2024.2160p.mkv",
        "expected": "movie",
        "evidence": {
            "anilist": "No anime match",
            "ffprobe": "166:00, audio: [eng], sub_formats: [hdmv_pgs_subtitle]",
            "wikipedia": "Dune: Part Two is a 2024 American epic science fiction film.",
        },
    },
    {
        "filename": "Bocchi.the.Rock.S01E12.1080p.mkv",
        "expected": "anime",
        "evidence": {
            "anilist": "Bocchi the Rock! (JP anime, TV, 12 eps, popularity 90k)",
            "ffprobe": "24:00, audio: [jpn], sub_formats: [ass]",
            "wikipedia": "Bocchi the Rock! is a Japanese four-panel manga series.",
        },
    },
]

MODELS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "gemma3:1b",
    "gemma3:4b",
    "phi4-mini:3.8b",
    "llama3.2:1b",
    "llama3.2:3b",
]

RUNS_PER_CASE = 3


def build_prompt(case):
    """Build the classification prompt with evidence context."""
    ev = case["evidence"]
    return (
        f"Classify this media file for a Jellyfin library.\n"
        f"Filename: {case['filename']}\n"
        f"AniList: {ev['anilist']}\n"
        f"ffprobe: {ev['ffprobe']}\n"
        f"Wikipedia: {ev['wikipedia']}\n"
        f"\nBased on ALL evidence above, classify as exactly one of: anime, tv, movie.\n"
        f'Respond with ONLY a JSON object: {{"category": "anime"|"tv"|"movie"}} /no_think'
    )


def query_model(host, model, prompt):
    """Query Ollama and return (category, latency_ms) or (None, latency_ms)."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": ["anime", "tv", "movie"]}},
            "required": ["category"],
        },
        "options": {"temperature": 0.1, "num_predict": 50},
    }).encode()

    start = time.monotonic()
    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        latency = (time.monotonic() - start) * 1000

        text = data.get("response", "").strip()
        result = json.loads(text)
        category = result.get("category", "").lower()
        if category in ("anime", "tv", "movie"):
            return category, latency
        return None, latency
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        print(f"    Error: {e}", file=sys.stderr)
        return None, latency


def check_model_available(host, model):
    """Check if model is available on the Ollama host."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        return model in available or any(model.split(":")[0] in m for m in available)
    except Exception:
        return False


def pull_model(host, model):
    """Pull a model via Ollama API."""
    print(f"  Pulling {model}...")
    try:
        payload = json.dumps({"name": model, "stream": False}).encode()
        req = urllib.request.Request(
            f"{host}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        print(f"  {model} pulled successfully")
        return True
    except Exception as e:
        print(f"  Failed to pull {model}: {e}", file=sys.stderr)
        return False


def run_benchmark(host, models, do_pull=False):
    """Run the full benchmark suite."""
    results = {}

    for model in models:
        if not check_model_available(host, model):
            if do_pull:
                if not pull_model(host, model):
                    print(f"  Skipping {model} (pull failed)")
                    continue
            else:
                print(f"  Skipping {model} (not available, use --pull)")
                continue

        print(f"\nTesting {model}...")
        model_results = []

        for case in TEST_CASES:
            prompt = build_prompt(case)
            case_runs = []

            for run in range(RUNS_PER_CASE):
                category, latency = query_model(host, model, prompt)
                case_runs.append({"answer": category, "latency": latency})
                # Print progress dot
                status = "." if category == case["expected"] else "X" if category else "?"
                print(status, end="", flush=True)

            model_results.append({
                "filename": case["filename"],
                "expected": case["expected"],
                "runs": case_runs,
            })

        print()  # Newline after dots
        results[model] = model_results

    return results


def analyze_results(results):
    """Analyze benchmark results and print comparison table."""
    summaries = []

    for model, cases in results.items():
        total_correct = 0
        total_runs = 0
        total_latency = 0
        consistent_cases = 0
        total_cases = len(cases)

        for case in cases:
            answers = [r["answer"] for r in case["runs"]]
            latencies = [r["latency"] for r in case["runs"]]

            correct = sum(1 for a in answers if a == case["expected"])
            total_correct += correct
            total_runs += len(answers)
            total_latency += sum(latencies)

            # Consistency: all runs gave same answer
            non_none = [a for a in answers if a is not None]
            if non_none and all(a == non_none[0] for a in non_none):
                consistent_cases += 1

        accuracy = (total_correct / total_runs * 100) if total_runs else 0
        avg_latency = (total_latency / total_runs) if total_runs else 0
        consistency = (consistent_cases / total_cases * 100) if total_cases else 0

        summaries.append({
            "model": model,
            "accuracy": accuracy,
            "avg_latency": avg_latency,
            "consistency": consistency,
        })

    # Sort by accuracy descending, then latency ascending
    summaries.sort(key=lambda s: (-s["accuracy"], s["avg_latency"]))

    # Find recommendation: smallest model with 95%+ accuracy and 90%+ consistency
    recommended = None
    for s in summaries:
        if s["accuracy"] >= 95 and s["consistency"] >= 90:
            recommended = s["model"]

    # Print table
    print()
    header = f"{'Model':<20} {'Accuracy':>10} {'Avg Latency':>12} {'Consistency':>13} {'Rec':>5}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for s in summaries:
        rec = "***" if s["model"] == recommended else ""
        print(
            f"{s['model']:<20} {s['accuracy']:>9.1f}% {s['avg_latency']:>10.0f}ms "
            f"{s['consistency']:>12.1f}% {rec:>5}"
        )

    print(sep)

    if recommended:
        print(f"\nRecommended: {recommended}")
    else:
        print("\nNo model met the 95% accuracy + 90% consistency threshold.")

    # Print per-case breakdown for best model
    if summaries:
        best = summaries[0]["model"]
        print(f"\nDetailed results for {best}:")
        for case in results[best]:
            answers = [r["answer"] for r in case["runs"]]
            correct = all(a == case["expected"] for a in answers)
            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] {case['filename']}: expected={case['expected']}, got={answers}")


def main():
    global RUNS_PER_CASE

    parser = argparse.ArgumentParser(description="Benchmark Ollama models for media classification")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--models", default=None, help="Comma-separated list of models to test")
    parser.add_argument("--pull", action="store_true", help="Pull missing models before testing")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CASE, help="Runs per test case")
    args = parser.parse_args()

    RUNS_PER_CASE = args.runs

    models = args.models.split(",") if args.models else MODELS

    # Check Ollama connectivity
    try:
        req = urllib.request.Request(f"{args.host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        print(f"Connected to {args.host}")
        print(f"Available models: {', '.join(available)}")
    except Exception as e:
        print(f"Cannot connect to Ollama at {args.host}: {e}", file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(args.host, models, do_pull=args.pull)

    if not results:
        print("No models were tested.", file=sys.stderr)
        sys.exit(1)

    analyze_results(results)


if __name__ == "__main__":
    main()
