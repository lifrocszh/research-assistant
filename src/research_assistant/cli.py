from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from .engine import ResearchRuntime
from .models import RunEvent, RuntimeLimits


class EventWriter:
    def __init__(self, target: str | None, append: bool = False) -> None:
        self.stream: TextIO | None = None
        self.owned = False
        if target == "-":
            self.stream = sys.stdout
        elif target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("a" if append else "w", encoding="utf-8")
            self.owned = True

    def __call__(self, event: RunEvent) -> None:
        if self.stream:
            self.stream.write(event.model_dump_json() + "\n")
            self.stream.flush()

    def close(self) -> None:
        if self.owned and self.stream:
            self.stream.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="start a research run")
    run.add_argument("question")
    run.add_argument("--thread", default=None)
    run.add_argument(
        "--mode",
        choices=["fixture", "live"],
        default="fixture",
        help="fixture (default): offline, deterministic, no LLM calls; live: adaptive model-guided research",
    )
    run.add_argument("--document", action="append", default=[])
    run.add_argument("--fixtures")
    _common(run, include_limits=True)
    resume = commands.add_parser("resume", help="continue a paused thread")
    resume.add_argument("thread")
    _common(resume)
    return parser


def _common(parser: argparse.ArgumentParser, *, include_limits: bool = False) -> None:
    parser.add_argument("--checkpoint", default=".research-assistant/checkpoints.deepagents.sqlite")
    parser.add_argument("--log-dir", default=".research-assistant/logs", help="directory for detailed per-run logs")
    parser.add_argument("--jsonl", metavar="PATH", help="event file; use - for stdout")
    parser.add_argument("--output", metavar="PATH", help="write final Markdown")
    parser.add_argument("--pause-after-turn", action="store_true")
    if include_limits:
        parser.add_argument("--max-parallel-agents", type=int, default=3)
        parser.add_argument("--max-total-agents", type=int, default=12)
        parser.add_argument("--max-research-depth", type=int, default=5)
        parser.add_argument("--max-runtime-seconds", type=float, default=180)
        parser.add_argument("--max-tool-calls-per-agent", type=int, default=8)
        parser.add_argument("--max-tokens-per-run", type=int, default=32_000)


def _limits(args: argparse.Namespace) -> RuntimeLimits:
    return RuntimeLimits(
        max_parallel_agents=args.max_parallel_agents,
        max_total_agents=args.max_total_agents,
        max_research_depth=args.max_research_depth,
        max_runtime_seconds=args.max_runtime_seconds,
        max_tool_calls_per_agent=args.max_tool_calls_per_agent,
        max_tokens_per_run=args.max_tokens_per_run,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    writer = EventWriter(args.jsonl, append=args.command == "resume")
    try:
        with ResearchRuntime(args.checkpoint, writer, args.log_dir) as runtime:
            if args.command == "run":
                state = runtime.run(
                    args.question,
                    thread_id=args.thread or str(uuid4()),
                    mode=args.mode,
                    documents=args.document,
                    fixture_path=args.fixtures,
                    limits=_limits(args),
                    pause_after_turn=args.pause_after_turn,
                )
            else:
                state = runtime.resume(args.thread, pause_after_turn=args.pause_after_turn)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        writer.close()
    destination = sys.stderr if args.jsonl == "-" else sys.stdout
    if state.status == "paused":
        print(f"Paused thread {state.thread_id} after turn {state.depth}.", file=destination)
        return
    answer = state.final_answer or ""
    if args.output:
        Path(args.output).write_text(answer, encoding="utf-8")
    else:
        print(answer, end="", file=destination)


if __name__ == "__main__":
    main()
