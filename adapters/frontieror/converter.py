"""The fixed converter model and the exchange protocol with it.

One model prepares the input for every evaluated model, once, offline. Its
answers are frozen and released, so what OptiMUS receives is a constant of the
experiment rather than something each evaluated model produced for itself.

The converter is deliberately not one of the models under evaluation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import openai

from . import config, prompts

# DeepSeek's API is OpenAI-compatible. The exact model id is configurable because
# it is a released detail of the setup, not an implementation choice.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_EXCHANGES = int(os.environ.get("ADAPTER_MAX_EXCHANGES", "12"))

_BLOCK = re.compile(r"```(?P<tag>[\w./-]+)\s*\n(?P<body>.*?)```", re.DOTALL)
ANSWER_FILES = ("transform.py", "parameters.json", "targets.json")


class ProtocolError(Exception):
    """The converter replied with something that is neither an exploration nor a
    complete answer."""


def model_id() -> str:
    return config.CONVERTER_MODEL or os.environ.get("ADAPTER_CONVERTER_MODEL") or DEFAULT_MODEL


def base_url() -> str:
    """Accept either an OpenAI-style base or a full chat-completions endpoint.

    DecisionBrain's .env stores the latter (LLM_MODEL_URL), and the OpenAI SDK
    appends the path itself, so the suffix is trimmed rather than duplicated.
    """
    url = os.environ.get("ADAPTER_BASE_URL") or os.environ.get("LLM_MODEL_URL") or DEFAULT_BASE_URL
    return url.removesuffix("/chat/completions").rstrip("/")


def client() -> openai.OpenAI:
    """Resolve the key without ever committing one.

    LLM_API_KEY is the name DecisionBrain already uses on this host, so sourcing
    its .env is enough to run the converter.
    """
    key = (
        os.environ.get("ADAPTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )
    config_path = config.REPO_ROOT / "config.json"
    if not key and config_path.is_file():
        key = json.loads(config_path.read_text(encoding="utf-8")).get("converter_api_key", "")
    if len(key) < 10:
        raise RuntimeError(
            "No converter API key. Set ADAPTER_API_KEY or LLM_API_KEY, or add "
            "converter_api_key to config.json."
        )
    return openai.OpenAI(api_key=key, base_url=base_url())


def parse_reply(text: str) -> tuple[str | None, dict[str, str] | None]:
    """Split a reply into either an exploration snippet or a complete answer.

    A reply that is neither -- a partial answer, prose only, both phases at once
    -- is a protocol error and is sent back for correction rather than guessed at.
    """
    blocks = {m.group("tag"): m.group("body") for m in _BLOCK.finditer(text)}
    explore = blocks.get("explore")
    files = {name: blocks[name] for name in ANSWER_FILES if name in blocks}

    if explore is not None and not files:
        return explore, None
    if files and explore is None:
        missing = [name for name in ANSWER_FILES if name not in files]
        if missing:
            raise ProtocolError(
                f"Your answer is incomplete: missing {missing}. Reply with all "
                "three blocks, or with a single ```explore block if you still "
                "need to inspect the instance."
            )
        return None, files
    if explore is not None and files:
        raise ProtocolError(
            "You mixed both phases in one reply. Send either one ```explore "
            "block, or the three answer blocks -- never both."
        )
    raise ProtocolError(
        "No recognised block found. Reply with one ```explore block, or with "
        "```transform.py, ```parameters.json and ```targets.json."
    )


def converse(problem_md: str, explore_fn, feedback: list[str]) -> dict[str, str]:
    """Run the exchange until the converter produces a complete answer.

    `explore_fn(snippet) -> str` executes inspection code against the instance
    and returns its stdout. `feedback` carries verbatim rejections from earlier
    attempts, so the converter sees exactly what the checker said.
    """
    api = client()
    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user", "content": prompts.initial(problem_md)},
    ]
    for rejection in feedback:
        messages.append({"role": "user", "content": prompts.rejection(rejection)})

    for _ in range(MAX_EXCHANGES):
        reply = api.chat.completions.create(
            model=model_id(), messages=messages, temperature=0
        ).choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})

        try:
            snippet, files = parse_reply(reply)
        except ProtocolError as exc:
            messages.append({"role": "user", "content": str(exc)})
            continue

        if files is not None:
            return files
        messages.append(
            {"role": "user", "content": f"Output of your code:\n\n-----\n{explore_fn(snippet)}\n-----"}
        )

    raise ProtocolError(
        f"Converter did not produce a complete answer within {MAX_EXCHANGES} exchanges."
    )
