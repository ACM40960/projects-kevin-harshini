"""
src/llm.py

Central LLM provider interface for QuantRAG.
All phases call get_llm_response() — never import Groq or Anthropic directly.

To switch provider:  change LLM_PROVIDER in your .env file.
  LLM_PROVIDER=groq       → uses Groq — free
  LLM_PROVIDER=anthropic  → uses Claude Sonnet           — Phase 3+
  LLM_PROVIDER=ollama     → uses local Llama 3.1         — offline

Nothing else in the codebase needs to change when you switch.
"""

import os
import re
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def get_llm_response(prompt: str, max_tokens: int = 1024) -> str:
    if PROVIDER == "groq":
        return _call_groq(prompt, max_tokens)
    elif PROVIDER == "anthropic":
        return _call_anthropic(prompt, max_tokens)
    elif PROVIDER == "ollama":
        return _call_ollama(prompt, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. Choose: groq | anthropic | ollama"
        )


# ── Groq ───────────────────────────────────────────────────────────────────

# Groq's free-tier catalog changes frequently. Try each in order —
# if one is deprecated or rate-limited, fall through automatically.
GROQ_MODEL_CANDIDATES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


def _call_groq(prompt: str, max_tokens: int) -> str:
    """
    Groq free tier with automatic model fallback AND correct reasoning
    control per model family.

    KEY FIX: "/no_think" is a Qwen-specific directive — it does nothing
    for OpenAI's gpt-oss models, which use a separate "reasoning_effort"
    parameter instead. Without setting this, gpt-oss models can spend
    their entire token budget on internal reasoning and return empty
    content. We now set the correct low-reasoning mode per model family,
    so every candidate actually gets a fair chance to answer directly.
    """
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    last_error = None
    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )

            if "gpt-oss" in model_name:
                # OpenAI gpt-oss reasoning control — "low" keeps the model
                # from spending the whole budget on hidden reasoning.
                kwargs["reasoning_effort"] = "low"
            elif "qwen" in model_name:
                # Qwen's own reasoning-disable directive
                kwargs["messages"] = [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ]

            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            if not cleaned:
                raise ValueError(
                    f"Model '{model_name}' returned empty content even "
                    f"with reasoning_effort=low (max_tokens={max_tokens})"
                )

            return cleaned

        except Exception as e:
            last_error = e
            console.print(
                f"[yellow]  ⚠ Groq model '{model_name}' failed "
                f"({str(e)[:70]}) — trying next candidate[/yellow]"
            )
            continue

    raise RuntimeError(
        f"All Groq model candidates failed. Last error: {last_error}\n"
        f"Check https://console.groq.com/docs/models for current models."
    )


# ── Anthropic ──────────────────────────────────────────────────────────────
def _call_anthropic(prompt: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: uv add anthropic")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Ollama (local fallback) ────────────────────────────────────────────────
def _call_ollama(prompt: str, max_tokens: int) -> str:
    try:
        import httpx
    except ImportError:
        raise ImportError("Run: uv add httpx")

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1", "prompt": prompt, "stream": False,
            "options": {"num_predict": max_tokens}
        },
        timeout=120.0
    )
    return response.json()["response"]


# ── Structured JSON output (Phase 3 only) ─────────────────────────────────
def get_structured_view(prompt: str) -> dict:
    import json
    if PROVIDER == "anthropic":
        return _structured_anthropic(prompt)
    else:
        return _structured_via_prompt(prompt)


def _structured_anthropic(prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tools = [{
        "name": "extract_investment_view",
        "description": "Extract a structured investment view from filing text",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction":  {"type": "string",  "enum": ["bullish", "bearish", "neutral"]},
                "magnitude":  {"type": "number"},
                "confidence": {"type": "number"},
                "reasoning":  {"type": "string"},
                "citation":   {"type": "string"},
            },
            "required": ["direction", "magnitude", "confidence", "reasoning", "citation"]
        }
    }]
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512, tools=tools,
        tool_choice={"type": "tool", "name": "extract_investment_view"},
        messages=[{"role": "user", "content": prompt}]
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Claude did not return a tool_use block")


def _structured_via_prompt(prompt: str) -> dict:
    import json
    json_prompt = prompt + """

IMPORTANT: Respond with ONLY a valid JSON object, no explanation:
{
  "direction": "bullish" or "bearish" or "neutral",
  "magnitude": number between -0.15 and 0.15,
  "confidence": number between 0.0 and 1.0,
  "reasoning": "one sentence",
  "citation": "source reference"
}"""
    raw = get_llm_response(json_prompt, max_tokens=256)
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from response:\n{raw}")