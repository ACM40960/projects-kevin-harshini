"""
src/llm.py

Central LLM provider interface for QuantRAG.
All phases call get_llm_response() — never import Groq or Anthropic directly.

To switch provider:  change LLM_PROVIDER in your .env file.
  LLM_PROVIDER=groq       → uses Groq (Llama 3.1 70B) — free
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

# ── Read which provider to use from .env ───────────────────────────────────
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def get_llm_response(prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to whichever LLM provider is configured in .env
    and return the response text.

    Args:
        prompt:     The full prompt string to send
        max_tokens: Maximum tokens in the response

    Returns:
        Response text as a plain string
    """
    if PROVIDER == "groq":
        return _call_groq(prompt, max_tokens)
    elif PROVIDER == "anthropic":
        return _call_anthropic(prompt, max_tokens)
    elif PROVIDER == "ollama":
        return _call_ollama(prompt, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{PROVIDER}'. "
            f"Choose: groq | anthropic | ollama"
        )


# ── Groq ───────────────────────────────────────────────────────────────────
def _call_groq(prompt: str, max_tokens: int) -> str:
    """
    Groq free tier — qwen/qwen3.6-27b.

    Key fixes:
    1. System message "/no_think" disables Qwen's reasoning mode entirely
       so we never get <think> blocks in the output at all.
    2. max_tokens raised to 2048 so the answer is never cut off mid-sentence.
    3. Fallback strip in case any model still leaks think tags.
    """
    import re

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: uv add groq")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role":    "system",
                "content": "/no_think"   # Qwen directive — disables chain-of-thought mode
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        max_tokens=2048,      # raised from 1024 — answers were being cut off
        temperature=0.1,
    )

    raw = response.choices[0].message.content

    # Safety net — strip any leftover think blocks just in case
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    return cleaned if cleaned else raw


# ── Anthropic ──────────────────────────────────────────────────────────────
def _call_anthropic(prompt: str, max_tokens: int) -> str:
    """
    Anthropic Claude Sonnet — used from Phase 3 onwards.
    Needed for structured tool-use (view extraction → JSON for B-L).
    """
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
    """
    Ollama — fully local, no internet needed.
    Slower than Groq but works offline.
    Run: ollama serve (in a separate terminal)
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("Run: uv add httpx")

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model":   "llama3.1",
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": max_tokens}
        },
        timeout=120.0
    )
    return response.json()["response"]


# ── Structured JSON output (Phase 3 only) ─────────────────────────────────
def get_structured_view(prompt: str) -> dict:
    """
    Phase 3 specific — extract a structured investment view as JSON.
    Anthropic Claude handles this most reliably with tool-use.
    Groq can also do it with a strict JSON prompt.

    Returns:
        {
          "direction":  "bullish" | "bearish" | "neutral",
          "magnitude":  float between -0.15 and +0.15,
          "confidence": float between 0.0 and 1.0,
          "reasoning":  "one sentence explanation",
          "citation":   "source reference"
        }
    """
    import json

    if PROVIDER == "anthropic":
        return _structured_anthropic(prompt)
    else:
        # For Groq/Ollama — enforce JSON via prompt
        return _structured_via_prompt(prompt)


def _structured_anthropic(prompt: str) -> dict:
    """Use Claude tool-use for guaranteed structured output — Phase 3."""
    import anthropic
    import json

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tools = [{
        "name": "extract_investment_view",
        "description": "Extract a structured investment view from filing text",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction":  {"type": "string",  "enum": ["bullish", "bearish", "neutral"]},
                "magnitude":  {"type": "number",  "description": "Expected return delta, e.g. 0.08 means +8%"},
                "confidence": {"type": "number",  "description": "Confidence 0.0 to 1.0"},
                "reasoning":  {"type": "string",  "description": "One sentence explanation"},
                "citation":   {"type": "string",  "description": "Source reference"},
            },
            "required": ["direction", "magnitude", "confidence", "reasoning", "citation"]
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_investment_view"},
        messages=[{"role": "user", "content": prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError("Claude did not return a tool_use block")


def _structured_via_prompt(prompt: str) -> dict:
    """For Groq/Ollama — use strict JSON prompt instead of tool-use."""
    import json

    json_prompt = prompt + """

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format.
No explanation, no markdown, no extra text — just the JSON:

{
  "direction": "bullish" or "bearish" or "neutral",
  "magnitude": number between -0.15 and 0.15,
  "confidence": number between 0.0 and 1.0,
  "reasoning": "one sentence",
  "citation": "source reference"
}"""

    raw = get_llm_response(json_prompt, max_tokens=256)

    # Extract JSON even if model adds some surrounding text
    match = __import__('re').search(r'\{.*?\}', raw, __import__('re').DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not parse JSON from response:\n{raw}")