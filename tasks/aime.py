# tasks/aime.py
"""
AIME evaluation task.

Dataset format: JSONL at data/aime2024.jsonl
Each line example (as you showed):
{
  "id": 60,
  "problem": "...",
  "solution": "...",
  "answer": "204",
  "url": "..."
}

Evaluation:
- Generative (model must produce an integer answer).
- We ask the model to output the final answer as \\boxed{NNN}.
- We extract predicted answer with a robust extractor:
  1) \\boxed{...}
  2) \\framebox{...}
  3) #### ...
  4) last integer in the response
- Compare as normalized integer (strip spaces, allow leading zeros).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from tasks.common import Task


# -------------------------
# Answer extraction helpers
# -------------------------

_BOX_LIKE_RE = re.compile(r"(boxed|framebox)\s*\{", re.IGNORECASE)
_HASH_RE = re.compile(r"####\s*([\-]?\d+)")
_LAST_INT_RE = re.compile(r"([\-]?\d+)(?!.*[\-]?\d+)")  # last integer in string


def _extract_braced_content_after_keyword(s: str, keyword: str) -> str:
    """
    Extract {...} content after occurrences like 'boxed{...}' or 'framebox{...}'.
    Uses brace matching to support nested braces.
    Takes the *last* occurrence to mimic many "final answer" conventions.
    Returns "" if not found or malformed.
    """
    idx = s.lower().rfind(keyword.lower())
    if idx == -1:
        return ""
    # find first '{' after keyword
    j = s.find("{", idx)
    if j == -1:
        return ""
    stack = 0
    out = []
    for c in s[j:]:
        if c == "{":
            stack += 1
            if stack == 1:
                continue
        elif c == "}":
            stack -= 1
            if stack == 0:
                break
        if stack >= 1:
            out.append(c)
    return "".join(out).strip()


def extract_aime_answer(model_text: str) -> Optional[str]:
    """
    Return a normalized integer string if found, else None.
    """
    if not isinstance(model_text, str):
        return None

    text = model_text.strip()

    # 1) \boxed{...}
    if "boxed" in text:
        inner = _extract_braced_content_after_keyword(text, "boxed")
        if inner:
            # inner might be "113" or "\\textbf{(113)}" etc; pull integer from it
            m = re.search(r"[-]?\d+", inner)
            if m:
                return m.group(0)

    # 2) \framebox{...}
    if "framebox" in text:
        inner = _extract_braced_content_after_keyword(text, "framebox")
        if inner:
            m = re.search(r"[-]?\d+", inner)
            if m:
                return m.group(0)

    # 3) GSM8K-style ####
    m = _HASH_RE.search(text)
    if m:
        return m.group(1)

    # 4) last integer in the response
    m = _LAST_INT_RE.search(text)
    if m:
        return m.group(1)

    return None


def normalize_int_string(x: Optional[str]) -> Optional[str]:
    """
    Normalize an integer string:
    - strip spaces
    - allow leading zeros (007 -> 7)
    - keep sign if any
    Returns None if cannot parse as int.
    """
    if x is None:
        return None
    x = str(x).strip()
    m = re.fullmatch(r"\s*([-]?\d+)\s*", x)
    if not m:
        return None
    try:
        return str(int(m.group(1)))  # int() removes leading zeros
    except Exception:
        return None


# ------------
# Task class
# ------------

class AIME(Task):
    def __init__(self, data_path: str = "data/aime2024.jsonl", **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"AIME data file not found: {data_path}")

        self.examples: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        # light validation
        for ex in self.examples[:3]:
            if "problem" not in ex or "answer" not in ex:
                raise ValueError(f"Bad AIME example keys: {ex.keys()}")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        ex = self.examples[index]
        problem = str(ex["problem"]).strip()
        answer = str(ex["answer"]).strip()

        user_prompt = (
            f"{problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        conversation = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": ""},  # IMPORTANT: required by tokenizer
            ],
            "answer": answer,  # gold, not in messages
            "id": ex.get("id", None),
            "url": ex.get("url", None),
        }
        return conversation

    def evaluate(self, conversation, assistant_response: str):
        # gold
        gold_raw = conversation.get("answer", None)
        gold = normalize_int_string(gold_raw)

        # pred
        pred_raw = extract_aime_answer(assistant_response)
        pred = normalize_int_string(pred_raw)

        # if cannot parse prediction, it's wrong
        if gold is None or pred is None:
            return 0
        return int(pred == gold)

    def reward(self, conversation, assistant_response: str):
        return float(self.evaluate(conversation, assistant_response))
