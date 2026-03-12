"""Six-dimensional verification scoring schema for OmniVerifier.

This module defines the structured error classification system used by
OmniVerifier to evaluate image-prompt alignment across six dimensions:

  1. object         — Are the correct objects present? No extras, no missing.
  2. count          — Are the quantities correct for each mentioned object?
  3. attribute      — Are colors, sizes, materials, textures correct?
  4. spatial_action — Are positions, poses, actions, interactions correct?
  5. semantic       — Does the image convey the intended meaning, concept,
                      metaphor, idiom, scientific principle, or reasoning?
  6. text_content   — Does the image contain the required text, labels,
                      numbers, dates, names, or informational content?

Scores are semi-discrete: {0.0, 0.25, 0.5, 0.75, 1.0} for stability.

Output JSON schema:
  {
    "answer": bool,
    "scores": {"object": float, "count": float,
               "attribute": float, "spatial_action": float,
               "semantic": float, "text_content": float},
    "primary_issue": str,   # one of the six keys, or "none"
    "explanation": str,
    "edit_prompt": str
  }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Score constants
# ---------------------------------------------------------------------------

VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

SCORE_DIMENSIONS = ("object", "count", "attribute", "spatial_action", "semantic", "text_content")

VALID_PRIMARY_ISSUES = (*SCORE_DIMENSIONS, "none")

# When the primary_issue is X, we allow piggybacking 1-2 related sub-issues
RELATED_ISSUES: dict[str, list[str]] = {
    "object":         ["attribute", "count"],
    "count":          ["object", "spatial_action"],
    "attribute":      ["object"],
    "spatial_action": ["count", "object"],
    "semantic":       ["attribute", "text_content"],
    "text_content":   ["semantic", "object"],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionScores:
    """Six-dimensional alignment scores."""
    object: float = 1.0
    count: float = 1.0
    attribute: float = 1.0
    spatial_action: float = 1.0
    semantic: float = 1.0
    text_content: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "object": self.object,
            "count": self.count,
            "attribute": self.attribute,
            "spatial_action": self.spatial_action,
            "semantic": self.semantic,
            "text_content": self.text_content,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DimensionScores":
        return cls(
            object=float(d.get("object", 1.0)),
            count=float(d.get("count", 1.0)),
            attribute=float(d.get("attribute", 1.0)),
            spatial_action=float(d.get("spatial_action", 1.0)),
            semantic=float(d.get("semantic", 1.0)),
            text_content=float(d.get("text_content", 1.0)),
        )

    def min_score(self) -> tuple[str, float]:
        """Return (dimension_name, score) of the lowest-scoring dimension."""
        scores = self.to_dict()
        worst = min(scores, key=scores.get)  # type: ignore[arg-type]
        return worst, scores[worst]

    def all_perfect(self) -> bool:
        return all(v == 1.0 for v in self.to_dict().values())

    def snap_to_grid(self) -> "DimensionScores":
        """Snap each score to the nearest valid discrete value."""
        def _snap(v: float) -> float:
            return min(VALID_SCORES, key=lambda s: abs(s - v))
        return DimensionScores(
            object=_snap(self.object),
            count=_snap(self.count),
            attribute=_snap(self.attribute),
            spatial_action=_snap(self.spatial_action),
            semantic=_snap(self.semantic),
            text_content=_snap(self.text_content),
        )


@dataclass
class ScoredVerificationResult:
    """Full verification output with six-dimensional scores."""
    answer: bool
    scores: DimensionScores
    primary_issue: str
    explanation: str
    edit_prompt: str
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "scores": self.scores.to_dict(),
            "primary_issue": self.primary_issue,
            "explanation": self.explanation,
            "edit_prompt": self.edit_prompt,
        }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SCORING_PROMPT_TEMPLATE = """\
This image was generated from the prompt: {prompt}

Carefully analyze the image and evaluate how well it matches the prompt \
across six dimensions. For each dimension, assign a score from exactly \
one of: 0.0, 0.25, 0.5, 0.75, 1.0 (where 1.0 = perfect, 0.0 = severe error).

=== Dimension Definitions ===

1. object (Are the correct objects present?)
   - Are all required objects present?
   - Are there any unwanted extra objects?
   - Are the object categories/types correct?
   Scoring:
     1.0  = All objects correct
     0.75 = Minor object issue (e.g. slightly wrong variant)
     0.5  = Noticeable object missing or extra, but not critical
     0.25 = Important object missing or wrong category
     0.0  = Key object completely missing or severely wrong

2. count (Are the quantities correct?)
   - Does each object appear the correct number of times as stated in the prompt?
   Scoring:
     1.0  = All counts exactly correct
     0.75 = Off by one on a minor object
     0.5  = Off by one on a key object
     0.25 = Count significantly wrong
     0.0  = Count completely wrong (e.g. 1 instead of 5)

3. attribute (Are the visual properties correct?)
   - Color, size, material, texture, shape, appearance
   Scoring:
     1.0  = All attributes correct
     0.75 = One minor attribute slightly off
     0.5  = One or more attributes noticeably wrong
     0.25 = Key attribute clearly wrong
     0.0  = Multiple critical attributes wrong

4. spatial_action (Are positions, poses, and interactions correct?)
   - Spatial relations: on/under, left/right, in front/behind, inside/outside
   - Actions and poses: sitting, standing, holding, riding, flying, etc.
   - Interactions between objects
   Scoring:
     1.0  = All spatial relations and actions correct
     0.75 = Minor positional imprecision
     0.5  = One relation or action noticeably wrong
     0.25 = Key spatial relation or action clearly wrong
     0.0  = Critical spatial/action relationship completely wrong

5. semantic (Does the image convey the intended meaning or concept?)
   - Abstract concepts, metaphors, idioms, scientific principles
   - Logical or causal reasoning implied by the prompt
   - Scene-level meaning and narrative intent
   - Cultural references, analogies, symbolic meaning
   Scoring:
     1.0  = Semantic meaning fully conveyed
     0.75 = Meaning mostly correct with minor nuance lost
     0.5  = Core concept present but partially misinterpreted
     0.25 = Significant semantic mismatch or wrong interpretation
     0.0  = Meaning completely wrong or absent

6. text_content (Does the image contain the required text or informational content?)
   - Text, labels, numbers, dates, names visible in the image
   - Signs, captions, watermarks, or written content specified in the prompt
   - Informational accuracy of any rendered text
   Scoring:
     1.0  = All required text/information present and correct
     0.75 = Text present with minor typos or formatting issues
     0.5  = Some required text missing or partially wrong
     0.25 = Key text missing or significantly wrong
     0.0  = Required text completely absent or unreadable

=== Output Rules ===

- If the image fully matches the prompt, set "answer" to true and all scores to 1.0.
- If any score is below 1.0, set "answer" to false.
- "primary_issue" must be the dimension with the LOWEST score. \
If tied, pick the one most visually impactful.
- "explanation" should briefly describe the errors found.
- "edit_prompt" must provide a concrete, actionable editing instruction \
focused on fixing the primary_issue. You may also address 1-2 closely \
related sub-issues, but do NOT try to fix everything at once.
  - The instruction must specify the exact action (add / remove / replace / move / recolor / rewrite).
  - The instruction must specify the location or reference point \
(e.g. "move the cat from on top of the table to under the table").
  - Do NOT give vague instructions like "fix the count" or "ensure correctness".

Each score MUST be one of: 0.0, 0.25, 0.5, 0.75, 1.0

=== Consistency Constraints (MANDATORY) ===

You MUST obey every rule below. Violating any one makes your output invalid.

1. If primary_issue is "none" AND all six scores are 1.0, \
then "answer" MUST be true.
2. If "answer" is false, you MUST:
   a. Set "primary_issue" to a concrete category (one of the six \
dimensions above — NEVER "none").
   b. List at least one specific, image-grounded discrepancy in \
"explanation" (cite what you see vs. what the prompt requires).
   c. Set at least one score strictly below 1.0.
3. If "answer" is true, then ALL six scores MUST be 1.0, \
"primary_issue" MUST be "none", and "edit_prompt" MUST be empty.
4. Do NOT output answer=false when you cannot identify any concrete \
error. If every dimension looks correct, answer MUST be true.

Respond strictly in this JSON format (no extra text outside the JSON):

{{
  "answer": true/false,
  "scores": {{
    "object": <score>,
    "count": <score>,
    "attribute": <score>,
    "spatial_action": <score>,
    "semantic": <score>,
    "text_content": <score>
  }},
  "primary_issue": "<dimension_name or none>",
  "explanation": "<brief description of errors>",
  "edit_prompt": "<concrete editing instruction, or empty string if answer is true>"
}}"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _try_extract_scored_json(text: str) -> dict | None:
    """Try multiple strategies to extract the scored JSON from model output."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    # 3. Greedy brace matching for nested JSON (scores is nested)
    match = re.search(r"\{.*\"scores\".*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        # Find the balanced closing brace
        depth, end = 0, 0
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                return json.loads(candidate[:end])
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def parse_scored_output(raw_output: str) -> ScoredVerificationResult:
    """Parse model output into a ScoredVerificationResult.

    Handles <think>...</think> tags and various JSON formats.
    Falls back to a safe default (all zeros, answer=False) on failure.
    """
    # Build candidate texts: after </think>, before <think>, full output
    candidates: list[str] = []
    if "</think>" in raw_output:
        candidates.append(raw_output.split("</think>", 1)[1].strip())
    if "<think>" in raw_output:
        candidates.append(raw_output.split("<think>", 1)[0].strip())
    candidates.append(raw_output)

    parsed: dict | None = None
    for text in candidates:
        if not text:
            continue
        parsed = _try_extract_scored_json(text)
        if parsed is not None:
            break

    if parsed is None:
        return ScoredVerificationResult(
            answer=False,
            scores=DimensionScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            primary_issue="object",
            explanation="Failed to parse verifier output.",
            edit_prompt="remain unchanged",
            raw_output=raw_output,
        )

    # --- answer ---
    answer = parsed.get("answer", False)
    if isinstance(answer, str):
        answer = answer.lower().strip() == "true"
    answer = bool(answer)

    # --- scores ---
    raw_scores = parsed.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores = DimensionScores.from_dict(raw_scores).snap_to_grid()

    # Consistency: scores and answer must agree
    if scores.all_perfect() and answer is False:
        # Model says "false" but can't articulate which dimension is wrong.
        # Trust the model's holistic judgment — keep answer=False and
        # attribute the issue to "semantic" (the hardest dimension to score).
        scores = DimensionScores(
            object=1.0, count=1.0, attribute=1.0,
            spatial_action=1.0, semantic=0.75, text_content=1.0,
        )
    elif answer is True and not scores.all_perfect():
        answer = False  # scores say there's an issue

    # --- primary_issue ---
    primary_issue = parsed.get("primary_issue", "")
    if isinstance(primary_issue, str):
        primary_issue = primary_issue.strip().lower()
    if primary_issue not in VALID_PRIMARY_ISSUES:
        # Auto-detect from lowest score
        primary_issue, _ = scores.min_score()
    if answer is True:
        primary_issue = "none"

    # --- explanation & edit_prompt ---
    explanation = str(parsed.get("explanation", ""))
    edit_prompt = str(parsed.get("edit_prompt", ""))
    if answer is True:
        explanation = explanation or "The image fully matches the prompt."
        edit_prompt = ""
    else:
        edit_prompt = edit_prompt or "remain unchanged"

    return ScoredVerificationResult(
        answer=answer,
        scores=scores,
        primary_issue=primary_issue,
        explanation=explanation,
        edit_prompt=edit_prompt,
        raw_output=raw_output,
    )


def build_scored_verification_question(prompt: str) -> str:
    """Build the verification question with six-dimensional scoring."""
    return SCORING_PROMPT_TEMPLATE.format(prompt=prompt)


# ---------------------------------------------------------------------------
# edit_prompt priority logic (for programmatic post-processing)
# ---------------------------------------------------------------------------

def select_primary_and_related(
    scores: DimensionScores,
) -> tuple[str, list[str]]:
    """Select the primary issue and up to 2 related sub-issues.

    Returns:
        (primary_dimension, [related_dimensions])
    """
    score_dict = scores.to_dict()
    if scores.all_perfect():
        return "none", []

    primary = min(score_dict, key=score_dict.get)  # type: ignore[arg-type]
    related_candidates = RELATED_ISSUES.get(primary, [])

    related = [
        dim for dim in related_candidates
        if score_dict[dim] < 1.0
    ]
    return primary, related[:2]
