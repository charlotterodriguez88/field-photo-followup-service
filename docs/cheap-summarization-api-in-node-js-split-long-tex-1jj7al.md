# Cheap Summarization API in Node.js: Split Long Text for Schema-First Moderation

**Short answer:** classify moderation reports with a schema-first, budget-gated pipeline, then summarize only the reports that pass validation. For a developer tool that sends reports to human reviewers, structured output correctness matters more than finding a nominally cheap summarization API. Split long text by measured tokens, estimate the request before dispatch, and make invalid output a normal review state rather than an exception that disappears in logs.

This is a different problem from making a nice paragraph. A reviewer needs a stable label, a reason, and enough evidence to act. A fluent summary with a missing `severity` field is operationally worse than a plain response that gets held for inspection.

## What should a long-text SaaS API validate before summarization?

Start with the contract. For each moderation report, define a small result such as `category`, `severity`, `evidence`, and `needs_human_review`. The values should be closed sets wherever the product can tolerate them. Free-form prose belongs in a separate field; it should never be the thing that decides the queue.

The admission layer has four jobs: normalize the source, count the model-facing input, split it within a configured budget, and estimate the maximum request cost. It should reject or defer a document before the first model call when the estimate exceeds the tenant's allowance. That is a product decision, not a retry policy.

The cheap path is tempting: send the entire report, ask for JSON, parse it, and hope the model's context limit is generous. It fails in several boring ways. A pasted transcript can hide its true size behind a small character count. A chunk can end between a claim and its evidence. A valid JSON object can still contain an unknown label. Each failure has a different repair, so one generic `except` block is not enough. Consider a moderation report assembled from a user complaint, a quoted conversation, and an automated detector's note. The complaint says the content is harassment; the quote contains the same words as an example; the detector says the confidence is low. A naive summarizer may compress all three into one confident allegation, and a naive parser may accept it because the JSON is syntactically valid. A schema-first classifier keeps the source roles separate, carries the chunk identifiers into the evidence field, and can set `needs_human_review` when the policy signal conflicts with the surrounding context. That decision is valuable even if the prose summary is discarded. It also gives the evaluation harness something precise to score: category, severity, evidence links, and the review flag, each independently.

The contract is the gate.

## How do long text, token count, chunking, and cost shape a SaaS API?

Use token count for admission and characters for display. Tokenization varies by model and by text, so a character-based threshold is only a rough warning. Count the exact text plus the instruction template, reserve output space, and keep a margin for a merge pass. Record the estimate with the job; this makes a later dispute about a quota explainable.

Chunk on paragraph boundaries first. If one paragraph is oversized, split it into sentences while retaining the report identifier and a small amount of section context. Do not blindly overlap every chunk: overlap raises cost and can duplicate evidence in the final result. Instead, preserve stable metadata and ask the merge step to cite chunk identifiers.

Here is the planning core from an evaluation harness. The production service can be Node.js, but this Python-only example keeps the policy easy to test in a notebook before it is mirrored behind a generic HTTP adapter. The tokenizer is injected because the harness should test the same counting behavior used by the selected runtime, not a convenient approximation.

```python
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Chunk:
    report_id: str
    text: str
    tokens: int


def plan_chunks(
    report_id: str,
    paragraphs: list[str],
    input_limit: int,
    output_reserve: int,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    budget = input_limit - output_reserve
    if budget <= 0:
        raise ValueError("input_limit must leave room for output")

    chunks: list[Chunk] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join([*current, paragraph])
        if current and count_tokens(candidate) > budget:
            text = "\n\n".join(current)
            chunks.append(Chunk(report_id, text, count_tokens(text)))
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        text = "\n\n".join(current)
        chunks.append(Chunk(report_id, text, count_tokens(text)))

    oversized = [chunk for chunk in chunks if chunk.tokens > budget]
    if oversized:
        raise ValueError("split an oversized paragraph by sentence")
    return chunks
```

The last exception is deliberate. Silently sending an oversized paragraph defeats the admission rule. A sentence splitter, or a human-review route for pathological input, is safer than pretending the policy completed.

## Which output checks belong between the model and the reviewer?

Parse JSON strictly, validate the schema, and validate the meaning of each field. Schema validation catches a missing key; a domain validator catches `severity: "urgent-ish"`. Also check that every evidence reference points to a chunk that was actually supplied. This turns a plausible hallucination into a visible rejected result.

For chunked reports, each partial result should carry `report_id`, `chunk_id`, and a short evidence list. The merge request should accept only those partial objects and produce the final schema. If a partial is invalid, retry that chunk with a narrower prompt once, then mark the report for human review. Do not merge whatever fields happened to parse.

A useful state machine is small: `received`, `admitted`, `chunked`, `classified`, `validated`, `needs_review`, and `failed_permanently`. Persist the transition and the validation reason. “Failed” without a reason is not an operational state; it is missing information.

## What does an eval-driven moderation workflow measure?

Build the evaluation set from representative reports, including long transcripts, empty evidence, mixed languages, quoted abuse, and labels that sit near a policy boundary. The primary score is exact schema validity, followed by label accuracy and evidence precision. A summary-quality score cannot compensate for a malformed routing decision.

I would run the harness before changing the chunk size, output cap, or prompt. Compare whole-document and chunked variants on the same fixtures, then inspect disagreements rather than only averaging them. Track input tokens, output tokens, validation retries, review deferrals, and latency. Three words can expose a costly regression: `retry rate increased`.

I'm not sure a single threshold will fit every moderation corpus, and your mileage may vary with OCR-heavy reports or copied tables. That uncertainty belongs in configuration and test data, not in an unbounded production call.

## What are the trade-offs of this admission design?

| Choice | Helps with | Cost or limitation |
| --- | --- | --- |
| Whole-document classification | Preserves global context | Hits size limits and makes spend harder to bound |
| Paragraph-first chunking | Keeps evidence readable | Can separate a claim from its context |
| Sentence fallback | Handles oversized paragraphs | Adds splitting complexity and may lose formatting |
| Strict schema rejection | Keeps the review queue predictable | Sends more edge cases to people |
| Automatic retry | Recovers from transient model variation | Adds tokens and can hide a weak prompt |
| Human-review fallback | Preserves safety for unknown cases | Increases queue volume and response time |

The catch is that this design is not suitable when the task needs a single uninterrupted interpretation of a short document, or when a human must see every original formatting detail exactly as authored. In those cases, stick with a whole-document path and a smaller, explicit input limit. Choose the chunked path when bounded admission, auditable evidence, and predictable routing matter more than perfect global context.

Before copying the pattern, measure the false-review rate on your own policy set, the percentage of reports needing sentence fallback, and the token estimate's error against actual usage. Those numbers tell you whether the extra control is earning its operational weight.

## References

- https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- https://github.com/BerriAI/litellm
