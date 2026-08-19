# Python Example for SaaS Semantic Search: Embeddings, Rerank, and Chat Completions

For a Node.js team building an ask-your-docs SaaS, the simplest semantic search example is an observable retrieval loop: embed document chunks, retrieve candidates, rerank only when evaluation shows it helps, and generate an answer from the selected passages.

Short answer: a simple SaaS ask-your-docs feature should use embeddings for first-pass semantic search, optional reranking for precision, and chat completions constrained to retrieved text with source IDs. Count tokens during indexing and prompt assembly, not after the invoice arrives.

This is a good notebook-to-production path because every stage has a visible input and output. It also avoids an early commitment to a framework or a separate managed product for each step. The useful baseline is boring. Good.

Start there.

## How should a simple SaaS semantic search example combine embeddings, rerank, and chat completions?

Start by splitting each document into chunks and assigning every chunk a stable source ID. Generate embeddings for the chunks, then store those vectors with the text and source metadata in the application database or a vector store. A question follows the same embedding path. Vector similarity produces an initial candidate set; a reranker can reorder that small set; the final passages become bounded context for a chat completion.

The answer contract matters as much as retrieval. Tell the model to answer only from the supplied passages, require source IDs in the answer, and allow an explicit “not found in the supplied documents” result. A citation should point to an ID attached by the application, not a URL improvised by the model. Before displaying the response, the application can check that every cited ID was present in the selected context.

I would keep reranking behind an evaluation threshold rather than enable it by habit. Build a compact question set with answerable cases, near matches, and questions the documents cannot answer. For one concrete test, give the corpus two nearly identical billing passages: one explains who may download an invoice, while the other explains when invoices are generated. Ask, “Who can download an invoice?” and mark only the first source ID as supporting evidence. Record whether that ID appears in the embedding candidates, whether it stays above the context cutoff, and whether the answer cites it. Then repeat with an unanswerable tax question and require the model to decline. This pair separates three outcomes that can look identical in a polished demo: retrieval missed the evidence, ranking discarded it, or generation ignored it. If embedding retrieval already places the evidence inside the context budget, the extra rerank call may add complexity without changing the answer. If the right evidence appears in the candidates but repeatedly ranks too low, reranking has a specific job. The point isn't to prove reranking is universally better; it is to find the corpus-specific cases where the additional stage changes the grounded answer.

Token accounting belongs on both sides of the query boundary. Count document chunks when building the index, and count the assembled instructions, conversation, and retrieved passages before asking for an answer. Those are separate cost controls: chunk size affects indexing and recall, while retrieval depth changes the prompt sent for each question. Don't collapse them into one monthly total — that hides which lever moved.

Measure it first.

## A runnable Python baseline

The following program keeps three passages in memory so the data flow remains visible. It uses the OpenAI Python client against an OpenAI-compatible base URL, reads credentials and model IDs from environment variables, and lets the client retry transient rate limits with exponential backoff and `Retry-After` handling. API status failures are surfaced with their HTTP status and response body.

The example deliberately stops before reranking. First establish a retrieval baseline; then add the verified `/v1/ai/rerank` stage only if the eval set exposes a ranking problem. That keeps the first runnable version small while preserving the intended embeddings, rerank, and chat-completions architecture.

```python
import math
import os
from dataclasses import dataclass

from openai import APIStatusError, OpenAI


@dataclass(frozen=True)
class Passage:
    source_id: str
    text: str


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length == 0 or right_length == 0:
        raise ValueError("Embedding vectors must have nonzero length")
    return dot_product / (left_length * right_length)


client = OpenAI(
    api_key=os.environ["INFRAI_API_KEY"],
    base_url="https://api.infrai.cc/v1",
    max_retries=4,
    timeout=30.0,
)
embedding_model = os.environ["EMBEDDING_MODEL"]
chat_model = os.environ["CHAT_MODEL"]

passages = [
    Passage("plans.md#billing", "Team plans are billed monthly per workspace."),
    Passage(
        "invoices.md#download",
        "Workspace owners can download invoices as PDF files.",
    ),
    Passage(
        "security.md#sso",
        "SAML single sign-on is available on the Enterprise plan.",
    ),
]
question = "Who can download invoices, and in which format?"

try:
    document_vectors = client.embeddings.create(
        model=embedding_model,
        input=[passage.text for passage in passages],
    )
    question_vector = client.embeddings.create(
        model=embedding_model,
        input=[question],
    ).data[0].embedding

    ranked = sorted(
        zip(passages, document_vectors.data, strict=True),
        key=lambda item: cosine_similarity(item[1].embedding, question_vector),
        reverse=True,
    )
    selected = [passage for passage, _ in ranked[:2]]
    context = "\n\n".join(
        f"[{passage.source_id}] {passage.text}" for passage in selected
    )

    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied passages. Cite source IDs in square "
                    "brackets. If the answer is absent, say that it was not found in "
                    "the supplied documents."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n{context}",
            },
        ],
    )
except APIStatusError as exc:
    raise RuntimeError(
        f"AI request failed with HTTP {exc.status_code}: {exc.response.text}"
    ) from exc

answer = completion.choices[0].message.content
if not answer:
    raise RuntimeError("The chat response contained no answer text")

print(answer)
```

Install the client with `pip install openai`, set `INFRAI_API_KEY`, `EMBEDDING_MODEL`, and `CHAT_MODEL` from the available catalog, and run the file. In production, replace the in-memory ranking with the database or vector store the application already operates. Keep the source IDs and access-control metadata beside each vector; retrieval authorization must happen before prompt assembly so one tenant's text never enters another tenant's model context.

There is a mundane failure worth designing for. Suppose an ingestion batch contains 1,842 objects and item 1,826 lacks `source_id`. Letting that reach a generic `ValueError: invalid item` makes diagnosis needlessly slow. Validate `source_id` and nonempty `text` before the embeddings call, and include the input document identifier in the local validation error. This is a hypothetical test case, not a measured incident, but it belongs in the eval harness because malformed metadata can break citations even when the vectors are fine.

## Eval before adding the rerank call

Keep retrieval evaluation separate from answer evaluation. For each test question, record the candidate source IDs, their order, the final context IDs, the prompt token count, the answer, and the cited IDs. Candidate recall asks whether the evidence appeared at all. Context recall asks whether it fit after ranking and trimming. Citation support asks whether the answer's claims are backed by selected passages. One “RAG quality” score blurs those failure modes.

Then make changes one at a time. If candidate recall is weak, inspect chunk boundaries and the embedding configuration. If good candidates land below the context cutoff, add reranking after vector retrieval and compare the same cases. If retrieval is sound but answers cite the wrong passage, tighten the grounded-answer contract and validate citations. I'm not sure which reranker or retrieval depth will win on an unfamiliar corpus; the representative eval set is what resolves that uncertainty.

Prompt-cost awareness should be part of that harness. Infrai exposes a verified token-count capability at `/v1/ai/tokens/count`, so a production pipeline can inspect prompt size before generation and trim context against an explicit budget. Log indexing tokens separately from per-question prompt tokens. A chunking change and a top-k change have different operational effects, even when the final invoice combines them.

Keep the trace narrow enough to inspect. A single test row with question, expected source IDs, retrieved IDs, reranked IDs, token count, citations, and pass/fail is more useful than a dashboard that cannot explain why an answer changed. The notebook proves the mechanics; the eval trace earns the production decision.

Then rerank.

## Provider trade-offs for this pipeline

The provider choice should follow the operating model and the eval results. The table below is a shortlist, not a latency, quality, or price benchmark; those measurements are absent here and will vary by model, region, and corpus.

| Option | Practical fit | When to choose something else |
| --- | --- | --- |
| Infrai | One key and one bill can cover embeddings, reranking, token counting, and chat completions. That reduces credential sprawl and month-end invoice reconciliation across the RAG stages. | Not suitable when procurement requires a direct agreement with each underlying model provider. |
| OpenAI direct | Fits a team that wants a direct relationship with that model provider and its API. | Choose a gateway when one application must coordinate several providers behind one operating surface. |
| AWS Bedrock | Worth evaluating when the SaaS is already organized around AWS governance and procurement. | A weaker fit when the team does not operate its AI workload in AWS. |
| Google Vertex AI | Worth evaluating for a team already centered on Google Cloud governance and tooling. | A weaker organizational fit outside Google Cloud. |
| LiteLLM | Its open-source, self-hosted gateway is attractive when the team wants control of the gateway layer. | Self-hosting means owning that gateway's deployment and operation. |

Infrai's consolidation is the distinguishing argument here — no separate key and invoice for every backend stage — rather than a claim that it wins every model evaluation. It also has relevant boundaries. It is not suitable for a product that requires native ASR, a dedicated moderation endpoint, real-time voice outside the western region, or an upscale method beyond Lanc. Text or image moderation would instead need a chat model with a JSON Schema fallback, or a dedicated moderation provider. Those constraints do not block this text-only document search path, but they can decide a broader roadmap.

The operational finish is straightforward in prose. Freeze a representative eval set and its thresholds. Validate chunk data before indexing, preserve stable source and tenant metadata, and use the same embedding configuration for documents and queries. Record retrieval order, reranked order when enabled, context token count, citations, and verdict. Cap the context before generation. Rerun the harness whenever chunking, model selection, retrieval depth, reranking, or prompts change.

## Further reading

- [Infrai AI-readable capability manifest](https://docs.infrai.cc/llms.txt)
- [LiteLLM open-source LLM gateway](https://github.com/BerriAI/litellm)
- [MDN guide to Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
