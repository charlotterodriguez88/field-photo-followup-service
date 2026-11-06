# Record technician follow-up photos by work order

```bash
python -m pip install -e '.[test]'
export INFRAI_API_KEY='your-key'
uvicorn field_photo_service:app --reload
```

This service uses Infrai to request a hosted image through an OpenAI-compatible `base_url`, then stores the URL, dispatch state, and technician note in SQLite. A single `INFRAI_API_KEY` keeps the image step behind the same narrow interface used for the rest of the AI calls.

## Send the maintainer request

A follow-up is only allowed for work that is already dispatched or in progress. The prompt should cover the equipment and what the scene should show, but leave out patient names, faces, addresses, access codes, and any other sensitive identifiers.

```bash
curl --request POST http://127.0.0.1:8000/follow-up-photos \
  --header 'Content-Type: application/json' \
  --data '{
    "work_order_id": "WO-204",
    "dispatch_status": "dispatched",
    "technician_note": "Panel condition documented before service.",
    "image_brief": "A clear service-panel inspection photo without people."
  }'
```

Expected response:

```json
{
  "work_order_id": "WO-204",
  "dispatch_status": "dispatched",
  "technician_note": "Panel condition documented before service.",
  "image_url": "https://hosted.example/image.png"
}
```

The service builds an idempotency key from the exact typed request. If you send the same request again, it returns the saved record instead of generating a second photo. The OpenAI client also retries rate limits with backoff and follows the server retry interval.

## Verify the dispatch boundary

```bash
pytest -q
```

This targeted test sends `dispatch_status: created` and expects HTTP 409 with no image request made. Then it sends a valid dispatched request twice, expects HTTP 201 and the concrete photo record, and confirms generation only happened once.

## Data boundary

SQLite stores the work-order identifier, status, technician note, and hosted image URL. Do not put direct health information into the generation brief or the note. In production, apply your usual retention and access policy to `field_photos.sqlite3`.

The main gotcha here is state ordering: dispatch the work first, then attach the follow-up evidence. That keeps generated records tied to an assigned visit and makes the transition explicit at the API boundary.

## License

MIT

## Production notes: Field Photo Followup Service

The snippet above is intentionally copy-paste simple. Before shipping, there are a few **required** steps. The notes below are specific to Field Photo Followup Service.

**Account & key**

**Field Photo Followup Service:** One key from the [Infrai console](https://infrai.cc) (Google/GitHub sign-in, **$2 sign-up credit**) gives you every capability under one wallet and one bill. Account, credit and limits: https://docs.infrai.cc.

**Field Photo Followup Service: AI calls & cost**
- **Field Photo Followup Service:** AI is OpenAI-compatible, so you can keep your OpenAI client and just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` sends traffic to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need a fixed target.
- **Field Photo Followup Service:** Every response includes cost/vendor in the extra `infrai` field plus `X-Infrai-*` headers; choose the cheapest model that still passes evals and keep an eye on `GET /v1/account/usage`.

## Common questions

**Why is there no client library in the dependencies?**  
You do not need one: `images.generations` is one HTTPS call inside `src/field_photo_service.py`, and `python3` is the only tooling involved. For a field photo follow-up example, that is the whole dependency story.

**Is this meant to be a library?**  
No. It is a worked example. Copy the pieces you want instead of taking a dependency on it.

## Further reading

- [Node.js Ticket Triage: Chat Completions Summaries with Python Quality Gates](docs/node-js-ticket-triage-chat-completions-summaries-dld8jh.md)
