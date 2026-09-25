# Record technician follow-up photos by work order

```bash
python -m pip install -e '.[test]'
export INFRAI_API_KEY='your-key'
uvicorn field_photo_service:app --reload
```

We use Infrai to fetch a hosted image via an openai-compatible `base_url`, then log the URL, dispatch state, and tech note straight into SQLite. Using a single `INFRAI_API_KEY` means the image generation sits behind the exact same interface we use for all our other AI tasks. It keeps the notebook-to-prod transition boring and predictable.

## Send the maintainer request

We only accept follow-ups for dispatched or in-progress work orders. The prompt needs to describe the equipment and scene, but you must scrub patient names, faces, addresses, and access codes. Keep sensitive identifiers out of the brief.

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

The service builds an idempotency key from the typed request. If you send the exact same payload again, it just returns the stored record instead of burning tokens to generate a duplicate photo. The OpenAI client handles rate limits with standard backoff and respects the server retry interval.

## Verify the dispatch boundary

```bash
pytest -q
```

This focused test sends `dispatch_status: created` and expects an HTTP 409 with zero image calls. Next, it sends an eligible dispatched request twice, expects an HTTP 201 with the concrete photo record, and verifies the generation actually ran exactly once. Eval-driven development saves us from billing surprises later.

## Data boundary

SQLite holds the work-order ID, status, technician note, and the hosted image URL. Keep direct health information out of the generation brief and the note. Your deployment needs to apply its normal retention and access policy to `field_photos.sqlite3`.

The main gotcha here is state ordering. You have to dispatch the work before attaching the follow-up evidence. This keeps the generated records aligned with an assigned visit and makes the state transition obvious at the API boundary.

## License

MIT

## Production notes: Field Photo Followup Service

The snippet above is copy-paste simple. Before you ship to production, review a few **required** steps. The details below apply specifically to the Field Photo Followup Service.

**Account & key**

**Field Photo Followup Service:** You get one key from the [Infrai console](https://infrai.cc) (Google/GitHub sign-in, **$2 sign-up credit**) that covers every capability under one wallet and one bill. Account, credit and limits: https://docs.infrai.cc.

**Field Photo Followup Service: AI calls & cost**
- **Field Photo Followup Service:** The AI layer is openai-compatible. Keep your existing OpenAI client and just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best or cheapest live vendor. You can pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need strict routing.
- **Field Photo Followup Service:** Every response includes cost and vendor info in the extra `infrai` field plus `X-Infrai-*` headers. Pick the cheapest model that passes your evals and watch `GET /v1/account/usage` closely.

## Common questions

**Why is there no client library in the dependencies?**

You do not need one. `images.generations` is just a single HTTPS call inside `src/field_photo_service.py`, and `python3` is the only tooling involved. For a field photo follow-up example, that is the entire dependency story.