---
title: Batches
section: guides
order: 17
---

# Batches

OpenAI's Batch API, answered by your own GPU. Give LlamaForge a `.jsonl` file of chat requests and it works through them one by one into a results file: now, or tonight once nobody is using the router. Scripts written for OpenAI's batch endpoints work against it by changing `base_url`.

## What it does

- **Same formats as OpenAI.** `/v1/files` and `/v1/batches` take and return the same objects, so the official `openai` SDK (or anything else that speaks the API) uploads, starts, polls, cancels and downloads without changes.
- **Tonight, if you like.** A batch started with `when: "idle"` (the card's **Run a file tonight** button) waits until the router has had no requests for 20 minutes, then runs to the end.
- **Steps aside.** While checked tasks, a speed comparison, a PR build or Auto-Forge is running, the batch pauses between requests and carries on afterwards.
- **Survives a restart.** Each answer is written before the next request goes out. If the panel stops, the batch resumes where it was; at most the request in flight is asked again.
- **Fails one line, not the file.** A request the model rejects goes to the errors file with the router's own reply; the rest carry on.

## Where

**Stats → Batches**. The card lists every batch with a progress bar, what it is waiting for, **Cancel** while it runs, and **Results** / **Errors** downloads and **Remove** when it is done. **Use it from a script** shows a Python snippet with this panel's address filled in.

## The input file

One request per line, the OpenAI batch format. Every line must name the same model, which LlamaForge loads before the first request:

```json
{"custom_id": "ticket-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-8b", "messages": [{"role": "user", "content": "Summarise: ..."}], "max_tokens": 200}}
{"custom_id": "ticket-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen3-8b", "messages": [{"role": "user", "content": "Summarise: ..."}], "max_tokens": 200}}
```

A bad line is refused when the batch is created, with its line number: not JSON, a repeated `custom_id`, another URL or method, or a different model. Only `/v1/chat/completions` is supported, and the completion window is `24h`; a batch still running after 24 hours ends as `expired`, and what it answered so far stays in its results.

## From Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="none")   # or your router key

f = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                          completion_window="24h")
# later
b = client.batches.retrieve(b.id)
if b.status == "completed":
    print(client.files.content(b.output_file_id).text)
```

Each results line is `{"id", "custom_id", "response": {"status_code", "body"}, "error"}`, where `body` is the chat completion exactly as the router returned it. Here results come back in input order, but OpenAI doesn't promise that, so match on `custom_id` and the same code works on both.

To run it tonight from a script, add `extra_body={"when": "idle"}` to `batches.create`.

## Limits and privacy

- Up to 64 MB per file, 200 files, 50,000 requests per batch. One batch runs at a time, oldest first; the router's own parallel slots are what make it fast.
- Files and results stay in `batches/` next to `config.json`. Nothing leaves the machine.
- When the router listens beyond this machine with a key, `/v1/files` and `/v1/batches` need the same key as `/v1/chat/completions`. The card uses the panel's own routes and needs none.
- Uploads from scripts are the one form-encoded POST the panel accepts, and only on `/v1/files` without a browser `Origin`, so a web page can't use it to drop files on you.

## See also

- [API reference](api.md#batches): every route.
- [Usage Stats](stats.md): the rest of the Stats tab.
