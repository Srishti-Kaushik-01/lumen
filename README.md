# Lumen Research Assistant

Lumen combines context-aware scholarly search, answers written by an open-weight language model on Groq (Gemini as backup) with numbered citations, quoted key evidence, and private conversation history. It runs a quantized Hugging Face MiniLM transformer on CPU and uses FAISS to rank results. The bundled corpus provides an offline path once model assets are cached.

## Run locally

Use Python 3.11 and create a virtual environment outside the deployment bundle. From `code/`:

```powershell
python -m pip install -r requirements.txt
python model.py
python app.py
```

For local use, variables can be placed in `code/.env` as `NAME=value` lines (the file is git-ignored), for example `GEMINI_API_KEY=...`.

Open `http://127.0.0.1:5000`, create an account, and try “How does retrieval-augmented generation work?” followed by “What are its limitations?”

| Variable                                       | Purpose                                                                                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`                               | PostgreSQL URL. Local development defaults to `~/.ignou-research.sqlite3`.                                                       |
| `SECRET_KEY`                                 | Random secret shared by all workers. Set a stable value for sessions to survive a restart.                                         |
| `APP_ENV`                                    | Set to `production` to require PostgreSQL and a secret and to enable secure cookies.                                             |
| `MODEL_CACHE_DIR`                            | Writable model cache. Defaults to `~/.cache/ignou-minilm`.                                                                       |
| `OPENALEX_API_KEY`                           | Free key from openalex.org (strongly recommended). Without it, searches share a small daily budget per IP address.                                                                           |
| `LOCAL_ONLY`                                 | Set to `1` to use the bundled corpus only.                                                                                       |
| `GROQ_API_KEY` | Free key from console.groq.com/keys. Answers are written by `openai/gpt-oss-120b` (override with `GROQ_MODEL`). |
| `GEMINI_API_KEY` | Free key from Google AI Studio (aistudio.google.com/apikey). Backup answer writer used when Groq is unavailable. Without either key, answers are quoted evidence only. |
| `GEMINI_MODEL`                               | Optional. Defaults to `gemini-flash-latest`.                                                                                     |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional. Enables “Continue with Google”. Without them only email sign-in is shown.                                              |

### Google sign-in

In the [Google Cloud console](https://console.cloud.google.com/apis/credentials), create an OAuth client ID of type **Web application** (configure the consent screen first if asked; the app only requests the `openid email` scopes). Add these authorised redirect URIs:

- `http://127.0.0.1:5000/auth/google/callback` for local use
- `https://<your-host>/auth/google/callback` for the deployed app

Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the environment and restart. The app uses the authorization code flow with PKCE, state, and nonce, and checks the ID token's issuer, audience, nonce, expiry, and verified email. A Google account whose verified email matches an existing email account signs into that account.

`python model.py` downloads the tokenizer and quantized ONNX graph from a pinned Hugging Face revision and initializes the runtime. The app also starts loading them in the background when it starts, so the first visitor does not wait for them. Windows machines with a policy blocking the Psycopg binary extension can use the driver’s pure-Python implementation with a locally installed PostgreSQL client library on `PATH` and `PSYCOPG_IMPL=python`.

## Deploy

Use `code/` as the Render service root, select Python 3.11, and provision a managed PostgreSQL database. Set `DATABASE_URL`, `SECRET_KEY`, `APP_ENV=production`, `GROQ_API_KEY` and `OPENALEX_API_KEY`, and optionally `GEMINI_API_KEY`, `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`, in the host dashboard. Use a private or filtered source repository so the academic documents and screenshots stay outside the deployment.

- Build command — `pip install -r requirements.txt && python model.py`
- Start command — `gunicorn --workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --access-logfile - --error-logfile - app:app` (keep one worker: sign-in and question limits are counted in memory)
- Health check — `/healthz`

Generate a deployment secret with `python -c "import secrets; print(secrets.token_hex(32))"`. For Heroku, put the contents of `code/` at the deployment repository root and use its included `Procfile`. Confirm persistence and live provider access on the chosen host. Free hosting may sleep or limit database retention.

## Verify

Install the requirements into the test environment first (`python -m pip install -r code/requirements.txt`). From the project root, run `python tests/evaluate.py`. `code/.env` is read on import, so point the tests at a throwaway database, for example `DATABASE_URL=test.sqlite3 python tests/evaluate.py`. For the pretrained model and live provider, use `--real --live`. For screenshots, install Playwright in the test environment and pass `--browser` with an installed Chromium browser or `--browser-executable`. Set `TEST_DATABASE_URL` to a **disposable** PostgreSQL database to exercise the hosted persistence path. See `python tests/evaluate.py --help` for evidence output.

## Evidence and privacy

The application uses the current question or the recovered topic to form a short keyword search for OpenAlex. Local searches stay on the application server. The writing service (Groq, or Gemini as a backup) receives only the question, the two most recent turns, and the selected paper summaries — never account details. Citation numbers that do not match a supplied paper are removed, each cited sentence is compared with the paper it cites and the citation is removed if they do not match, and quoted key evidence is shown under each answer. When no paper matches a question, no answer is written. The app stores conversations until the owner deletes them. It uses the signed-in browser session to scope saved data. Treat provider coverage and extractive evidence as aids to research, and check the cited paper for full methodology and context.
