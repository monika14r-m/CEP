# ML-based Intrusion Detection System — end-to-end pipeline

This project turns the notebook cells into a runnable pipeline, a
deployable inference service, and a live comparison dashboard, mapped onto
the Task 10 architecture:

```
Network Traffic → Data Collection → Preprocessing → Feature Engineering →
ML Model → Risk Score → Decision Threshold → Alert Classification →
Security Analyst → Feedback → Model Retraining
```

| Architecture stage | Where it lives |
|---|---|
| Data Collection | `train_pipeline.py::load_data()` (NSL-KDD here; swap for a live feed in production) |
| Preprocessing + Feature Engineering | `train_pipeline.py::preprocess()` |
| ML Model | `train_pipeline.py` — Decision Tree, Random Forest, Random Forest+SMOTE, compared by PR-AUC |
| Risk Score | `model.predict_proba()`, exposed via `app.py`'s `risk_score` field |
| Decision Threshold | `deploy_meta.json::default_threshold`, overridable per-request in `app.py` |
| Alert Classification | `app.py::/predict`, `/predict_all`, `/predict_batch` |
| Security Analyst | `docs/index.html` — the live multi-model risk console |
| Feedback / Retraining | Not automated here — see "Closing the loop" below |

## Files

- `train_pipeline.py` — loads NSL-KDD, cleans it, handles class imbalance,
  trains and compares Decision Tree / Random Forest / Random Forest+SMOTE,
  evaluates them (confusion matrix, precision, recall, F1, FPR, FNR, PR-AUC),
  runs the threshold sweep, the System A/B scenario, and the zero-day
  breakdown. Saves **every** trained model (not just the best) into
  `artifacts/`, so the API can serve real multi-model comparison.
- `app.py` — FastAPI service that loads `artifacts/` and scores new
  connection records, either with the single best model (`/predict`) or
  every trained model at once (`/predict_all`).
- `docs/index.html` — a self-contained, no-build dashboard: pick a sample
  connection (or wire it to real traffic), see every model's risk score
  side-by-side, and see at a glance where they agree or disagree. Works
  standalone in demo mode, or live against a deployed `app.py`.
- `sample_record.json` — example payload for testing the API.
- `requirements.txt`, `Dockerfile` — local setup and containerized deployment.
- `render.yaml` — one-file config to deploy the API on Render.
- `.github/workflows/docker-build.yml` — CI: builds the image on every push,
  smoke-tests it on pull requests, publishes to GHCR on `main`.

## Running locally

```bash
pip install -r requirements.txt
python train_pipeline.py          # trains models, writes artifacts/ and PNG/CSV reports
uvicorn app:app --reload --port 8000
```

Open `docs/index.html` directly in a browser (demo mode), or paste
`http://localhost:8000` into its "Backend API URL" field to drive it from
your local API.

Test the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @sample_record.json

curl -X POST http://localhost:8000/predict_all \
  -H "Content-Type: application/json" -d @sample_record.json

curl http://localhost:8000/models
```

Adjust the threshold per-request (e.g. to favor recall over fewer alerts):

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.3" \
  -H "Content-Type: application/json" -d @sample_record.json
```

## Can this run on GitHub? Yes — with one caveat

GitHub itself doesn't run a live Python process, so the pieces split:

| Piece | Where it can live |
|---|---|
| Code, Dockerfile, CI | GitHub repo — works as-is |
| `docs/index.html` dashboard | **GitHub Pages**, served directly, for free |
| `app.py` FastAPI service | A container host — GitHub can *build* the image (Actions) but something else has to *run* it 24/7 |

### 1. Push the repo

```bash
git init
git add .
git commit -m "ML-IDS: pipeline, API, and comparison dashboard"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### 2. Put the dashboard on GitHub Pages (free, static)

Repo → **Settings → Pages** → Source: `Deploy from a branch` → Branch:
`main`, folder `/docs` → Save. Your dashboard is now live at
`https://<you>.github.io/<repo>/` in demo mode immediately — no backend
required to look at and click through.

### 3. Deploy the API somewhere that runs containers

GitHub Actions (`.github/workflows/docker-build.yml`) already builds and
pushes the image to `ghcr.io/<you>/<repo>` on every push to `main`. Point any
of these at that image, or build straight from the repo:

- **Render** — the included `render.yaml` makes this one click: New →
  Blueprint → point at your repo. Free tier available.
- **Railway / Fly.io** — `railway up` or `fly launch`, both auto-detect the
  Dockerfile.
- **Google Cloud Run / AWS ECS / Azure Container Apps** — push the GHCR
  image, point the service at port 8000. These scale, handle TLS, and
  restart the container for you — the lowest-effort path to a "real"
  production deployment.

Once deployed, paste that service's URL into the dashboard's "Backend API
URL" field — it calls `/models` and `/predict_all` and switches from Demo
mode to Live automatically. Set `ALLOWED_ORIGINS` on the API to your Pages
URL (instead of `*`) to lock down CORS once you're past testing.

### 4. Batch/streaming instead of request/response

If the real deployment target is inline traffic scoring rather than an
HTTP API, replace `app.py`'s FastAPI layer with a consumer that reads from
your traffic pipeline (e.g. Kafka) and calls the same `_prepare_frame` +
`model.predict_proba` logic per message, writing alerts to wherever your
SOC tooling reads from (Splunk, a ticketing queue, etc.). The
preprocessing/model logic doesn't change — only the transport does.

## Closing the loop (feedback + retraining)

The pipeline trains once and stops there, which is fine for the
assignment's scope but not for a real deployment. To actually close the
architecture diagram's feedback loop:

1. **Log every decision**: store `(features, risk_score, alert, threshold)`
   for every scored connection, plus the analyst's eventual verdict
   (confirmed attack / false positive) once triaged.
2. **Track drift**: periodically compare the live score distribution and
   feature distributions against training-time statistics (e.g. population
   stability index) — this is how you catch constraint #2 ("attack patterns
   may change over time") before recall silently degrades.
3. **Scheduled retraining**: retrain on a rolling window of
   analyst-confirmed labels plus the original dataset, re-run
   `train_pipeline.py`'s evaluation step, and only promote the new model if
   it doesn't regress recall on known attack types or the zero-day holdout.
4. **Human-in-the-loop threshold tuning**: let the security team see the
   precision/recall trade-off curve (`threshold_curve.png`) and the live
   dashboard's per-model agreement view when alert volume becomes
   unmanageable, rather than hard-coding one threshold forever.

## Failure points worth naming (Task 10 asks for these explicitly)

- **Preprocessing**: a new `service` or `protocol_type` value never seen in
  training hits the "unknown category" fallback in `app.py` — this silently
  degrades accuracy on legitimately novel traffic rather than failing loudly.
  Consider alerting when the fallback path triggers frequently.
- **Model staleness**: no retraining schedule here — recall on genuinely new
  attack families will erode over time (see the Task 9 zero-day analysis run
  in the pipeline for a concrete before-the-fact estimate of this risk).
- **Threshold as a single global knob**: one threshold applied uniformly
  ignores that some traffic segments (e.g. externally-facing services) may
  warrant a lower threshold than others.
- **Single point of failure**: one inference service with no fallback if
  the container crashes mid-traffic-spike — a real deployment needs
  redundancy and a fail-open/fail-closed decision (does traffic pass
  through unexamined, or queue, if the IDS is down?).
- **Dashboard demo data**: `docs/index.html`'s sample connections are
  hand-set for illustration until it's pointed at a live backend — don't
  read its demo-mode numbers as real evaluated metrics; the real ones come
  from `artifacts/model_comparison.csv` and the `/models` endpoint.
