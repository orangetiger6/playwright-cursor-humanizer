# humanmouse

Learn human mouse movement from your own recordings and replay it in Playwright.

```
collect/            recorder page + tiny server (writes data/raw/*.json)
humanmouse/
  data.py           resample → trim → canonical frame (start (0,0), target (1,0)) + condition vector
  model.py          GRU + bivariate Gaussian mixture density head (Graves 2013), end-of-movement flag
  generate.py       PathGenerator: sample, snap endpoint, map back to screen coords
  playwright_driver.py  HumanMouse: move_to / click with real-time pacing
  metrics.py        kinematic metrics, Fitts' law fit, KS statistic
scripts/
  make_synthetic.py hand-built fake data for smoke tests
  import_public.py  SapiMouse / BOUN / Balabit CSV logs → trial JSON
  train.py          training loop
  evaluate.py       real vs generated vs in-browser comparison (+ plots)
  demo.py           model performs the recorder task in Chromium, recorded to data/bot/
```

## How it works

Each trial (move from one click to the next target click) is resampled onto a 10 ms grid,
the pre-movement reaction time is trimmed, and the path is rotated and scaled so that it runs from
`(0,0)` to `(1,0)`. The removed information becomes the condition vector
`[log(D/100), log(W/D), sin θ, cos θ]` (distance, target width → Fitts' index of difficulty, direction).

The model autoregressively predicts the next `(dx, dy)` as a mixture of 2D Gaussians, plus a
probability that the movement ends (the click). Its inputs at each step are the previous delta, the current
offset from the target, and the condition. At generation time, the best of a few samples is
chosen, and the small remaining endpoint error is blended into the tail of the path.

## Usage

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

**1. Record** (the most important step: quality and quantity of data drive everything)

```bash
python collect/server.py
```

Open http://127.0.0.1:8765, click **Begin**, then keep clicking the blue targets. Saves every 25 trials
(or press `S`). Tips:
- Aim for 1,500+ trials across several sessions and days. Short sessions keep the data free of fatigue.
- Record on the same mouse, OS pointer speed and screen scaling that you want to reproduce.
- Work at a normal pace. Don't try to be fast or careful; the goal is ordinary behavior.

**2. Smoke-test the pipeline** (optional, no recording needed)

```bash
python scripts/make_synthetic.py
python scripts/train.py --data "data/synthetic/*.json" --out checkpoints/smoke.pt --epochs 15 --hidden 128
```

**3. Train**

```bash
python scripts/train.py --data "data/raw/*.json" --out checkpoints/model.pt --epochs 80
```

**3b. Optional: pretrain on a public dataset, then fine-tune on your data**

Public logs are free-form use (no known targets), so they teach general motor dynamics. Your own
recordings then teach your hand and the target-width effect. `scripts/import_public.py` turns each
movement between two clicks into a trial, drops drags, and cuts off long pauses.

| dataset | what | size | license |
|---|---|---|---|
| [SapiMouse](https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.html) | 120 users, browser, 1 + 3 min sessions | 8 MB zip | none stated; cite Antal et al. 2021 |
| [Boğaziçi (BOUN)](https://data.mendeley.com/datasets/w6cxr8yc7p/2) | 24 users, months of desktop use | 4.9 GB split zip | CC BY 4.0 |
| [Balabit challenge](https://github.com/balabit/Mouse-Dynamics-Challenge) | 10 users, remote desktop sessions | small | see repo |

```bash
# unzip into data/external/<name>/ first
python scripts/import_public.py sapimouse data/external/sapimouse          # -> data/public/sapimouse/
python scripts/train.py --data "data/public/sapimouse/*.json" --out checkpoints/pretrained.pt
python scripts/train.py --init checkpoints/pretrained.pt --data "data/raw/*.json" --out checkpoints/model.pt --lr 2e-4 --epochs 30
```

For BOUN, pass `--max-trials-per-user 3000` so a few heavy users don't dominate. Target width is unknown
for public data (`--default-width`, default 24 px). Fine-tuning on your recordings fixes the Fitts'-law
dependence on width.

**4. Evaluate**

```bash
python scripts/evaluate.py --ckpt checkpoints/model.pt --plot eval.png
python scripts/demo.py --ckpt checkpoints/model.pt --trials 50
python scripts/evaluate.py --ckpt checkpoints/model.pt --compare "data/bot/*.json"
```

KS columns near 0 mean the generated distribution matches yours. The `browser` column measures what
actually reached the page, which captures timing effects the model alone can't see.

**5. Use it**

```python
from humanmouse import HumanMouse, PathGenerator

gen = PathGenerator("checkpoints/model.pt")
mouse = HumanMouse(page, gen, position=(400, 300))
await mouse.click("button#submit")
await mouse.move_to(800, 200)
```

## Knobs and next steps

- `temperature` (default 0.8): lower values give smoother, more stereotyped paths; higher values give more variety and wobble.
- Hold out a whole recording session for validation instead of random trials to measure generalization honestly.
- Ideas: condition on a per-user/session embedding; add scroll and drag trials to the recorder; model the idle
  drift between actions; replace the GRU with a small transformer once you have >10k trials.

Use this responsibly: for testing your own apps, UX/accessibility research, and demos, within the terms of the sites you automate.
