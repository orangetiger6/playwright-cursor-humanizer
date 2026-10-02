# humanmouse

Learn human mouse movement, typing and trackpad scrolling from your own recordings and replay
them in Playwright: clicks, drags (kanban cards, sliders, color pickers), typing and scrolling.

## Quick start (included model)

A trained model and timings ship inside the package, so you can use it without recording anything. The path model was pretrained on [SapiMouse](https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.html)
(Antal et al., 2021; 120 users) and fine-tuned on ~300 clicks from one person on a trackpad; the typing,
drag and scroll timing comes from that same person. For movement that looks like *you*, record and train
your own (below).

```bash
pip install -e .
python -m playwright install chromium
python scripts/demo.py --trials 20      # opens Chromium and performs the recorder tasks; watch the red cursor
```

**Install size:** the path model runs on PyTorch. On Linux, `pip` pulls the CUDA build by default (several GB).
Generation is CPU-only, so install the CPU wheel first (~200 MB):
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. Windows and macOS get a CPU build by default.

**In your own script** (async):

```python
from humanmouse import HumanMouse

human = HumanMouse(page)                       # bundled model and timings
await human.click("button#submit")
await human.type_into("input[name=q]", "hello world", submit=True)
await human.drag("#card", "#done-column")
```

**With `sync_playwright`:**

```python
from humanmouse.sync_api import HumanMouse

human = HumanMouse(page)
human.click("button#submit")
```

Like `page.click`, the actions take a selector or a locator, wait for the element to be visible and stop
moving, and scroll it into view first if needed. That includes elements inside scrolling lists, using
trackpad flicks rather than a jump. Playwright doesn't expose where the cursor is, so `HumanMouse` tracks it.
Use one `HumanMouse` per page, and pass `position=(x, y)` if the cursor isn't at the top-left corner.

```
collect/            recorder page + tiny server (writes data/raw/*.json)
humanmouse/
  data.py           resample → trim → canonical frame (start (0,0), target (1,0)) + condition vector
  model.py          GRU + bivariate Gaussian mixture density head (Graves 2013), end-of-movement flag
  generate.py       PathGenerator: sample, snap endpoint, map back to screen coords
  behavior.py       drag / typing / trackpad-scroll timing: fitted from recordings, with defaults
  playwright_driver.py  HumanMouse: move_to / click / drag / type_into / scroll_to with real-time pacing
  sync_api.py       the same HumanMouse for sync_playwright
  assets/           the bundled model.pt and behavior.json
  metrics.py        kinematic metrics, Fitts' law fit, KS statistic
scripts/
  make_synthetic.py hand-built fake data for smoke tests
  import_public.py  SapiMouse / BOUN / Balabit CSV logs → trial JSON
  train.py          training loop
  evaluate.py       real vs generated vs in-browser comparison (+ plots)
  fit_behavior.py   drag / type / scroll recordings → checkpoints/behavior.json
  demo.py           model performs the recorder task in Chromium, recorded to data/bot/
tests/              unit tests + headless-Chromium tests of every action (pytest)
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
pip install -e ".[eval,test]"      # or: pip install -r requirements.txt
python -m playwright install chromium
```

**1. Record** (the most important step: quality and quantity of data drive everything)

```bash
python collect/server.py
```

Open http://127.0.0.1:8765, click **Begin**, then follow the instruction in the top bar. Most tasks are
"click the blue target"; mixed in are drags (orange square into the dashed box), typing (click the field,
type the phrase, Enter) and scrolling (scroll the list to the blue row and click it). Saves every 25
trials (or press `S`). `?tasks=scroll,type` limits the mix. Tips:
- Aim for 1,500+ trials across several sessions and days. Short sessions keep the data free of fatigue.
- Record on the same mouse, OS pointer speed and screen scaling that you want to reproduce.
- Work at a normal pace. Don't try to be fast or careful; the goal is ordinary behavior.
- Scroll with the device you want reproduced. Only pixel-mode wheel events (trackpads) become scroll flicks.

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
python scripts/demo.py --ckpt checkpoints/model.pt --behavior checkpoints/behavior.json --trials 50
python scripts/evaluate.py --ckpt checkpoints/model.pt --compare "data/bot/*.json"
```

For drags, typing and scrolling, the model isn't needed: their pauses, keystroke rhythm and trackpad
flicks are sampled from your recordings (anything with fewer than 8 samples uses built-in defaults).

```bash
python scripts/fit_behavior.py --data "data/raw/*.json"      # -> checkpoints/behavior.json
```

KS columns near 0 mean the generated distribution matches yours. The `browser` column measures what
actually reached the page, which captures timing effects the model alone can't see.

**5. Use it**

Leave out `generator` and `behavior` to use the bundled ones. To use your own:

```python
from humanmouse import Behavior, HumanMouse, PathGenerator

gen = PathGenerator("checkpoints/model.pt")
mouse = HumanMouse(page, gen, position=(400, 300), behavior=Behavior("checkpoints/behavior.json", rng=gen.rng))
await mouse.click("button#submit")
await mouse.move_to(800, 200)
await mouse.drag("#card", "#done-column")
await mouse.drag((x0, y), (x1, y))            # exact points: press here, release there
await mouse.set_slider("#volume", 70)          # <input type=range>: reads back and nudges if off
await mouse.set_slider("#track", fraction=0.3, handle="#knob")   # custom (div) slider
await mouse.drag(await mouse.point_at("#sv", 0.1, 0.9), await mouse.point_at("#sv", 0.8, 0.2))  # color picker
await mouse.type_into("input[name=q]", "hello world", submit=True)
await mouse.scroll_to("#footer")               # trackpad flicks, re-checking after each
```

## Tests

```bash
python -m pytest
```

The browser tests run in headless Chromium (about a minute). They drive real sliders, a color picker, an
HTML5 drag-and-drop kanban board, a text field, scrolling lists, off-screen and late-appearing elements, and
the sync API. They're skipped if Chromium is missing.

## Knobs and next steps

- `temperature` (default 0.8): lower values give smoother, more stereotyped paths; higher values give more variety and wobble.
- Hold out a whole recording session for validation instead of random trials to measure generalization honestly.
- Ideas: condition on a per-user/session embedding; typing mistakes and corrections; model the idle
  drift between actions; replace the GRU with a small transformer once you have >10k trials.

Use this responsibly: for testing your own apps, UX/accessibility research, and demos, within the terms of the sites you automate.

## License

MIT (see `LICENSE`). If you use the included model, please also cite SapiMouse:
M. Antal, N. Fejér, K. Buza, "SapiMouse: Mouse Dynamics-based User Authentication Using Deep Feature Learning", SACI 2021.
