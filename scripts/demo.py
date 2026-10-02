"""Let the trained model do the recorder task in a real browser.

Starts the local recorder server, opens it with ?source=bot, and clicks targets
with HumanMouse. The page records the resulting events exactly as it does for a
person, into data/bot/, so you can run evaluate.py --compare "data/bot/*.json"
to check what actually reached the browser.

A red dot with a fading trail shows where the model is moving the (invisible) automated
cursor, and pulses on each click. It ignores pointer events, so it doesn't affect what the
page records. Pass --no-cursor to hide it.

    python scripts/demo.py --ckpt checkpoints/model.pt --trials 30
"""
import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect.server import serve_in_background  # noqa: E402
from humanmouse import HumanMouse, PathGenerator  # noqa: E402

CURSOR_OVERLAY = """
window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.createElement("canvas");
  const dot = document.createElement("div");
  canvas.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483646";
  dot.style.cssText = "position:fixed;left:0;top:0;width:14px;height:14px;margin:-7px 0 0 -7px;" +
    "border-radius:50%;background:#ff2d2d;box-shadow:0 0 8px 3px rgba(255,45,45,.8),0 0 22px 8px rgba(255,45,45,.35);" +
    "pointer-events:none;z-index:2147483647;transition:transform .08s";
  document.body.append(canvas, dot);
  const ctx = canvas.getContext("2d");
  const trail = [];  // [x, y, time]
  const resize = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  resize();
  addEventListener("resize", resize);
  addEventListener("mousemove", e => {
    dot.style.left = e.clientX + "px";
    dot.style.top = e.clientY + "px";
    trail.push([e.clientX, e.clientY, performance.now()]);
  }, true);
  addEventListener("mousedown", e => {
    dot.style.transform = "scale(1.8)";
    const ring = document.createElement("div");
    ring.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:40px;height:40px;` +
      "margin:-20px 0 0 -20px;border:2px solid #ff2d2d;border-radius:50%;pointer-events:none;" +
      "z-index:2147483647;transition:transform .4s ease-out,opacity .4s ease-out";
    document.body.append(ring);
    requestAnimationFrame(() => { ring.style.transform = "scale(1.6)"; ring.style.opacity = "0"; });
    setTimeout(() => ring.remove(), 450);
  }, true);
  addEventListener("mouseup", () => { dot.style.transform = ""; }, true);
  (function draw() {
    const now = performance.now(), keep = 600;  // trail length in ms
    while (trail.length && now - trail[0][2] > keep) trail.shift();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    for (let i = 1; i < trail.length; i++) {
      const a = 1 - (now - trail[i][2]) / keep;
      ctx.strokeStyle = `rgba(255,45,45,${0.7 * a})`;
      ctx.lineWidth = 1 + 3 * a;
      ctx.beginPath();
      ctx.moveTo(trail[i - 1][0], trail[i - 1][1]);
      ctx.lineTo(trail[i][0], trail[i][1]);
      ctx.stroke();
    }
    requestAnimationFrame(draw);
  })();
});
"""


async def run(args):
    server = serve_in_background(args.port)
    gen = PathGenerator(args.ckpt, seed=args.seed)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=args.headless)
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            if args.cursor:
                await page.add_init_script(CURSOR_OVERLAY)
            await page.goto(f"http://127.0.0.1:{args.port}/?source=bot")
            mouse = HumanMouse(page, gen, position=(640, 700), temperature=args.temperature)
            await page.mouse.move(*mouse.position)

            await mouse.click("#begin")
            for i in range(args.trials):
                await mouse.click("#target")
                await asyncio.sleep(gen.rng.uniform(0.05, 0.25))
            print(await page.evaluate("window.saveSession()"))
            await browser.close()
    finally:
        server.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--seed", type=int)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-cursor", dest="cursor", action="store_false", help="hide the red cursor overlay")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
