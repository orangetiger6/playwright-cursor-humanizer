"""Let the trained model do the recorder tasks in a real browser.

Starts the local recorder server, opens it with ?source=bot, and does the click, drag,
type and scroll tasks with HumanMouse (--tasks picks which). The page records the
resulting events exactly as it does for a person, into data/bot/, so you can run evaluate.py --compare "data/bot/*.json"
to check what actually reached the browser.

A red dot with a fading trail shows where the model is moving the (invisible) automated
cursor, and pulses on each click; keystrokes and scroll flicks show up in a small label
beside it. It ignores pointer events, so it doesn't affect what the page records. Pass
--no-cursor to hide it.

    python scripts/demo.py --ckpt checkpoints/model.pt --trials 30
    python scripts/demo.py --tasks click          # pointing only
"""
import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect.server import serve_in_background  # noqa: E402
from humanmouse import Behavior, HumanMouse, PathGenerator  # noqa: E402

CURSOR_OVERLAY = """
window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.createElement("canvas");
  const dot = document.createElement("div");
  canvas.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483646";
  dot.style.cssText = "position:fixed;left:0;top:0;width:14px;height:14px;margin:-7px 0 0 -7px;" +
    "border-radius:50%;background:#ff2d2d;box-shadow:0 0 8px 3px rgba(255,45,45,.8),0 0 22px 8px rgba(255,45,45,.35);" +
    "pointer-events:none;z-index:2147483647;transition:transform .08s";
  const label = document.createElement("div");
  label.style.cssText = "position:fixed;left:0;top:0;padding:1px 6px;border-radius:4px;background:#ff2d2d;color:#fff;" +
    "font:12px/18px monospace;white-space:pre;pointer-events:none;z-index:2147483647;opacity:0;transition:opacity .3s";
  document.body.append(canvas, dot, label);
  let labelText = "", labelTimer = 0, scrollSum = 0;
  const say = (text, append) => {
    labelText = (append ? labelText + text : text).slice(-24);
    label.textContent = labelText;
    label.style.left = parseFloat(dot.style.left || 0) + 14 + "px";
    label.style.top = parseFloat(dot.style.top || 0) + 10 + "px";
    label.style.opacity = "1";
    clearTimeout(labelTimer);
    labelTimer = setTimeout(() => { label.style.opacity = "0"; labelText = ""; scrollSum = 0; }, 700);
  };
  addEventListener("keydown", e => say(e.key === " " ? "␣" : e.key.length === 1 ? e.key : `[${e.key}]`, true), true);
  addEventListener("wheel", e => {
    scrollSum += e.deltaY;
    say(`${scrollSum > 0 ? "↓" : "↑"} scroll ${Math.abs(Math.round(scrollSum))}px`);
  }, true);
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
    fit = args.behavior if Path(args.behavior).exists() else None
    behavior = Behavior(fit, rng=gen.rng)
    print(f"behavior: {fit or 'built-in defaults'}")
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=args.headless)
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            if args.cursor:
                await page.add_init_script(CURSOR_OVERLAY)
            await page.goto(f"http://127.0.0.1:{args.port}/?source=bot&tasks={args.tasks}")
            mouse = HumanMouse(page, gen, position=(640, 700), temperature=args.temperature, behavior=behavior)
            await page.mouse.move(*mouse.position)

            await mouse.click("#begin")
            for i in range(args.trials):
                task = await page.evaluate("window.currentTask()")
                if task["kind"] == "drag":
                    await mouse.drag("#chip", "#zone")
                elif task["kind"] == "type":
                    await mouse.type_into("#field", task["phrase"], submit=True)
                elif task["kind"] == "scroll":
                    await mouse.scroll_to(".row.hot", container="#scroller")
                    await mouse.click(".row.hot")
                else:
                    await mouse.click("#target")
                await asyncio.sleep(gen.rng.uniform(0.05, 0.25))
            print(await page.evaluate("window.saveSession()"))
            await browser.close()
    finally:
        server.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--trials", type=int, default=30, help="number of tasks")
    p.add_argument("--tasks", default="click,drag,type,scroll", help="comma-separated mix of recorder tasks")
    p.add_argument("--behavior", default="checkpoints/behavior.json", help="from fit_behavior.py; defaults if missing")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--seed", type=int)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-cursor", dest="cursor", action="store_false", help="hide the red cursor overlay")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
