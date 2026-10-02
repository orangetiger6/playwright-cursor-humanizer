"""Let the trained model do the recorder task in a real browser.

Starts the local recorder server, opens it with ?source=bot, and clicks targets
with HumanMouse. The page records the resulting events exactly as it does for a
person, into data/bot/, so you can run evaluate.py --compare "data/bot/*.json"
to check what actually reached the browser.

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


async def run(args):
    server = serve_in_background(args.port)
    gen = PathGenerator(args.ckpt, seed=args.seed)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=args.headless)
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
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
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
