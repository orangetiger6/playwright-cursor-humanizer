"""Drive a Playwright page's mouse along model-generated paths."""
import asyncio
import time

from .generate import PathGenerator


class HumanMouse:
    def __init__(self, page, generator: PathGenerator, position=(0.0, 0.0), temperature=0.8, timing_jitter_ms=1.5):
        self.page = page
        self.gen = generator
        self.position = tuple(position)  # Playwright doesn't expose the cursor position, so track it
        self.temperature = temperature
        self.jitter = timing_jitter_ms

    async def move_to(self, x, y, target_width=20.0):
        pts, times = self.gen.path(self.position, (x, y), target_width, self.temperature)
        due = times + self.gen.rng.normal(0.0, self.jitter, len(times))
        t0 = time.perf_counter()
        i = 1
        while i < len(pts):
            delay = t0 + due[i] / 1000 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Behind schedule (e.g. coarse timers on Windows): jump to the latest due point
                # rather than firing a burst of back-to-back events.
                now_ms = (time.perf_counter() - t0) * 1000
                while i + 1 < len(pts) and due[i + 1] <= now_ms:
                    i += 1
            await self.page.mouse.move(float(pts[i, 0]), float(pts[i, 1]))
            i += 1
        self.position = (float(pts[-1, 0]), float(pts[-1, 1]))

    async def click(self, target, button="left"):
        """Move to a locator (or selector string) like a person would, then press and release."""
        locator = self.page.locator(target) if isinstance(target, str) else target
        box = await locator.bounding_box()
        if box is None:
            raise ValueError(f"element not visible: {target}")
        x, y = self.gen.point_in_box(box["x"], box["y"], box["width"], box["height"])
        await self.move_to(x, y, min(box["width"], box["height"]))
        rng = self.gen.rng
        await asyncio.sleep(rng.uniform(0.04, 0.12))
        await self.page.mouse.down(button=button)
        await asyncio.sleep(rng.uniform(0.05, 0.11))
        await self.page.mouse.up(button=button)
