"""Drive a Playwright page's mouse along model-generated paths."""
import asyncio
import time

import numpy as np

from .behavior import DEFAULT_BEHAVIOR, Behavior
from .generate import PathGenerator


# Insets (left, top, right, bottom) of the part of an element not cut off by overflow-clipping
# ancestors or the viewport, so clicks only aim at what a person could actually see.
VISIBLE_INSETS_JS = """e => {
  const b = e.getBoundingClientRect();
  let l = b.left, t = b.top, r = b.right, bt = b.bottom;
  for (let p = e.parentElement; p && p !== document.body && p !== document.documentElement; p = p.parentElement) {
    const s = getComputedStyle(p), c = p.getBoundingClientRect();
    const x0 = c.left + p.clientLeft, y0 = c.top + p.clientTop;
    if (s.overflowX !== 'visible') { l = Math.max(l, x0); r = Math.min(r, x0 + p.clientWidth); }
    if (s.overflowY !== 'visible') { t = Math.max(t, y0); bt = Math.min(bt, y0 + p.clientHeight); }
  }
  l = Math.max(l, 0); t = Math.max(t, 0); r = Math.min(r, innerWidth); bt = Math.min(bt, innerHeight);
  return [l - b.left, t - b.top, b.right - r, b.bottom - bt];
}"""

# The nearest scrollable ancestor that hides part of the element vertically, or null (= the page).
CLIPPING_SCROLLER_JS = """e => {
  const b = e.getBoundingClientRect();
  for (let p = e.parentElement; p && p !== document.body && p !== document.documentElement; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight) {
      const c = p.getBoundingClientRect(), y0 = c.top + p.clientTop;
      if (b.top < y0 || b.bottom > y0 + p.clientHeight) return p;
    }
  }
  return null;
}"""


class HumanMouse:
    def __init__(self, page, generator: PathGenerator = None, position=(0.0, 0.0), temperature=0.8,
                 timing_jitter_ms=1.5, behavior: Behavior = None):
        """`generator` and `behavior` default to the bundled model and recorded timings."""
        self.page = page
        self.gen = generator or PathGenerator()
        self.position = tuple(position)  # Playwright doesn't expose the cursor position, so track it
        self.temperature = temperature
        self.jitter = timing_jitter_ms
        self.behavior = behavior or Behavior(DEFAULT_BEHAVIOR if DEFAULT_BEHAVIOR.exists() else None, rng=self.gen.rng)

    async def _sleep(self, seconds):  # the sync API swaps in a blocking sleep
        await asyncio.sleep(seconds)

    async def _play(self, times_ms, step):
        """Call `step(lo, hi)` on a real-time schedule until every index 1..n-1 is covered. Usually
        lo == hi; when behind (e.g. coarse timers on Windows), lo..hi spans every due index, so a
        single event catches up rather than a burst of back-to-back ones."""
        due = times_ms + self.gen.rng.normal(0.0, self.jitter, len(times_ms))
        t0 = time.perf_counter()
        i = 1
        while i < len(due):
            lo = i
            delay = t0 + due[i] / 1000 - time.perf_counter()
            if delay > 0:
                await self._sleep(delay)
            else:
                now_ms = (time.perf_counter() - t0) * 1000
                while i + 1 < len(due) and due[i + 1] <= now_ms:
                    i += 1
            await step(lo, i)
            i += 1

    async def _pause(self, name):
        await self._sleep(self.behavior.ms(name) / 1000)

    async def move_to(self, x, y, target_width=20.0, time_scale=1.0, not_before=0.0):
        """Move along a generated path. Generation takes real time (~0.1-0.3 s on CPU), so it
        happens first and only the rest of any required pause (`not_before`, a perf_counter
        time) is waited out."""
        pts, times = self.gen.path(self.position, (x, y), target_width, self.temperature)
        wait = not_before - time.perf_counter()
        if wait > 0:
            await self._sleep(wait)

        async def step(lo, i):  # only the latest position matters
            await self.page.mouse.move(float(pts[i, 0]), float(pts[i, 1]))

        await self._play(times * time_scale, step)
        self.position = (float(pts[-1, 0]), float(pts[-1, 1]))

    def _locator(self, target):
        return self.page.locator(target) if isinstance(target, str) else target

    async def _raw_box(self, target):
        box = await self._locator(target).bounding_box()
        if box is None:
            raise ValueError(f"element not visible: {target}")
        return box["x"], box["y"], box["width"], box["height"]

    async def _stable_box(self, loc, tries=20):
        """Bounding box once it stops changing (animations, late layout), like Playwright's checks."""
        prev = None
        for _ in range(tries):
            box = await loc.bounding_box()
            if box is not None and box == prev:
                break
            prev = box
            await self._sleep(0.03)
        if prev is None:
            raise ValueError("element not visible")
        return prev

    async def _visible_part(self, loc, box):
        il, it, ir, ib = await loc.evaluate(VISIBLE_INSETS_JS)
        x, y, w, h = box["x"] + il, box["y"] + it, box["width"] - il - ir, box["height"] - it - ib
        vp = self.page.viewport_size
        if vp:  # elements in iframes are also clipped by the page's viewport
            x2, y2 = min(x + w, vp["width"]), min(y + h, vp["height"])
            x, y = max(x, 0.0), max(y, 0.0)
            w, h = x2 - x, y2 - y
        return (x, y, w, h) if w > 1 and h > 1 else None

    async def _box(self, target, visible_only=True):
        """Wait for an element like page.click does (visible, then stable), flick-scroll it into
        view if it's (mostly) hidden, and return its box: the whole box, or with `visible_only`
        just the part that can be seen and clicked."""
        loc = self._locator(target)
        if hasattr(loc, "wait_for"):  # locators; element handles are already attached
            await loc.wait_for(state="visible")
        box = await self._stable_box(loc)
        vis = await self._visible_part(loc, box)
        if vis is None or vis[2] * vis[3] < 0.9 * box["width"] * box["height"]:
            scroller = await loc.evaluate_handle(CLIPPING_SCROLLER_JS)
            if await scroller.evaluate("p => p !== null"):
                await self._box(scroller)  # the scrolling container itself may be off-screen
                await self.scroll_to(loc, container=scroller)
            else:
                await self.scroll_to(loc)
            box = await self._stable_box(loc)
            vis = await self._visible_part(loc, box)
            if vis is None:
                raise ValueError(f"could not bring element into view: {target}")
        if visible_only:
            return vis
        return box["x"], box["y"], box["width"], box["height"]

    async def hover(self, target):
        """Move onto a locator (or selector string) like a person would and stay there."""
        x, y, w, h = await self._box(target)
        await self.move_to(*self.gen.point_in_box(x, y, w, h), min(w, h))

    async def click(self, target, button="left", click_count=1):
        """Move to a locator (or selector string), then press and release (`click_count=2`: double-click)."""
        await self.hover(target)
        rng = self.gen.rng
        await self._sleep(rng.uniform(0.04, 0.12))
        for n in range(1, click_count + 1):
            if n > 1:
                await self._sleep(rng.uniform(0.06, 0.14))
            await self.page.mouse.down(button=button, click_count=n)
            await self._sleep(rng.uniform(0.05, 0.11))
            await self.page.mouse.up(button=button, click_count=n)

    async def drag(self, source, target, precision=None):
        """Press on `source`, carry it to `target` and release.

        Either end can be an element (locator / selector) or an exact (x, y) page point:
        - element -> element (kanban card to a column): the grab offset is kept, so the card's
          center lands near a natural drop point inside the target.
        - (x, y) target (slider value, color picker spot): the cursor is released exactly there.
        - (x, y) source: press at that point, e.g. on a canvas or a slider track.
        `precision` is the effective target size in px (default: the target element's size, or
        8 px for a point); smaller means a slower, more careful approach."""
        if isinstance(source, tuple):
            await self.move_to(*source, precision or 8.0)
            grab = (0.0, 0.0)
        else:
            sx, sy, sw, sh = await self._box(source)
            await self.move_to(*self.gen.point_in_box(sx, sy, sw, sh), min(sw, sh))
            grab = (self.position[0] - (sx + sw / 2), self.position[1] - (sy + sh / 2))
        await self._sleep(self.gen.rng.uniform(0.04, 0.12))
        await self.page.mouse.down()
        start_at = time.perf_counter() + self.behavior.ms("drag_press_ms") / 1000
        if isinstance(target, tuple):
            (dx, dy), width = target, precision or 8.0
        else:
            tx, ty, tw, th = await self._box(target)
            dx, dy = self.gen.point_in_box(tx, ty, tw, th)
            dx, dy, width = dx + grab[0], dy + grab[1], precision or min(tw, th)
        await self.move_to(dx, dy, width, time_scale=self.behavior.drag_time_scale, not_before=start_at)
        await self._pause("drag_settle_ms")
        await self.page.mouse.up()

    async def point_at(self, target, fx, fy):
        """Page point at fraction (fx, fy) of an element's box, e.g. a spot in a color picker's
        saturation/brightness square: point_at("#sv", saturation, 1 - brightness)."""
        x, y, w, h = await self._box(target, visible_only=False)
        return x + fx * w, y + fy * h

    async def set_slider(self, slider, value=None, fraction=None, handle=None, thumb_px=16.0, max_corrections=2):
        """Drag a slider to `value`, or to `fraction` (0..1) of its range.

        Native <input type=range>: pass the input; `value` is in its min..max. The value is read
        back after the drag and, if it missed, nudged like a person would (up to
        `max_corrections` times). Returns the final value.
        Custom sliders: pass the track as `slider` and its `handle`, plus `fraction` (or `value`
        as 0..1). There's no generic way to read those back, so there are no corrections.
        Custom sliders are vertical if the track is taller than wide (then 0 = bottom)."""
        rng = self.gen.rng
        loc = self._locator(slider)
        x, y, w, h = await self._box(loc, visible_only=False)
        vertical = handle is not None and h > w
        start, length = (y + h, -h) if vertical else (x, w)  # axis origin and signed length at 0

        def to_px(f, inset):  # position of fraction f along the track, thumb inset at the ends
            return start + (inset / 2 if length > 0 else -inset / 2) + f * (length - (inset if length > 0 else -inset))

        def point(axis_pos, cross):
            return (cross, axis_pos) if vertical else (axis_pos, cross)

        cross = (x + w / 2) if vertical else (y + h / 2)
        if handle is None:
            lo, hi, step, cur = await loc.evaluate(
                "e => [e.min === '' ? 0 : +e.min, e.max === '' ? 100 : +e.max,"
                " e.step === 'any' ? 0 : (+e.step || 1), +e.value]")
            if fraction is None:
                fraction = (value - lo) / (hi - lo)
            want = lo + fraction * (hi - lo)
            px_per_unit = (abs(length) - thumb_px) / (hi - lo)
            tol = max(step / 2, 0.5 / px_per_unit)
            src = point(to_px((cur - lo) / (hi - lo), thumb_px) + rng.normal(0, 1.5), cross + rng.normal(0, 1.5))
            dst_axis = to_px(fraction, thumb_px)
            await self.drag(src, point(dst_axis, cross + rng.normal(0, 2)), precision=6.0)
            for _ in range(max_corrections):
                got = float(await loc.input_value())
                if abs(got - want) <= tol:
                    break
                await self._sleep(rng.uniform(0.2, 0.5))  # notice the value is off
                err = (want - got) * px_per_unit * (1 if length > 0 else -1)
                px, py = self.position
                await self.drag((px, py), point((py if vertical else px) + err, (px if vertical else py)), precision=4.0)
            return float(await loc.input_value())

        fraction = value if fraction is None else fraction
        hx, hy, hw, hh = await self._box(handle, visible_only=False)
        grab = self.gen.point_in_box(hx, hy, hw, hh)
        handle_axis = (hy + hh / 2) if vertical else (hx + hw / 2)
        shift = to_px(fraction, 0.0) - handle_axis  # custom handles' centers usually span the whole track
        gx, gy = grab
        await self.drag(grab, (gx, gy + shift) if vertical else (gx + shift, gy), precision=6.0)

    async def scroll_by(self, dy):
        """One trackpad flick of about `dy` px (positive = down) at the current cursor position."""
        g = self.behavior.gesture(dy)
        times = [g[0, 0] - 16.7] + list(g[:, 0])
        wheel = self.page.mouse.wheel

        async def step(lo, hi):  # skipped frames' deltas are summed, not lost
            dx, dy = g[lo - 1 : hi, 1:].sum(axis=0)
            await wheel(float(dx), float(dy))

        await self._play(np.array(times) - times[0], step)

    async def scroll_to(self, target, container=None, max_flicks=20):
        """Flick-scroll until `target` sits comfortably in view (re-measuring after each flick,
        like a person watching the page), with the cursor over `container` (default: the page)."""
        rng = self.gen.rng
        vp = self.page.viewport_size
        cx, cy, cw, ch = 0, 0, vp["width"], vp["height"]
        if container is not None:  # only the on-screen part: that's where the cursor and the target can be
            x, y, w, h = await self._raw_box(container)
            cx, cy = max(x, 0), max(y, 0)
            cw, ch = min(x + w, vp["width"]) - cx, min(y + h, vp["height"]) - cy
        px, py = self.position
        if not (cx < px < cx + cw and cy < py < cy + ch):
            await self.move_to(*self.gen.point_in_box(cx, cy, cw, ch), min(cw, ch))
        goal = rng.uniform(0.3, 0.6)
        prev = None
        for _ in range(max_flicks):
            box = await self._locator(target).bounding_box()
            if box is None:
                raise ValueError(f"element not found: {target}")
            center = box["y"] + box["height"] / 2
            if cy + 0.15 * ch < center < cy + 0.85 * ch:
                return
            if prev is not None and abs(center - prev) < 1 and cy <= box["y"] and box["y"] + box["height"] <= cy + ch:
                return  # at the end of the scroll range (e.g. a footer) but fully in view
            prev = center
            remaining = center - (cy + goal * ch)
            # People misjudge the distance a bit and rarely do one huge flick.
            size = remaining * rng.lognormal(0, 0.2)
            size = max(min(size, 2.5 * ch), -2.5 * ch)
            await self.scroll_by(size)
            await self._pause("scroll_pause_ms")
        raise TimeoutError(f"could not scroll {target} into view")
