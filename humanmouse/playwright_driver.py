"""Drive a Playwright page's mouse along model-generated paths."""
import asyncio
import time

import numpy as np

from .behavior import SHIFTED, Behavior
from .generate import PathGenerator


class HumanMouse:
    def __init__(self, page, generator: PathGenerator, position=(0.0, 0.0), temperature=0.8, timing_jitter_ms=1.5,
                 behavior: Behavior = None):
        self.page = page
        self.gen = generator
        self.position = tuple(position)  # Playwright doesn't expose the cursor position, so track it
        self.temperature = temperature
        self.jitter = timing_jitter_ms
        self.behavior = behavior or Behavior(rng=generator.rng)
        self._resume_at = 0.0  # after typing, the hand needs time to get back to the mouse

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
                await asyncio.sleep(delay)
            else:
                now_ms = (time.perf_counter() - t0) * 1000
                while i + 1 < len(due) and due[i + 1] <= now_ms:
                    i += 1
            await step(lo, i)
            i += 1

    async def _pause(self, name):
        await asyncio.sleep(self.behavior.ms(name) / 1000)

    async def move_to(self, x, y, target_width=20.0, time_scale=1.0, not_before=0.0):
        """Move along a generated path. Generation takes real time (~0.1-0.3 s on CPU), so it
        happens first and only the rest of any required pause (`not_before`, a perf_counter
        time) is waited out."""
        pts, times = self.gen.path(self.position, (x, y), target_width, self.temperature)
        wait = max(self._resume_at, not_before) - time.perf_counter()
        self._resume_at = 0.0
        if wait > 0:
            await asyncio.sleep(wait)

        async def step(lo, i):  # only the latest position matters
            await self.page.mouse.move(float(pts[i, 0]), float(pts[i, 1]))

        await self._play(times * time_scale, step)
        self.position = (float(pts[-1, 0]), float(pts[-1, 1]))

    async def _box(self, target):
        locator = self.page.locator(target) if isinstance(target, str) else target
        box = await locator.bounding_box()
        if box is None:
            raise ValueError(f"element not visible: {target}")
        return box["x"], box["y"], box["width"], box["height"]

    async def click(self, target, button="left"):
        """Move to a locator (or selector string) like a person would, then press and release."""
        x, y, w, h = await self._box(target)
        await self.move_to(*self.gen.point_in_box(x, y, w, h), min(w, h))
        rng = self.gen.rng
        await asyncio.sleep(rng.uniform(0.04, 0.12))
        await self.page.mouse.down(button=button)
        await asyncio.sleep(rng.uniform(0.05, 0.11))
        await self.page.mouse.up(button=button)

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
        await asyncio.sleep(self.gen.rng.uniform(0.04, 0.12))
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
        x, y, w, h = await self._box(target)
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
        loc = self.page.locator(slider) if isinstance(slider, str) else slider
        x, y, w, h = await self._box(loc)
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
                await asyncio.sleep(rng.uniform(0.2, 0.5))  # notice the value is off
                err = (want - got) * px_per_unit * (1 if length > 0 else -1)
                px, py = self.position
                await self.drag((px, py), point((py if vertical else px) + err, (px if vertical else py)), precision=4.0)
            return float(await loc.input_value())

        fraction = value if fraction is None else fraction
        hx, hy, hw, hh = await self._box(handle)
        grab = self.gen.point_in_box(hx, hy, hw, hh)
        handle_axis = (hy + hh / 2) if vertical else (hx + hw / 2)
        shift = to_px(fraction, 0.0) - handle_axis  # custom handles' centers usually span the whole track
        gx, gy = grab
        await self.drag(grab, (gx, gy + shift) if vertical else (gx + shift, gy), precision=6.0)

    async def type_into(self, target, text, submit=False):
        """Click a field (skip with target=None to type where the focus already is), then type
        with a human rhythm: key rollover, explicit Shift for capitals and symbols."""
        if target is not None:
            await self.click(target)
        await self._pause("homing_ms")
        text = text + ("\n" if submit else "")
        b, rng = self.behavior, self.gen.rng
        events, t, prev, shift_until = [], 0.0, " ", -1.0
        for i, ch in enumerate(text):
            if i:
                t += b.key_gap_ms(prev, ch)
            key = "Enter" if ch == "\n" else ch
            hold = b.ms("hold_ms")
            if ch == prev:  # the same key has to come up before it can go down again
                hold = min(hold, 0.7 * (t - events[-1][0]) if events else hold)
            if ch.isupper() or ch in SHIFTED:
                start = t - rng.uniform(30, 80)
                if start <= shift_until:  # still held from the previous capital: extend it
                    events = [e for e in events if e[1:] != ("up", "Shift")]
                else:
                    events.append((max(start, events[-1][0] + 1 if events else start), "down", "Shift"))
                shift_until = t + hold + rng.uniform(10, 40)
                events.append((shift_until, "up", "Shift"))
            events.append((t, "down", key))
            events.append((t + hold, "up", key))
            prev = ch
        events.sort(key=lambda e: e[0])
        times = [e[0] for e in events]
        times = [times[0] - 1.0] + times  # _play starts at index 1
        kb = self.page.keyboard

        async def step(lo, hi):  # every key event must happen, in order
            for at, action, key in events[lo - 1 : hi]:
                try:
                    await (kb.down(key) if action == "down" else kb.up(key))
                except Exception:  # not on the US layout (é, emoji...): insert the text instead
                    if action == "down":
                        await kb.insert_text(key)

        await self._play(np.array(times) - times[0], step)
        # resume_ms is measured from the last key press, not its release.
        last_down = max(e[0] for e in events if e[1] == "down")
        self._resume_at = time.perf_counter() + (b.ms("resume_ms") - (events[-1][0] - last_down)) / 1000

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
        if container is not None:
            cx, cy, cw, ch = await self._box(container)
        else:
            vp = self.page.viewport_size
            cx, cy, cw, ch = 0, 0, vp["width"], vp["height"]
        px, py = self.position
        if not (cx < px < cx + cw and cy < py < cy + ch):
            await self.move_to(*self.gen.point_in_box(cx, cy, cw, ch), min(cw, ch))
        goal = rng.uniform(0.3, 0.6)
        for _ in range(max_flicks):
            locator = self.page.locator(target) if isinstance(target, str) else target
            box = await locator.bounding_box()
            if box is None:
                raise ValueError(f"element not found: {target}")
            center = box["y"] + box["height"] / 2
            if cy + 0.15 * ch < center < cy + 0.85 * ch:
                return
            remaining = center - (cy + goal * ch)
            # People misjudge the distance a bit and rarely do one huge flick.
            size = remaining * rng.lognormal(0, 0.2)
            size = max(min(size, 2.5 * ch), -2.5 * ch)
            await self.scroll_by(size)
            await self._pause("scroll_pause_ms")
        raise TimeoutError(f"could not scroll {target} into view")
