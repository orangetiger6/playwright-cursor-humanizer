"""End to end: each HumanMouse action against real page widgets in headless Chromium."""
import pytest

from humanmouse import HumanMouse

WIDGETS = """<style>#fat{-webkit-appearance:none;height:8px}
#fat::-webkit-slider-thumb{-webkit-appearance:none;width:40px;height:30px;background:#36c}</style>
<body style="margin:0">
<input id=r1 type=range style="position:absolute;left:100px;top:60px;width:400px">
<input id=r2 type=range min=0 max=1000 step=5 value=0 style="position:absolute;left:100px;top:140px;width:300px">
<input id=fat type=range min=-50 max=50 value=0 style="position:absolute;left:100px;top:220px;width:500px">
<div id=track style="position:absolute;left:800px;top:100px;width:10px;height:400px;background:#ddd">
  <div id=knob style="position:absolute;left:-10px;width:30px;height:20px;background:#c33;top:390px"></div></div>
<canvas id=sv width=200 height=200 style="position:absolute;left:300px;top:350px;background:#f80"></canvas>
<div id=todo class=col style="position:absolute;left:560px;top:300px;width:180px;height:150px;background:#eee">
  <div id=card draggable=true style="margin:10px;height:40px;background:#fc0">card</div></div>
<div id=done class=col style="position:absolute;left:560px;top:500px;width:180px;height:150px;background:#dfd"></div>
<button id=far onclick="this.textContent='clicked'" style="position:absolute;left:100px;top:1200px">far</button>
<script>
let d = null;
knob.onmousedown = e => { d = e.clientY - knob.getBoundingClientRect().top - 10 };
addEventListener('mousemove', e => {
  if (d === null) return;
  const r = track.getBoundingClientRect();
  knob.style.top = Math.max(-10, Math.min(r.height - 10, e.clientY - r.top - d - 10)) + 'px';
});
addEventListener('mouseup', () => d = null);
window.knobFrac = () => {
  const r = track.getBoundingClientRect(), k = knob.getBoundingClientRect();
  return 1 - ((k.top + 10) - r.top) / r.height;
};
window.pick = null; let down = false;
sv.onmousedown = e => { down = true; pick = [e.offsetX, e.offsetY] };
addEventListener('mousemove', e => { if (down) pick = [e.offsetX, e.offsetY] });
addEventListener('mouseup', () => down = false);
card.ondragstart = e => e.dataTransfer.setData('text', 'card');
for (const c of document.querySelectorAll('.col')) {
  c.ondragover = e => e.preventDefault();
  c.ondrop = e => { e.preventDefault(); c.appendChild(card) };
}
</script></body>"""

ROWS = "".join(f'<div id="row{i}" style="height:44px">row {i}</div>' for i in range(200))
SCROLLER = f'<body style="margin:0"><div id="box" style="height:400px;width:500px;overflow:auto">{ROWS}</div></body>'


def test_native_sliders(generator, run_page):
    async def fn(page):
        m = HumanMouse(page, generator, position=(500, 650))
        return [await m.set_slider(sel, v) for sel, v in [("#r1", 70), ("#r2", 615), ("#fat", -32)]]
    assert run_page(WIDGETS, fn) == [70, 615, -32]


def test_custom_slider_color_picker_and_kanban(generator, run_page):
    async def fn(page):
        m = HumanMouse(page, generator, position=(500, 650))
        await m.set_slider("#track", fraction=0.75, handle="#knob")
        await m.drag(await m.point_at("#sv", 0.1, 0.9), await m.point_at("#sv", 0.8, 0.25))
        await m.drag("#card", "#done")
        return await page.evaluate("[knobFrac(), pick, card.parentElement.id]")
    frac, pick, column = run_page(WIDGETS, fn)
    assert abs(frac - 0.75) < 0.01
    assert pick == [160, 50]
    assert column == "done"


def test_scroll_to_row(generator, run_page):
    async def fn(page):
        m = HumanMouse(page, generator, position=(250, 200))
        await m.scroll_to("#row150", container="#box")
        box = await page.locator("#row150").bounding_box()
        return box["y"] + box["height"] / 2
    center = run_page(SCROLLER, fn)
    assert 0.15 * 400 < center < 0.85 * 400


TALL = """<body style="margin:0"><div style="height:2500px"></div>
<button id=far onclick="this.textContent='clicked'">far</button>
<div style="height:300px;width:400px;overflow:auto" id=list>""" + "".join(
    f'<div id="item{i}" style="height:40px" onclick="window.picked={i}">item {i}</div>' for i in range(100)) + """
</div><div style="height:1500px"></div></body>"""


def test_click_scrolls_off_screen_targets_into_view(generator, run_page):
    async def fn(page):
        m = HumanMouse(page, generator, position=(500, 300))
        await m.click("#far")  # below the fold: the page is flick-scrolled first
        await m.click("#item80")  # inside a scrolling list that is itself off-screen
        return await page.evaluate("[far.textContent, window.picked, scrollY > 0]")
    assert run_page(TALL, fn) == ["clicked", 80, True]


def test_waits_for_late_elements(generator, run_page):
    late = """<body><script>setTimeout(() => document.body.insertAdjacentHTML('beforeend',
      '<button id=late onclick="this.textContent=1">late</button>'), 500)</script></body>"""

    async def fn(page):
        await HumanMouse(page, generator, position=(300, 300)).click("#late")
        return await page.locator("#late").text_content()
    assert run_page(late, fn) == "1"


def test_sync_api_with_bundled_defaults():
    sync_api = pytest.importorskip("playwright.sync_api")
    from humanmouse.sync_api import HumanMouse as SyncHumanMouse

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"Chromium not available: {e}")
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.set_content(WIDGETS)
        human = SyncHumanMouse(page, position=(500, 650))  # bundled model and timings
        human.click("#far")
        value = human.set_slider(page.locator("#r1"), 30)  # locators work as well as selectors
        human.drag("#card", "#done")
        result = [page.locator("#far").text_content(), value, page.evaluate("card.parentElement.id")]
        browser.close()
    assert result == ["clicked", 30, "done"]


COUNT_MOVES = "window.moves = 0; addEventListener('mousemove', () => moves++)"


def test_humanize_routes_playwright_calls(generator, run_page):
    from humanmouse import humanize

    async def fn(page):
        await page.evaluate(COUNT_MOVES)
        human = humanize(page, generator=generator, position=(500, 650))
        await page.click("#far")  # below the fold: scrolled into view first
        await page.locator("#card").drag_to(page.locator("#done"))
        moved = await page.evaluate("moves")
        await page.mouse.move(40, 50)
        tracked = human.position
        await page.locator("#r1").click(position={"x": 5, "y": 5})  # not supported: Playwright's own click
        return [await page.evaluate("[far.textContent, card.parentElement.id]"), moved > 50, tracked, human.position]
    texts, many_moves, tracked, after_native = run_page(WIDGETS, fn)
    assert texts == ["clicked", "done"]
    assert many_moves
    assert tracked == pytest.approx((40, 50)) and after_native == pytest.approx((40, 50))


def test_humanize_sync_context():
    sync_api = pytest.importorskip("playwright.sync_api")
    from humanmouse import humanize

    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"Chromium not available: {e}")
        context = browser.new_context(viewport={"width": 1000, "height": 700})
        humanize(context)  # pages opened later are covered too
        page = context.new_page()
        page.set_content(WIDGETS)
        page.evaluate(COUNT_MOVES)
        page.dblclick("#card")
        page.locator("#far").click()
        result = [page.locator("#far").text_content(), page.evaluate("moves") > 20]
        browser.close()
    assert result == ["clicked", True]
