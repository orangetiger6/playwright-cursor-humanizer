"""End to end: each HumanMouse action against real page widgets in headless Chromium."""
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
<input id=field style="position:absolute;left:100px;top:600px;width:300px">
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


def test_typing_is_exact(generator, run_page):
    text = "Hello, World! #42 abc"

    async def fn(page):
        m = HumanMouse(page, generator, position=(500, 400))
        await m.type_into("#field", text)
        return await page.locator("#field").input_value()
    assert run_page(WIDGETS, fn) == text


def test_scroll_to_row(generator, run_page):
    async def fn(page):
        m = HumanMouse(page, generator, position=(250, 200))
        await m.scroll_to("#row150", container="#box")
        box = await page.locator("#row150").bounding_box()
        return box["y"] + box["height"] / 2
    center = run_page(SCROLLER, fn)
    assert 0.15 * 400 < center < 0.85 * 400
