import asyncio
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


@pytest.fixture(scope="session")
def generator():
    if not MODEL.exists():
        pytest.skip("checkpoints/model.pt not found")
    from humanmouse import PathGenerator
    return PathGenerator(str(MODEL), seed=0)


@pytest.fixture
def run_page():
    """run_page(html, fn): open html in headless Chromium and return `await fn(page)`."""
    pw = pytest.importorskip("playwright.async_api")

    def run(html, fn):
        async def main():
            async with pw.async_playwright() as p:
                try:
                    browser = await p.chromium.launch()
                except Exception as e:
                    pytest.skip(f"Chromium not available: {e}")
                page = await browser.new_page(viewport={"width": 1000, "height": 700})
                await page.set_content(html)
                try:
                    return await fn(page)
                finally:
                    await browser.close()
        return asyncio.run(main())
    return run
