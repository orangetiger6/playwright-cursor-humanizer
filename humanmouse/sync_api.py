"""HumanMouse for `sync_playwright` pages.

    from playwright.sync_api import sync_playwright
    from humanmouse.sync_api import HumanMouse

    with sync_playwright() as p:
        page = p.chromium.launch(headless=False).new_page()
        page.goto("https://example.com")
        human = HumanMouse(page)
        human.click("text=More information")

This runs the async implementation without an event loop: the sync page is wrapped so its
calls look awaitable, sleeps block, and so each coroutine finishes without ever suspending.
"""
import functools
import time

from .playwright_driver import HumanMouse as AsyncHumanMouse

# Playwright methods that are synchronous in the async API too (they build a locator, no I/O).
_SYNC_METHODS = {"locator"}


def _is_playwright(obj):
    return type(obj).__module__.startswith("playwright.")


def _wrap(obj):
    return _Awaitable(obj) if _is_playwright(obj) else obj


def _unwrap(obj):
    return obj._obj if isinstance(obj, _Awaitable) else obj


class _Awaitable:
    """Presents a sync Playwright object (page, locator, mouse...) with the async API's shape."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if not callable(attr):
            return _wrap(attr)
        if name in _SYNC_METHODS:
            return lambda *a, **kw: _wrap(attr(*[_unwrap(x) for x in a], **kw))

        async def call(*a, **kw):
            return _wrap(attr(*[_unwrap(x) for x in a], **{k: _unwrap(v) for k, v in kw.items()}))
        return call


async def _blocking_sleep(seconds):
    if seconds > 0:
        time.sleep(seconds)


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise RuntimeError("humanmouse.sync_api: the async implementation suspended unexpectedly")


class HumanMouse:
    """Same methods and arguments as `humanmouse.HumanMouse`, without `await`."""

    def __init__(self, page, *args, **kwargs):
        self._impl = AsyncHumanMouse(_Awaitable(page), *args, **kwargs)
        self._impl._sleep = _blocking_sleep

    @property
    def page(self):
        return self._impl.page._obj

    def __getattr__(self, name):  # position, gen, behavior, temperature...
        return getattr(self._impl, name)

    def __setattr__(self, name, value):
        if name == "_impl":
            object.__setattr__(self, name, value)
        else:
            setattr(self._impl, name, value)


def _sync_method(name):
    method = getattr(AsyncHumanMouse, name)

    @functools.wraps(method)
    def call(self, *args, **kwargs):
        args = [_wrap(a) for a in args]
        kwargs = {k: _wrap(v) for k, v in kwargs.items()}
        return _run(method(self._impl, *args, **kwargs))
    return call


for _name in ["move_to", "click", "drag", "point_at", "set_slider", "scroll_by", "scroll_to"]:
    setattr(HumanMouse, _name, _sync_method(_name))
