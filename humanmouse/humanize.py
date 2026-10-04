"""Make a page's own Playwright calls move like a person, without rewriting the script.

    from humanmouse import humanize

    humanize(page)              # or humanize(context): all of its pages, including ones opened later
    await page.click("#buy")    # now a generated path, then press and release
    await page.locator(".card").drag_to(page.locator("#done"))

Routed through HumanMouse: click, dblclick, hover and drag_and_drop on pages, click, dblclick,
hover and drag_to on locators, and page.mouse.move / page.mouse.wheel (vertical). Calls with
options HumanMouse can't honor (position, modifiers, force, trial, delay) and pages that weren't
humanized keep Playwright's own behavior. Works with async and sync Playwright.
"""
from .generate import PathGenerator
from .playwright_driver import HumanMouse
from .sync_api import HumanMouse as SyncHumanMouse

NATIVE_ONLY = ("position", "source_position", "target_position", "modifiers", "force", "trial", "delay")


class _State:
    def __init__(self, human):
        self.human = human
        self.busy = False  # HumanMouse's own page.mouse calls must reach Playwright


def _plain(kw):
    return not any(kw.get(k) for k in NATIVE_ONLY)


# Each handler gets the Playwright call's arguments and returns `human -> result`, or None to
# leave the call to Playwright.
def _click(count):
    def handler(target, kw):
        if _plain(kw):
            return lambda h: h.click(target, button=kw.get("button") or "left", click_count=kw.get("click_count") or count)
    return handler


def _hover(target, kw):
    if _plain(kw):
        return lambda h: h.hover(target)


def _drag(source, target, kw):
    if _plain(kw):
        return lambda h: h.drag(source, target)


def _move(x, y, kw):
    return lambda h: h.move_to(x, y)


def _wheel(dx, dy, kw):
    if not dx:
        return lambda h: h.scroll_by(dy)


# class -> {method: handler}. Handlers get the call's positional arguments, then its kwargs; for
# Locator methods the locator itself comes first, as the target.
PATCHES = {
    "Page": {"click": _click(1), "dblclick": _click(2), "hover": _hover, "drag_and_drop": _drag},
    "Locator": {"click": _click(1), "dblclick": _click(2), "hover": _hover, "drag_to": _drag},
    "Mouse": {"move": _move, "wheel": _wheel},
}


def _owner(obj, cls):
    return obj.page if cls == "Locator" else obj  # the object humanize() marked: the page, or page.mouse


def _wrap(cls, native, handler, is_async):
    takes_self = cls == "Locator"

    def plan(self, args, kwargs):
        state = getattr(_owner(self, cls), "_humanmouse", None)
        if state is None or state.busy:
            return None, None
        call = handler(self, *args, kwargs) if takes_self else handler(*args, kwargs)
        return state, call

    if is_async:
        async def method(self, *args, **kwargs):
            state, call = plan(self, args, kwargs)
            if call is None:
                return await native(self, *args, **kwargs)
            state.busy = True
            try:
                return await call(state.human)
            finally:
                state.busy = False
    else:
        def method(self, *args, **kwargs):
            state, call = plan(self, args, kwargs)
            if call is None:
                return native(self, *args, **kwargs)
            state.busy = True
            try:
                return call(state.human)
            finally:
                state.busy = False
    method.__name__, method.__doc__, method._humanmouse_native = native.__name__, native.__doc__, native
    return method


def _install():
    import playwright.async_api as async_api
    import playwright.sync_api as sync_api

    for module, is_async in ((async_api, True), (sync_api, False)):
        for cls_name, methods in PATCHES.items():
            cls = getattr(module, cls_name)
            for name, handler in methods.items():
                native = getattr(cls, name)
                if not hasattr(native, "_humanmouse_native"):
                    setattr(cls, name, _wrap(cls_name, native, handler, is_async))


def humanize(target, **kwargs):
    """Route a page's (or every page of a browser context's) mouse actions through HumanMouse.

    `kwargs` go to HumanMouse (generator, behavior, position, temperature...). Returns the page's
    HumanMouse, for actions Playwright has no method for (set_slider, scroll_to...); for a context,
    returns None. Calling it again on the same page returns the same HumanMouse."""
    _install()
    if hasattr(target, "pages"):  # a BrowserContext
        kwargs.setdefault("generator", PathGenerator())  # load the model once for all its pages
        for page in target.pages:
            humanize(page, **kwargs)
        target.on("page", lambda page: humanize(page, **kwargs))
        return None
    state = getattr(target, "_humanmouse", None)
    if state is None:
        sync = type(target).__module__.startswith("playwright.sync_api")
        state = _State((SyncHumanMouse if sync else HumanMouse)(target, **kwargs))
        target._humanmouse = target.mouse._humanmouse = state
    return state.human
