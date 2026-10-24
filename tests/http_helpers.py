"""Fake `requests` responses and a URL router for mocking HTTP in tests."""

import json

import requests


class FakeResponse:
    def __init__(self, status=200, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self._text = text

    def json(self):
        if self._text is not None:
            return json.loads(self._text)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class Router:
    """Maps URL paths (without host or query) to responses; records calls."""

    def __init__(self, base, routes):
        self.base, self.routes, self.calls = base, routes, []

    def __call__(self, url, params=None, timeout=None, **_):
        path = url[len(self.base):].split("?")[0] if url.startswith(self.base) else url
        self.calls.append(path)
        route = self.routes.get(path)
        if isinstance(route, Exception):
            raise route
        if route is None:
            return FakeResponse(404, None)
        return route if isinstance(route, FakeResponse) else FakeResponse(200, route)


def _route(monkeypatch, module, base, routes):
    router = Router(base, routes)
    monkeypatch.setattr(module.requests, "get", router)
    return router
