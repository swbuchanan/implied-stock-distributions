"""Minimal pytest stand-in, so the suite can be executed without pytest.

Only used in environments where pytest cannot be installed. The test file
itself is plain pytest and runs unchanged under the real thing.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import sys
import tempfile
import traceback
import warnings
from pathlib import Path


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, actual, expected):
        rel = self.rel if self.rel is not None else (0.0 if self.abs is not None else 1e-6)
        tol = max(
            self.abs if self.abs is not None else 0.0,
            rel * abs(expected) if expected == expected else 0.0,
        )
        if tol == 0.0:
            tol = 1e-12
        return abs(actual - expected) <= tol

    def __eq__(self, other):
        try:
            import numpy as np
            if np.ndim(self.expected) or np.ndim(other):
                a, e = np.asarray(other, float), np.asarray(self.expected, float)
                return a.shape == e.shape and all(
                    self._close(x, y) for x, y in zip(a.ravel(), e.ravel())
                )
        except Exception:
            pass
        return self._close(float(other), float(self.expected))

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


class _Pytest:
    @staticmethod
    def approx(expected, rel=None, abs=None):
        return _Approx(expected, rel, abs)

    @staticmethod
    @contextlib.contextmanager
    def warns(category, match=None):
        import re
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield caught
        hits = [w for w in caught if issubclass(w.category, category)]
        assert hits, f"no {category.__name__} raised"
        if match:
            assert any(re.search(match, str(w.message)) for w in hits), (
                f"no {category.__name__} matching {match!r}; "
                f"got {[str(w.message) for w in hits]}"
            )

    @staticmethod
    @contextlib.contextmanager
    def raises(category, match=None):
        import re
        try:
            yield
        except category as exc:
            if match:
                assert re.search(match, str(exc)), f"{exc!r} does not match {match!r}"
            return
        raise AssertionError(f"{category.__name__} not raised")

    class mark:  # noqa: N801
        @staticmethod
        def parametrize(*args, **kwargs):
            def wrap(fn):
                return fn
            return wrap


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None):
        if value is None:
            raise NotImplementedError("string form of setattr is not supported")
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


def run(module_path: str) -> int:
    sys.modules["pytest"] = _Pytest  # type: ignore[assignment]

    import importlib.util
    spec = importlib.util.spec_from_file_location("suite_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tests = [
        (name, obj) for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    ]

    passed, failed = 0, []
    for name, fn in tests:
        parameters = inspect.signature(fn).parameters
        kwargs, patch, tmp = {}, None, None

        if "monkeypatch" in parameters:
            patch = _MonkeyPatch()
            kwargs["monkeypatch"] = patch
        if "tmp_path" in parameters:
            tmp = tempfile.TemporaryDirectory()
            kwargs["tmp_path"] = Path(tmp.name)

        try:
            fn(**kwargs)
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
        finally:
            if patch:
                patch.undo()
            if tmp:
                tmp.cleanup()

    print()
    print(f"{passed} passed, {len(failed)} failed, {len(tests)} total")
    for name, tb in failed:
        print()
        print("=" * 70)
        print(name)
        print("=" * 70)
        print(tb)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))
