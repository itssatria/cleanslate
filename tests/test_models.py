import pytest

from cleanslate.models import ModelDownloadError, ensure_lama_model, lama_is_downloaded


def _fake_model(path, size=2048):
    path.write_bytes(b"x" * size)
    return path.as_uri()


def test_downloads_when_missing(tmp_path):
    src = tmp_path / "src.pt"
    url = _fake_model(src)
    dest = tmp_path / "cache" / "big-lama.pt"
    calls = []
    out = ensure_lama_model(lambda f, m: calls.append(f), dest=dest, urls=[url], min_size=1000)
    assert out == dest and lama_is_downloaded(dest, 1000)
    assert not (tmp_path / "cache" / "big-lama.pt.part").exists()
    assert calls


def test_skips_when_present(tmp_path):
    dest = tmp_path / "big-lama.pt"
    dest.write_bytes(b"x" * 5000)
    # URL is invalid on purpose: it must not be touched
    assert ensure_lama_model(dest=dest, urls=["file:///does/not/exist"], min_size=1000) == dest


def test_redownloads_truncated_file(tmp_path):
    url = _fake_model(tmp_path / "src.pt", 4000)
    dest = tmp_path / "big-lama.pt"
    dest.write_bytes(b"x" * 10)  # broken earlier download
    ensure_lama_model(dest=dest, urls=[url], min_size=1000)
    assert dest.stat().st_size == 4000


def test_falls_back_to_next_url_then_fails(tmp_path):
    good = _fake_model(tmp_path / "src.pt")
    dest = tmp_path / "a" / "big-lama.pt"
    ensure_lama_model(dest=dest, urls=["file:///nope", good], min_size=1000)
    assert dest.exists()
    with pytest.raises(ModelDownloadError):
        ensure_lama_model(dest=tmp_path / "b" / "big-lama.pt", urls=["file:///nope"], min_size=1000)


def test_find_lama_uses_env_path(tmp_path, monkeypatch):
    from cleanslate.models import find_lama

    f = tmp_path / "custom.pt"
    f.write_bytes(b"x" * 5000)
    monkeypatch.setenv("LAMA_MODEL", str(f))
    assert find_lama(min_size=1000) == f
