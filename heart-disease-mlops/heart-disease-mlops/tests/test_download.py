import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_looks_valid_accepts_expected_csv():
    text = dl.EXPECTED_HEADER + "\n" + "\n".join(["63,1,3,145,233,1,0,150,0,2.3,0,0,1,1"] * 260)
    assert dl.looks_valid("\ufeff" + text)


def test_looks_valid_rejects_html_error_page_and_short_files():
    assert not dl.looks_valid("<html>404</html>")
    assert not dl.looks_valid(dl.EXPECTED_HEADER + "\n1,2,3")


def test_download_uses_next_mirror_on_failure(tmp_path, monkeypatch):
    good = dl.EXPECTED_HEADER + "\n" + "\n".join(["63,1,3,145,233,1,0,150,0,2.3,0,0,1,1"] * 260)

    class Resp:
        def __init__(self, body): self.body = body
        def read(self): return self.body.encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(url, timeout=30):
        if "bad" in url:
            raise OSError("boom")
        return Resp(good)

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    out = dl.download(tmp_path / "heart.csv", ["https://bad.example/x", "https://good.example/y"])
    assert out.read_text().startswith("age,sex")
