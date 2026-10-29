import functools
import http.server
import json
import threading

import numcodecs
import numpy as np
import pytest

from windcon.zarr_io import ZarrArray


def _write_zarr(root, data, chunks, compressor=None, drop=()):
    root.mkdir()
    meta = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(chunks),
        "dtype": data.dtype.str,
        "compressor": compressor.get_config() if compressor else None,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "dimension_separator": "/",
    }
    (root / ".zarray").write_text(json.dumps(meta))
    grid = [range(-(-s // c)) for s, c in zip(data.shape, chunks)]
    for idx in np.ndindex(*[len(g) for g in grid]):
        if idx in drop:
            continue
        block = np.zeros(chunks, dtype=data.dtype)  # zarr v2 edge chunks are stored full size
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, data.shape))
        part = data[sl]
        block[tuple(slice(0, n) for n in part.shape)] = part
        raw = block.tobytes()
        path = root.joinpath(*map(str, idx))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressor.encode(raw) if compressor else raw)


@pytest.fixture
def served(tmp_path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{server.server_address[1]}", server
    server.shutdown()


@pytest.mark.parametrize("compressor", [None, numcodecs.Blosc(cname="zstd", clevel=1)])
def test_read_spans_chunks_edges_and_missing_chunks(served, compressor):
    root, url, server = served
    data = np.random.default_rng(1).integers(1, 255, size=(10, 13, 11), dtype=np.uint8)
    _write_zarr(root / "vol", data, (4, 5, 6), compressor, drop={(1, 0, 0)})
    vol = ZarrArray(f"{url}/vol", cache_dir=root / "cache")
    expected = data.copy()
    expected[4:8, 0:5, 0:6] = 0  # the dropped chunk reads as fill_value
    box = ((-2, 10), (3, 15), (1, 11))  # starts before and ends after the array
    out = vol.read(box)
    want = np.zeros((12, 12, 10), dtype=np.uint8)
    want[2:12, 0:10, 0:10] = expected[0:10, 3:13, 1:11]
    np.testing.assert_array_equal(out, want)

    server.shutdown()  # a repeated read is served entirely from the chunk cache
    np.testing.assert_array_equal(ZarrArray(f"{url}/vol", cache_dir=root / "cache").read(box), want)
