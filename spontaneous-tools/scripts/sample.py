"""Draw a small pinned sample. Source rows are losslessly preserved in results/sample.jsonl."""
import bisect
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DOLCI = ("allenai/Dolci-Instruct-SFT-Tool-Use", "dc042846f0f2de0f15eedae3d6ced04223ed47eb")
NVIDIA = ("nvidia/Nemotron-Agentic-v1", "650d590978ca35c8f1ecea2faf136e5fac421b62")
SEED = 182
SPLITS = {"interactive_agent": 448570455, "tool_calling": 5338348607}


class RemoteFile(io.RawIOBase):
    """Seekable HTTP range reader, checking the response before consuming its body."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                           status_forcelist=[429, 500, 502, 503, 504])))

    def close(self):
        self.session.close()
        super().close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [0, self.pos, self.size][whence] + offset
        return self.pos

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos + size)
        if end <= self.pos:
            return b""
        span = f"{self.pos}-{end - 1}"
        with self.session.get(self.url, headers={"Range": f"bytes={span}", "Accept-Encoding": "identity"},
                              timeout=90, stream=True) as r:
            r.raise_for_status()
            if r.headers.get("Content-Range") != f"bytes {span}/{self.size}":
                raise IOError(f"unexpected HTTP range for {span}")
            block = r.content
        if len(block) != end - self.pos:
            raise IOError("incomplete range")
        self.pos = end
        return block


def url(dataset, path):
    repo, rev = dataset
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def record(key, dataset, path, row, **location):
    return dict(key=key, repo=dataset[0], revision=dataset[1], path=path, **location, raw=row)


def viewer_rows(offset, revision):
    # The rows endpoint tracks main. Use it only if the serving revision matches
    # the frozen Parquet snapshot and the service reports a complete cell.
    try:
        response = requests.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": DOLCI[0], "config": "default", "split": "train",
            "offset": offset, "length": 1}, timeout=15)
        if response.ok and response.headers.get("X-Revision") == revision:
            row = response.json()["rows"][0]
            if row["row_idx"] == offset and row["truncated_cells"] == []:
                return row["row"]
    except (requests.RequestException, KeyError, ValueError, IndexError):
        pass
    return None


def dolci(item):
    shard, entry = item
    path = entry["path"]
    with RemoteFile(url(DOLCI, path), entry["size"]) as handle:
        parquet = pq.ParquetFile(handle)
        meta = parquet.metadata
        ends = []
        total = 0
        for g in range(meta.num_row_groups):
            total += meta.row_group(g).num_rows
            ends.append(total)
        indices = sorted(random.Random(SEED + shard).sample(range(total), 3))
        records = []
        for i in indices:
            # The six pinned shards contain 37,930 rows each, except the final
            # shard which has 37,929. Check before using the global viewer index.
            assert total == (37929 if shard == 5 else 37930)
            row = viewer_rows(shard * 37930 + i, DOLCI[1])
            if row is not None:
                records.append(record(f"D{shard}-{i}", DOLCI, path, row, row_in_shard=i))
                continue
            g = bisect.bisect_right(ends, i)
            local = i - (ends[g - 1] if g else 0)
            row = parquet.read_row_group(g).slice(local, 1).to_pylist()[0]
            records.append(record(f"D{shard}-{i}", DOLCI, path, row, row_in_shard=i))
    return records


def window_rows(reader, start, stop):
    # Include a row iff its FIRST byte is in [start, stop). Finish the last row beyond stop.
    first = max(0, start - 1)
    reader.seek(first)
    buf = reader.read(stop - first + 65536)
    if start:
        newline = buf.find(b"\n")
        if newline < 0:
            return
        pos = newline + 1
    else:
        pos = 0
    while first + pos < stop:
        end = buf.find(b"\n", pos)
        while end < 0 and reader.tell() < reader.size:
            buf += reader.read(65536)
            end = buf.find(b"\n", pos)
        if end < 0:
            end = len(buf)
        yield first + pos, json.loads(buf[pos:end])
        pos = end + 1


def nemotron(split):
    size = SPLITS[split]
    path = "data/" + split + ".jsonl"
    # Eight equal-width byte strata. Uniform circular window within each stratum gives
    # every row start nearly the same inclusion probability regardless of row length.
    rng = random.Random(f"{SEED}:{split}")
    records = []
    with RemoteFile(url(NVIDIA, path), size) as reader:
        for h in range(8):
            lo, hi = size * h // 8, size * (h + 1) // 8
            start = rng.randrange(lo, hi)
            stop = start + 32768
            spans = [(start, min(stop, hi))]
            if stop > hi:
                spans.append((lo, lo + stop - hi))
            for left, right in spans:
                for offset, row in window_rows(reader, left, right):
                    key = f"{split[0].upper()}-{offset}"
                    records.append(record(key, NVIDIA, path, row, byte_offset=offset,
                                          window=h, window_start=start))
    return records


def main():
    repo, rev = DOLCI
    listing = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/data", timeout=30)
    listing.raise_for_status()
    shards = sorted(listing.json(), key=lambda item: item["path"])
    assert len(shards) == 6 and all(s["path"].endswith(".parquet") for s in shards)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(dolci, arg) for arg in enumerate(shards)]
        jobs += [pool.submit(nemotron, split) for split in SPLITS]
        rows = [row for job in jobs for row in job.result()]
    out = Path("results/sample.jsonl")
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"{len(rows)} rows in {out}")


if __name__ == "__main__":
    main()
