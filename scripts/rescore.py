"""Re-score existing results with the fixed scorer."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def score_output(output: str, answer: str) -> bool:
    lo = output.lower()
    parts = [p.strip().lower() for p in answer.split("...") if p.strip()]
    return all(p in lo for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.path.open()]
    by_mode = {}
    for r in rows:
        for mode in ("baseline", "reattn", "full"):
            if mode in r:
                ok = score_output(r[mode]["output"], r["answer"])
                by_mode.setdefault(mode, []).append((r["id"], ok, r["answer"], r[mode]["output"]))
    for m, items in by_mode.items():
        print(f"\n== {m}: {sum(ok for _,ok,_,_ in items)}/{len(items)} ==")
        for cid, ok, ans, out in items:
            mark = "OK" if ok else " X"
            print(f"  [{mark}] case={cid}  ans={ans!r}  out={out!r}")


if __name__ == "__main__":
    main()
