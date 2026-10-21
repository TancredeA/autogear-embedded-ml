"""Download the six public source files (pinned commits, SHA-256 checked) into data/raw/."""
from autogear.data.download import fetch_all
from autogear.data.sources import ALL_SOURCES
from autogear.paths import RAW, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    paths = fetch_all(RAW)
    for s in ALL_SOURCES:
        print(f"{s.key:20s} {s.licence:42s} {paths[s.key].relative_to(RAW.parent.parent)}")
