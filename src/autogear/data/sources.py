"""Public sources of the dataset, pinned to a commit and a SHA-256 checksum.

None of these files is redistributed with the repository: `scripts/01_download_sources.py` fetches
them from their original location. Each source keeps its own licence (see `data/README.md`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str          # identifier of the ride in the dataset
    rider: str        # identifier of the cyclist
    repo: str         # GitHub repository
    commit: str       # pinned commit
    path: str         # path of the file inside the repository
    sha256: str
    licence: str
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.path}"

    @property
    def filename(self) -> str:
        return f"{self.key}__{self.path.rsplit('/', 1)[-1]}"


INRIA = Source(
    key="inria", rider="inria", repo="INRIA/scikit-learn-mooc",
    commit="1b294d95296e7e52dec5b9c69076f8010b2bc51f", path="datasets/bike_rides.csv",
    sha256="87ce62c8c7ff61d73bd4abeda824cacc6150b85a48e23979576681a3469225ca",
    licence="CC-BY 4.0 (scikit-learn MOOC)",
    note="four rides by one cyclist, 1 Hz, cadence, speed, slope, acceleration, power")

FIT_SOURCES = (
    Source("elemnt_2019", "elemnt_2019", "dtcooper/python-fitparse",
           "d88bb6997138c38d0c578aeaaf4108df0711d1f6",
           "tests/files/2019-02-17-062644-ELEMNT-297E-195-0.fit",
           "3f73dc39741c63f3883e2db9f80447f421a3845e7313d900a8ec66629b49a656", "MIT"),
    Source("edge530_2020", "edge530_2020", "tormoder/fit",
           "e0d27005ab71b74f34a64f9fab836d375fac875b", "testdata/bpg/garmin.fit",
           "dc352ab9e082ecc10038416e69d44534cc0b3e3d4cd545f4e37429b510df2d73", "MIT"),
    Source("edge810_2013", "edge810_2013", "dtcooper/python-fitparse",
           "d88bb6997138c38d0c578aeaaf4108df0711d1f6",
           "tests/files/Edge810-Vector-2013-08-16-15-35-10.fit",
           "e278d20809b1fda62f4b60dfe291a4b16f00933ad37e923a99e5a59bb48f6eab", "MIT"),
    Source("edge1030_2020", "edge1030_2020", "muktihari/fit",
           "5aae3d5d2b0e464279c6ec6ba3e00d754e7d5676",
           "testdata/from_official_sdk/activity_lowbattery.fit",
           "f0809b5cf5edc511b5002485759b20edd0577788d8ce4098c5fba6aea2775776",
           "BSD-3-Clause (file from the Garmin FIT SDK)"),
    Source("di2_edge1040_2022", "di2_edge1040_2022", "garmin/fit-python-sdk",
           "7e4b3132d50de66f5adfcd6c0addeaade41536d4", "tests/fits/WithGearChangeData.fit",
           "7220d6a86e0bafcb1884070081757a2a65caa83d46afccc113ebbe17df60fa79",
           "Garmin FIT Protocol License",
           note="carries real Shimano Di2 rear_gear_change events: the validation ride"),
)

ALL_SOURCES = (INRIA, *FIT_SOURCES)
DI2_RIDE = "di2_edge1040_2022"
