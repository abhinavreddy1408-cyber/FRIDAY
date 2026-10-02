import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from friday import config
config.config = config

import datetime
_original_now = datetime.datetime.now
class PatchedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return _original_now(datetime.timezone.utc).replace(tzinfo=None)
sys.modules['datetime'].datetime = PatchedDatetime



