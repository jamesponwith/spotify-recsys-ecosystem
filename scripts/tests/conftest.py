"""Put scripts/ on the import path.

pytest inserts the directory holding the test file, which is scripts/tests, not
scripts/. There is no package here to import through and adding one would make
`scripts/` an installable thing it is not -- these are repo tools, run by path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
