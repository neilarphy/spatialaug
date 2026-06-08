"""Dataset adapters used by the spatialaug benchmark.

Convention for adding a new loader
----------------------------------
``fns.py`` is the reference implementation. A new dataset adapter
should keep the same shape so it composes with the rest of the
benchmark out of the box:

- A function named ``load_<source>(<unit>, target, *, feature_cols=None, ...)``
  that returns a ``pd.DataFrame`` with at least
  (lat_col, lon_col, target_col) columns.
- Optional companion constants exposed at module level — e.g.
  ``<SOURCE>_DEFAULT_TARGETS``, ``<SOURCE>_DEFAULT_FEATURES``,
  ``<SOURCE>_BUSINESS_TARGETS`` — so notebooks can discover sensible
  defaults without hard-coding strings.
- An ``available_<unit>s()`` helper (e.g. ``available_cities()``)
  listing the units present in the data.
- Re-export both the loader and the constants from this
  ``__init__.py`` under predictable, source-prefixed names so the
  namespace stays unambiguous when several datasets are present.

"""

from spatialaug.benchmark.datasets.fns import (
    ALL_AVAILABLE_TARGETS as FNS_ALL_AVAILABLE_TARGETS,
)
from spatialaug.benchmark.datasets.fns import (
    BUSINESS_TARGETS as FNS_BUSINESS_TARGETS,
)
from spatialaug.benchmark.datasets.fns import (
    DEFAULT_FEATURES as FNS_DEFAULT_FEATURES,
)
from spatialaug.benchmark.datasets.fns import (
    DEFAULT_TARGETS as FNS_DEFAULT_TARGETS,
)
from spatialaug.benchmark.datasets.fns import (
    available_cities as fns_available_cities,
)
from spatialaug.benchmark.datasets.fns import (
    load_fns_city,
)

__all__ = [
    "load_fns_city",
    "fns_available_cities",
    "FNS_DEFAULT_TARGETS",       
    "FNS_BUSINESS_TARGETS",      
    "FNS_ALL_AVAILABLE_TARGETS", 
    "FNS_DEFAULT_FEATURES",
]
