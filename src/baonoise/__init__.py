"""baonoise: RFI-masking noise-tolerance forecasts for 21cm BAO detection,
built on RadioFisher (Bull, Ferreira, Patel & Santos 2015) and the pilot-proxy
ATSC DTV masking statistics.
"""
from . import (channels, compat, fisherbank, forecast, incumbent, layout,
               pkcache, residual, resources, scenarios, survey)

__all__ = ["channels", "compat", "fisherbank", "forecast", "incumbent",
           "layout", "pkcache", "residual", "resources", "scenarios",
           "survey"]
__version__ = "0.1.0"
