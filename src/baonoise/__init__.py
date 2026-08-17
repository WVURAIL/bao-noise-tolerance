"""baonoise: RFI-masking noise-tolerance forecasts for 21cm BAO detection,
built on RadioFisher (Bull, Ferreira, Patel & Santos 2015) and the pilot-proxy
ATSC DTV masking statistics.
"""
from . import (api, channels, compat, constants, cosmologies, fisherbank,
               forecast, incumbent, layout, pkcache, products, residual,
               resources, scenarios, survey)

__all__ = ["api", "channels", "compat", "constants", "cosmologies",
           "fisherbank", "forecast", "incumbent", "layout", "pkcache",
           "products", "residual", "resources", "scenarios", "survey"]
__version__ = "1.0.0"
