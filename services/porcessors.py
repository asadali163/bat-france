# Re-export shim — preserves backward compatibility.
# All functions now live in domain-specific modules.
from services.processors_base import *  # noqa: F401, F403
from services.processors_rain import *  # noqa: F401, F403
from services.processors_temp import *  # noqa: F401, F403
from services.processors_wind import *  # noqa: F401, F403
from services.processors_sky import *   # noqa: F401, F403
from services.processors_prophet import *  # noqa: F401, F403

# Re-export the alias that was defined in the original file
from services.processors_wind import aggregate_ols_fl  # noqa: F401
