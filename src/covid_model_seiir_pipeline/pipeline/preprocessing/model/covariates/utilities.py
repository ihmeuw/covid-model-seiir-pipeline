from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import pandas as pd

# Alias for covariate prep return types.
Covariate = Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
CovariateGroup = Dict[str, Covariate]
