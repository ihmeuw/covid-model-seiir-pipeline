from .__about__ import *

import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='limetr')
warnings.filterwarnings('ignore', category=FutureWarning, module='cyipopt')
warnings.filterwarnings('ignore', category=FutureWarning, module='ipopt')
warnings.filterwarnings('ignore', category=UserWarning, module='mrtool')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

del warnings
