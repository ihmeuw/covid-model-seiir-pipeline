from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import (
    OOS_HOLDOUT_JOBS,
    OOSHoldoutSpecification,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.data import (
    OOSHoldoutDataInterface,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.task import (
    oos_holdout_regression,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.main import (
    do_oos_holdout,
    oos_holdout,
)

SPECIFICATION = OOSHoldoutSpecification
COMMAND = oos_holdout
APPLICATION_MAIN = do_oos_holdout
TASKS = {
    OOS_HOLDOUT_JOBS.oos_regression: oos_holdout_regression,
}
