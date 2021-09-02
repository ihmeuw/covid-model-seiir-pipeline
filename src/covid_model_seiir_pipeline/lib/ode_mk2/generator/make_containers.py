import itertools
import textwrap
from pathlib import Path
from typing import List

from covid_model_seiir_pipeline.lib.ode_mk2.generator import (
    utils,
)
from covid_model_seiir_pipeline.lib.ode_mk2.generator.make_constants import (
    TAB,
    PRIMITIVES,
    SPECS,
)


def make_doctring() -> str:
    description = "Static definitions for data containers."
    return utils.make_module_docstring(description, __file__)


def make_imports() -> str:
    out = utils.make_import('dataclasses', ['dataclass'])
    out += utils.make_import('typing', ['Dict', 'List', 'Iterator', 'Tuple']) + '\n'

    out += utils.make_import('pandas as pd') + '\n'

    out += utils.make_import('covid_model_seiir_pipeline.lib', ['utilities'])
    return out + '\n'


def make_fields(field_specs: List[List[str]], prefix=None) -> str:
    out = ""
    for field_spec in field_specs:
        field_vars = [PRIMITIVES[field_var_name] for field_var_name in field_spec if field_var_name != 'all']
        for elements in itertools.product(*field_vars):
            elements = elements if prefix is None else [prefix] + list(elements)
            out += f"{TAB}{'_'.join(elements)}: pd.Series\n"
    return out


def make_ode_parameters() -> str:
    out = ""
    out += "@dataclass(repr=False, eq=False)\n"
    out += "class Parameters:\n"
    out += make_fields(SPECS['PARAMETERS']) + "\n"

    vaccine_field_spec = [s + ['risk_group'] for s in SPECS['VACCINE_TYPES']]
    out += make_fields(vaccine_field_spec, 'vaccinations')

    out += """
    def to_dict(self) -> Dict[str, pd.Series]:
        return {k: v.rename(k) for k, v in utilities.asdict(self).items()}

    def to_df(self) -> pd.DataFrame:
        return pd.concat(self.to_dict().values(), axis=1)

    def reindex(self, index: pd.Index) -> 'Parameters':
        # noinspection PyArgumentList
        return type(self)(
            **{key: value.reindex(index) for key, value in self.to_dict().items()}
        )
    """
    return out


def make_containers() -> str:
    out = make_doctring()
    out += make_imports() + "\n"
    out += make_ode_parameters()
    return out


if __name__ == '__main__':
    here = Path(__file__).resolve()
    containers_file = here.parent.parent / 'containers.py'
    print(f'Generating {str(containers_file)} from {str(here)}')
    constants_string = make_containers()
    with containers_file.open('w') as f:
        f.write(constants_string)
