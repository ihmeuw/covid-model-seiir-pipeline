import itertools
from pathlib import Path
from typing import List

from covid_model_seiir_pipeline.lib.ode_mk2.generator import (
    utils,
)
from covid_model_seiir_pipeline.lib.ode_mk2.generator.make_constants import (
    TAB,
    PRIMITIVE_TYPES,
    Spec,
    SPECS,
    unpack_spec_fields,
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


def make_fields(group_spec: Spec, prefix: str = '', suffix: str = '') -> str:
    out = ""
    _, field_names = unpack_spec_fields(group_spec)
    for field_name in field_names:
        if field_name[-4:] == '_all':
            field_name = field_name[:-4]
        out += f"{TAB}{prefix}{field_name}{suffix}: pd.Series\n"
    return out


def make_ode_parameters() -> str:
    out = ""
    out += "@dataclass(repr=False, eq=False)\n"
    out += "class Parameters:\n"
    out += make_fields(SPECS['PARAMETERS']) + "\n"

    for risk_group in PRIMITIVE_TYPES['risk_group']:
        out += make_fields(SPECS['VACCINATIONS'], 'vaccinations_', f'_{risk_group}')

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
