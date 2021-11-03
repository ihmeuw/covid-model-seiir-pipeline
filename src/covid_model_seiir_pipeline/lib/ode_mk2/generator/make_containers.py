from pathlib import Path

from covid_model_seiir_pipeline.lib.ode_mk2.generator import (
    utils,
)
from covid_model_seiir_pipeline.lib.ode_mk2.generator.make_constants import (
    TAB,
    PRIMITIVE_TYPES,
    DERIVED_TYPES,
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
    out += utils.make_import('covid_model_seiir_pipeline.lib.ode_mk2.constants',
                             ['PARAMETERS_NAMES', 'ETA_NAMES'])
    return out + '\n'


def make_fields(group_spec: Spec, prefix: str = '', suffix: str = '') -> str:
    out = ""
    _, field_names = unpack_spec_fields(group_spec)
    for field_name in field_names:        
        out += f"{TAB}{prefix}{field_name}{suffix}: pd.Series\n"
    return out


def make_ode_parameters() -> str:
    out = ""
    out += "@dataclass(repr=False, eq=False)\n"
    out += "class Parameters:\n"
    out += make_fields(SPECS['PARAMETERS']) + "\n"

    for risk_group in PRIMITIVE_TYPES['risk_group']:
        for vaccination_status in ['vaccinations', 'boosters']:
            out += f"{TAB}{vaccination_status}_{risk_group}: pd.Series\n"
    out += '\n'

    for risk_group in PRIMITIVE_TYPES['risk_group']:
        for vaccine_status in DERIVED_TYPES['vaccine_status'][1]:
            for variant in DERIVED_TYPES['variant'][1]:
                out += f"{TAB}eta_{vaccine_status}_{variant}_{risk_group}: pd.Series\n"
    out += '\n'

    out += f"{TAB}natural_waning_distribution: pd.Series\n"
    out += f"{TAB}phi: pd.DataFrame\n"

    out += """
    def get_params(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() if k in PARAMETERS_NAMES], axis=1)
        
    def get_vaccinations(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if 'vaccinations' in k or 'boosters' in k], axis=1)    

    def get_etas(self) -> pd.DataFrame:
        return pd.concat([v.rename(k) for k, v in utilities.asdict(self).items() 
                          if '_'.join(k.split('_')[1:-1]) in ETA_NAMES], axis=1)
        
        
    def to_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        return self.get_params(), self.get_vaccinations(), self.get_etas(), self.natural_waning_distribution, self.phi
        
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
