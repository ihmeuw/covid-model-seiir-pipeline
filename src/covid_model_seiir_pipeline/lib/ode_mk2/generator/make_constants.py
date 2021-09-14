from collections import namedtuple
import itertools
from pathlib import Path
from typing import List, Tuple
import textwrap

import inflection

from covid_model_seiir_pipeline.lib.ode_mk2.generator import utils

TAB = '    '  # type: str

PRIMITIVE_TYPES = {
    'compartment_type': [
        'S',
        'E',
        'I',
        'R',
        'N',
    ],
    'tracking_compartment_type': [
        'NewE',
        'NewR',
        'NewVaxImmune',
        'Waned',
    ],
    'parameter_type': [
        'alpha',
        'sigma',
        'gamma',
        'pi',
        'new_e',
        'beta',
        'kappa',
        'rho',
    ],
    'risk_group': [
        'lr',
        'hr',
    ],
    'index_1_type': [
        'all',
        'total',

        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',

        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ],
    'index_2_type': [
        'unvaccinated',
        'vaccinated',
        'newly_vaccinated',
    ],
    'agg_index_type': [
        'all',
        'total',

        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',

        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
        'escape_immune',
        'omega_immune',

        'unvaccinated',
        'vaccinated',
        'newly_vaccinated',

        'non_immune',
        'natural',
        'vaccine',
        'vaccine_eligible',
    ]
}

DERIVED_TYPES = {
    'base_compartment': ('compartment_type', [
        'S',
        'E',
        'I',
        'R',
    ]),
    'tracking_compartment': ('tracking_compartment_type', [
        'NewE',
        'NewR',
        'NewVaxImmune',
        'Waned',
    ]),
    'base_parameter': ('parameter_type', [
        'alpha',
        'sigma',
        'gamma',
        'pi',
        'new_e',
    ]),
    'variant_parameter': ('parameter_type', [
        'beta',
        'kappa',
        'rho',
    ]),
    'variant_group': ('index_1_type', [
        'all',
        'total'
    ]),
    'variant': ('index_1_type', [
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',
    ]),
    'susceptible_type': ('index_1_type', [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ]),
    'protection_status': ('index_1_type', [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
    ]),
    'immune_status': ('index_1_type', [
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ]),
    'vaccine_type': ('index_1_type', [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ]),
    'vaccination_status': ('index_2_type', [
        'unvaccinated',
        'vaccinated',
    ]),
    'removed_vaccination_status': ('index_2_type', [
        'unvaccinated',
        'vaccinated',
        'newly_vaccinated',
    ]),
    'agg_variant': ('agg_index_type', [
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',
    ]),
    'agg_waned': ('agg_index_type', [
        'natural',
        'vaccine',
    ]),
    'agg_immune_status': ('agg_index_type', [
        'non_immune',
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ]),
    'agg_vaccination_status': ('agg_index_type', [
        'unvaccinated',
        'vaccinated',
    ]),
    'agg_other': ('agg_index_type', [
        'all',
        'total',
        'non_immune',
        'unvaccinated',
        'vaccinated',
        'vaccine_eligible'
    ])
}

Spec = namedtuple('Spec', ['offset', 'axes_primitives', 'field_specs'])

SPECS = {
    'PARAMETERS': Spec(
        offset='',
        axes_primitives=['parameter_type', 'index_1_type'],
        field_specs=[
            ['base_parameter', 'all'],
            ['variant_parameter', 'variant'],
        ],
    ),
    'COMPARTMENTS': Spec(
        offset='',
        axes_primitives=['compartment_type', 'index_1_type', 'index_2_type'],
        field_specs=[
            ['S', 'susceptible_type', 'vaccination_status'],
            ['E', 'variant', 'vaccination_status'],
            ['I', 'variant', 'vaccination_status'],
            ['R', 'variant', 'removed_vaccination_status'],
        ]
    ),
    'TRACKING_COMPARTMENTS': Spec(
        offset='COMPARTMENTS',
        axes_primitives=['tracking_compartment_type', 'agg_index_type'],
        field_specs=[
            ['NewE', 'agg_vaccination_status'],
            ['NewE', 'agg_variant'],
            ['NewE', 'total'],
            ['NewVaxImmune', 'agg_immune_status'],
            ['NewVaxImmune', 'total'],
            ['NewR', 'agg_variant'],
            ['NewR', 'total'],
            ['Waned', 'natural'],
            ['Waned', 'vaccine'],
            ['Waned', 'agg_variant'],
            ['Waned', 'agg_immune_status'],
        ],
    ),
    'AGGREGATES': Spec(
        offset='',
        axes_primitives=['compartment_type', 'agg_index_type'],
        field_specs=[
            ['base_compartment', 'agg_variant'],
            ['N', 'agg_vaccination_status'],
            ['N', 'total'],
        ],
    ),
    'WANED': Spec(
        offset='',
        axes_primitives=['agg_index_type'],
        field_specs=[
            ['natural'],
            ['vaccine'],
        ],
    ),
}

# Susceptible statuses are rank ordered. Here we map susceptible statuses
# onto which variants they're susceptible to.
SUSCEPTIBLE_BY_VARIANT = {
    'ancestral': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
    ],
    'alpha': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
    ],
    'beta': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
    ],
    'gamma': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
    ],
    'delta': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
    ],
    'other': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
    ],
    'omega': [
        'unprotected',
        'non_escape_protected',
        'escape_protected',
        'omega_protected',
        'non_escape_immune',
        'escape_immune',
    ],
}


def make_doctring() -> str:
    description = "Static definitions for the compartments and model parameters."
    return utils.make_module_docstring(description, __file__)


def make_imports() -> str:
    out = utils.make_import('collections', ['namedtuple'])
    out += utils.make_import('os') + '\n'

    out += utils.make_import('numpy as np')
    out += utils.make_import('numba.core', ['types'])
    out += utils.make_import('numba.typed', ['Dict'])
    return out + '\n'


def make_primitive_types() -> str:
    out = textwrap.dedent("""
    #######################
    # Primitive variables #
    #######################
           
    """)
    for primitive_group_name, variables in PRIMITIVE_TYPES.items():
        content = utils.make_content_array(rows=variables, columns=[''])
        out += utils.make_named_tuple(
            primitive_group_name,
            content,
        ) + '\n'

    for primitive_group_name in PRIMITIVE_TYPES:
        out += utils.make_index_map(primitive_group_name)
        out += utils.make_name_map(primitive_group_name)
    return out


def make_derived_types() -> str:
    out = textwrap.dedent("""
    #####################
    # Derived variables #
    #####################

    """)
    for derived_group_name, (primitive_type, variables) in DERIVED_TYPES.items():
        content = utils.make_content_array(rows=variables, columns=[''])
        out += utils.make_named_tuple(
            derived_group_name,
            content,
        )

        out += f'{inflection.underscore(derived_group_name).upper()} = _{inflection.camelize(derived_group_name)}(\n'
        for variable in variables:
            out += f'{TAB}{variable}={inflection.underscore(primitive_type).upper()}.{variable},\n'
        out += ')\n'
        out += utils.make_name_map(derived_group_name)
    return out

def unpack_spec_fields(spec: Spec) -> Tuple[List[str], List[str]]:
    field_keys = []
    field_names = []
    for field_spec in spec.field_specs:
        fields = {}
        for field_type, primitive in zip(field_spec, spec.axes_primitives):
            if field_type in DERIVED_TYPES:
                fields[field_type] = DERIVED_TYPES[field_type][1]
            else:
                fields[primitive] = [field_type]
        for elements in itertools.product(*fields.values()):
            field_keys.append(
                '['
                + ', '.join([f'{inflection.underscore(primitive).upper()}.{element}'
                             for primitive, element in zip(fields.keys(), elements)])
                + ']'
            )
            field_names.append('_'.join(elements))
    return field_keys, field_names


def make_specs() -> str:
    out = ''
    counts = {'': 0}
    for spec_name, spec in SPECS.items():
        out += f'{spec_name} = np.full(('
        out += ', '.join([f'len({inflection.underscore(axis_primitive).upper()})'
                          for axis_primitive in spec.axes_primitives])
        out += '), -1, dtype=np.int64)\n'

        count = counts[spec.offset]
        field_keys, field_names = unpack_spec_fields(spec)
        for field_key in field_keys:
            out += f'{spec_name}{field_key} = {count}\n'
            count += 1

        out += f"{spec_name}_NAMES = [\n"
        for name in field_names:
            out += f"{TAB}'{name}',\n"
        out += ']\n'
        out += '\n'
        counts[spec_name] = count
    return out


def make_compartment_group(compartment: str, variant: str, *levels: List[str]):
    out = f"COMPARTMENT_GROUPS[('{compartment}', '{variant}')] = np.array([\n"
    for elements in itertools.product(*levels):
        out += f"{TAB}COMPARTMENTS[('{compartment}', "
        for element in elements:
            out += f"'{element}', "
        out = out[:-2] + ")],\n"
    out += "], dtype=np.int64)\n"
    return out


def make_susceptible_compartment_group(label: str, susceptible_types: List[str]) -> str:
    out = f"CG_SUSCEPTIBLE[{label}] = np.array([\n"
    s_groups = itertools.product(susceptible_types, DERIVED_TYPES['vaccination_status'][1])
    for susceptible_type, vaccination_status in s_groups:
        out += f"{TAB}COMPARTMENTS["
        out += f"COMPARTMENT_TYPE.S, "
        out += f"SUSCEPTIBLE_TYPE.{susceptible_type}, "
        out += f"VACCINATION_STATUS.{vaccination_status}],\n"
    out += "], dtype=np.int64)\n"
    return out


def make_eir_compartment_group(compartment: str, label: str,
                              variants: List[str], vaccination_statuses: List[str]) -> str:
    out = f'CG_{compartment.upper()}[{label}] = np.array([\n'
    groups = itertools.product(variants, vaccination_statuses)
    for variant, vaccination_status in groups:
        out += f"{TAB}COMPARTMENTS["
        out += f"COMPARTMENT_TYPE.{compartment[0].upper()}, "
        out += f"AGG_VARIANT.{variant}, "
        out += f"REMOVED_VACCINATION_STATUS.{vaccination_status}],\n"
    out += "], dtype=np.int64)\n"
    return out


def make_compartment_groups() -> str:
    out = ''
    out += 'CG_SUSCEPTIBLE = Dict.empty(\n'
    out += f'{TAB}types.int64,\n'
    out += f'{TAB}types.int64[:],\n'
    out += f')\n'
    for variant, susceptible_types in SUSCEPTIBLE_BY_VARIANT.items():
        out += make_susceptible_compartment_group(f'AGG_VARIANT.{variant}', susceptible_types)
    out += make_susceptible_compartment_group(f'AGG_OTHER.non_immune', DERIVED_TYPES['protection_status'][1])
    out += make_susceptible_compartment_group(f'AGG_OTHER.total', DERIVED_TYPES['susceptible_type'][1])

    for compartment in ['exposed', 'infectious', 'removed']:
        out += ''
        out += f'CG_{compartment.upper()} = Dict.empty(\n'
        out += f'{TAB}types.int64,\n'
        out += f'{TAB}types.int64[:],\n'
        out += f')\n'

        if compartment == 'removed':
            vaccination_status = DERIVED_TYPES['removed_vaccination_status'][1]
        else:
            vaccination_status = DERIVED_TYPES['vaccination_status'][1]

        for variant in DERIVED_TYPES['variant'][1]:
            out += make_eir_compartment_group(compartment, f'AGG_VARIANT.{variant}',
                                              [variant], vaccination_status)
        out += make_eir_compartment_group(compartment, f'AGG_OTHER.total',
                                          DERIVED_TYPES['variant'][1], vaccination_status)

    out += f'CG_TOTAL = Dict.empty(\n'
    out += f'{TAB}types.int64,\n'
    out += f'{TAB}types.int64[:],\n'
    out += f')\n'

    compartment_keys, _ = unpack_spec_fields(SPECS['COMPARTMENTS'])

    out += f"CG_TOTAL[AGG_OTHER.total] = np.array([\n"
    for compartment_key in compartment_keys:
        out += f"{TAB}COMPARTMENTS{compartment_key},\n"
    out += "], dtype=np.int64)\n"
    out += f"CG_TOTAL[AGG_OTHER.unvaccinated] = np.array([\n"
    for compartment_key in compartment_keys:
        out += f"{TAB}COMPARTMENTS{compartment_key},\n"
        if 'VACCINATION_STATUS.unvaccinated' in compartment_key:
            out += f"{TAB}COMPARTMENTS{compartment_key},\n"
    out += "], dtype=np.int64)\n"
    out += f"CG_TOTAL[AGG_OTHER.vaccinated] = np.array([\n"
    for compartment_key in compartment_keys:
        out += f"{TAB}COMPARTMENTS{compartment_key},\n"
        if 'VACCINATION_STATUS.vaccinated' in compartment_key:
            out += f"{TAB}COMPARTMENTS{compartment_key},\n"
    out += "], dtype=np.int64)\n"
    out += f"CG_TOTAL[AGG_OTHER.vaccine_eligible] = np.array([\n"
    for compartment_key in compartment_keys:
        out += f"{TAB}COMPARTMENTS{compartment_key},\n"
        if ('VACCINATION_STATUS.unvaccinated' in compartment_key
                and 'COMPARTMENTS.E' not in compartment_key
                and 'COMPARTMENTS.I' not in compartment_key):
            out += f"{TAB}COMPARTMENTS{compartment_key},\n"
    out += "], dtype=np.int64)\n"

    return out


def make_debug_flag() -> str:
    return textwrap.dedent("""
    # Turning off the JIT is operationally 1-to-1 with
    # saying something is broken in the ODE code and
    # I need to figure it out.
    DEBUG = int(os.getenv('NUMBA_DISABLE_JIT', 0))
    """)


def make_constants() -> str:
    out = make_doctring()
    out += make_imports()
    out += make_primitive_types()
    out += make_derived_types()
    out += make_specs()
    out += make_compartment_groups()
    out += make_debug_flag()
    return out


if __name__ == '__main__':
    here = Path(__file__).resolve()
    constants_file = here.parent.parent / 'constants.py'
    print(f'Generating {str(constants_file)} from {str(here)}')
    constants_string = make_constants()
    with constants_file.open('w') as f:
        f.write(constants_string)
