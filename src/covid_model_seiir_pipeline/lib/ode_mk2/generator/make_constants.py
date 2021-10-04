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
        'N',
    ],
    'tracking_compartment_type': [
        'NewE',
        'NewVaccination',
        'NewBooster',
        'EffectiveSusceptible',
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
        'eta',
    ],
    'risk_group': [
        'lr',
        'hr',
    ],
    'variant_index_type': [
        'none',
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',

        'all',
        'total',
    ],
    'vaccine_index_type': [
        'unvaccinated',
        'vaccinated',
        'booster',
    ],
    'agg_index_type': [
        'none',
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',

        'all',
        'total',

        'unvaccinated',
        'vaccinated',
        'booster',
    ]
}

DERIVED_TYPES = {
    'compartment': ('compartment_type', [
        'S',
        'E',
        'I',
    ]),
    'tracking_compartment': ('tracking_compartment_type', [
        'NewE',
        'NewVaccination',
        'NewBooster',
        'EffectiveSusceptible',
    ]),
    'base_parameter': ('parameter_type', [
        'alpha',
        'sigma',
        'gamma',
        'pi',
        'new_e',
        'beta',
    ]),
    'variant_parameter': ('parameter_type', [
        'kappa',
        'rho',
        'eta',
    ]),
    'variant': ('variant_index_type', [
        'none',
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'other',
        'omega',
    ]),
    'variant_group': ('variant_index_type', [
        'all',
        'total'
    ]),
    'vaccine_status': ('vaccine_index_type', [
        'unvaccinated',
        'vaccinated',
        'booster',
    ]),
}

Spec = namedtuple('Spec', ['offset', 'axes_primitives', 'field_specs'])

SPECS = {
    'PARAMETERS': Spec(
        offset='',
        axes_primitives=['parameter_type', 'variant_index_type'],
        field_specs=[
            ['base_parameter', 'all'],
            ['variant_parameter', 'variant'],
        ],
    ),
    'PHI': Spec(
        offset='',
        axes_primitives=['variant_index_type', 'variant_index_type'],
        field_specs=[
            ['variant', 'variant'],
        ],
    ),
    'NEW_E': Spec(
        offset='',
        axes_primitives=['vaccine_index_type', 'variant_index_type', 'variant_index_type'],
        field_specs=[
            ['vaccine_status', 'variant', 'variant'],
        ],
    ),
    'COMPARTMENTS': Spec(
        offset='',
        axes_primitives=['compartment_type', 'variant_index_type', 'vaccine_index_type'],
        field_specs=[
            ['compartment', 'variant', 'vaccine_status'],
        ]
    ),
    'TRACKING_COMPARTMENTS': Spec(
        offset='COMPARTMENTS',
        axes_primitives=['tracking_compartment_type', 'variant_index_type'],
        field_specs=[
            ['tracking_compartment', 'variant']
        ],
    ),
    'AGGREGATES': Spec(
        offset='',
        axes_primitives=['compartment_type', 'variant_index_type'],
        field_specs=[
            ['I', 'variant'],
            ['N', 'total'],
        ]
    )
}


def make_doctring() -> str:
    description = "Static definitions for the compartments and model parameters."
    return utils.make_module_docstring(description, __file__)


def make_imports() -> str:
    out = utils.make_import('collections', ['namedtuple'])
    out += utils.make_import('os') + '\n'

    out += utils.make_import('numpy as np')
    # out += utils.make_import('numba')
    # out += utils.make_import('numba.core', ['types'])
    # out += utils.make_import('numba.typed', ['Dict'])
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
    return out + '\n'


def unpack_spec_fields(spec: Spec) -> Tuple[List[str], List[str]]:
    field_keys = []
    field_names = []
    for field_spec in spec.field_specs:
        fields = []
        for field_type, primitive in zip(field_spec, spec.axes_primitives):
            if field_type in DERIVED_TYPES:
                fields.append((field_type, DERIVED_TYPES[field_type][1]))
            else:
                fields.append((primitive, [field_type]))
        for elements in itertools.product(*[field[1] for field in fields]):
            field_keys.append(
                '['
                + ', '.join([f'{inflection.underscore(primitive).upper()}.{element}'
                             for primitive, element in zip([field[0] for field in fields], elements)])
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


# def make_compartment_group(compartment: str, variant: str, *levels: List[str]):
#     out = f"COMPARTMENT_GROUPS[('{compartment}', '{variant}')] = np.array([\n"
#     for elements in itertools.product(*levels):
#         out += f"{TAB}COMPARTMENTS[('{compartment}', "
#         for element in elements:
#             out += f"'{element}', "
#         out = out[:-2] + ")],\n"
#     out += "], dtype=np.int64)\n"
#     return out
#
#
# def make_susceptible_compartment_group(label: str, susceptible_types: List[str]) -> str:
#     out = ""
#     out += f"{TAB}if group == {label}:\n"
#     out += f"{2*TAB}return np.array([\n"
#     s_groups = itertools.product(susceptible_types, DERIVED_TYPES['vaccination_status'][1])
#     for susceptible_type, vaccination_status in s_groups:
#         out += f"{3*TAB}COMPARTMENTS["
#         out += f"COMPARTMENT_TYPE.S, "
#         out += f"SUSCEPTIBLE_TYPE.{susceptible_type}, "
#         out += f"VACCINATION_STATUS.{vaccination_status}],\n"
#     out += f"{2*TAB}], dtype=np.int64)\n"
#     return out
#
#
# def make_eir_compartment_group(compartment: str, label: str,
#                                variants: List[str], vaccination_statuses: List[str]) -> str:
#     out = ""
#     out += f"{TAB}if group == {label}:\n"
#     out += f"{2*TAB}return np.array([\n"
#     groups = itertools.product(variants, vaccination_statuses)
#     for variant, vaccination_status in groups:
#         out += f"{3*TAB}COMPARTMENTS["
#         out += f"COMPARTMENT_TYPE.{compartment[0].upper()}, "
#         out += f"AGG_VARIANT.{variant}, "
#         out += f"REMOVED_VACCINATION_STATUS.{vaccination_status}],\n"
#     out += f"{2*TAB}], dtype=np.int64)\n"
#     return out
#
#
# def make_compartment_groups() -> str:
#     out = ''
#     out += '@numba.njit\n'
#     out += 'def CG_SUSCEPTIBLE(group: int) -> np.ndarray:\n'
#     for variant, susceptible_types in SUSCEPTIBLE_BY_VARIANT.items():
#         out += make_susceptible_compartment_group(f'AGG_VARIANT.{variant}', susceptible_types)
#     out += make_susceptible_compartment_group(f'AGG_OTHER.non_immune', DERIVED_TYPES['protection_status'][1])
#     out += make_susceptible_compartment_group(f'AGG_OTHER.total', DERIVED_TYPES['susceptible_type'][1])
#     out += '\n'
#
#     for compartment in ['exposed', 'infectious', 'removed']:
#         out += '@numba.njit\n'
#         out += f'def CG_{compartment.upper()}(group: int) -> np.ndarray:\n'
#         if compartment == 'removed':
#             vaccination_status = DERIVED_TYPES['removed_vaccination_status'][1]
#         else:
#             vaccination_status = DERIVED_TYPES['vaccination_status'][1]
#
#         for variant in DERIVED_TYPES['variant'][1]:
#             out += make_eir_compartment_group(compartment, f'AGG_VARIANT.{variant}',
#                                               [variant], vaccination_status)
#         out += make_eir_compartment_group(compartment, f'AGG_OTHER.total',
#                                           DERIVED_TYPES['variant'][1], vaccination_status)
#         out += '\n'
#
#     out += '@numba.njit\n'
#     out += f'def CG_TOTAL(group: int) -> np.ndarray:\n'
#
#     compartment_keys, _ = unpack_spec_fields(SPECS['COMPARTMENTS'])
#     aggregate_map = {
#         'AGG_OTHER.total': lambda x: True,
#         'AGG_OTHER.unvaccinated': lambda x: 'VACCINATION_STATUS.unvaccinated' in x,
#         'AGG_OTHER.vaccinated': lambda x: 'VACCINATION_STATUS.vaccinated' in x,
#         'AGG_OTHER.vaccine_eligible': (lambda x: ('VACCINATION_STATUS.unvaccinated' in compartment_key
#                                                   and 'COMPARTMENT_TYPE.E' not in compartment_key
#                                                   and 'COMPARTMENT_TYPE.I' not in compartment_key))
#     }
#
#     for aggregate, in_aggregate in aggregate_map.items():
#         out += f"{TAB}if group == {aggregate}:\n"
#         out += f"{2 * TAB}return np.array([\n"
#         for compartment_key in compartment_keys:
#             if in_aggregate(compartment_key):
#                 out += f"{3 * TAB}COMPARTMENTS{compartment_key},\n"
#         out += f"{2 * TAB}], dtype=np.int64)\n"
#
#     return out


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
    # out += make_compartment_groups()
    out += make_debug_flag()
    return out


if __name__ == '__main__':
    here = Path(__file__).resolve()
    constants_file = here.parent.parent / 'constants.py'
    print(f'Generating {str(constants_file)} from {str(here)}')
    constants_string = make_constants()
    with constants_file.open('w') as f:
        f.write(constants_string)
