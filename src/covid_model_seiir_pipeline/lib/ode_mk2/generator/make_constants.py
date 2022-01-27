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
        'Beta',
        'Infection',
        'Death',
        'Admission',
        'Case',
        'Vaccination',
        'Booster',
        'EffectiveSusceptible',
    ],
    'epi_measure_type': [
        'infection',
        'death',
        'admission',
        'case',
    ],
    'parameter_type': [
        'alpha',
        'sigma',
        'gamma',
        'pi',
        'beta',
        'kappa',
        'rho',
        'count',
        'weight',
        'rate',
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
        'omicron',
        'other',
        'omega',

        'all',
        'total',
    ],
    'vaccine_index_type': [
        'unvaccinated',
        'vaccinated',
        'booster',
        'all',
    ],
    'agg_index_type': [
        'unvaccinated',
        'vaccinated',
        'booster',
        'all',
        'infection',
        'death',
        'admission',
        'case',
    ],
    'system_type': [
        'rates_and_measures',
        'beta_and_measures',
        'beta_and_rates',
    ]
}

DERIVED_TYPES = {
    'compartment': ('compartment_type', [
        'S',
        'E',
        'I',
    ]),
    'tracking_compartment': ('tracking_compartment_type', [
        'Beta',
        'Infection',
        'Death',
        'Admission',
        'Case',
        'Vaccination',
        'Booster',
        'EffectiveSusceptible',
    ]),
    'epi_measure': ('epi_measure_type', [
        'infection',
        'death',
        'admission',
        'case',
    ]),
    'reported_epi_measure': ('epi_measure_type', [
        'death',
        'admission',
        'case',
    ]),
    'base_parameter': ('parameter_type', [
        'alpha',
        'pi',
        'beta',
    ]),
    'epi_parameter': ('parameter_type', [
        'count',
        'weight',
        'rate',
    ]),
    'variant_parameter': ('parameter_type', [
        'rho',
    ]),
    'epi_variant_parameter': ('parameter_type', [
        'sigma',
        'gamma',
        'kappa',
    ]),
    'variant': ('variant_index_type', [
        'none',
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'omicron',
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
        axes_primitives=['parameter_type', 'variant_index_type', 'epi_measure_type'],
        field_specs=[
            ['base_parameter', 'all', 'infection'],
            ['epi_parameter', 'all', 'reported_epi_measure'],
            ['variant_parameter', 'variant', 'infection'],
            ['epi_variant_parameter', 'variant', 'epi_measure']
        ],
    ),
    'AGE_SCALARS': Spec(
        offset='',
        axes_primitives=['epi_measure_type'],
        field_specs=[
            ['reported_epi_measure'],
        ],
    ),
    'BASE_RATES': Spec(
        offset='',
        axes_primitives=['epi_measure_type'],
        field_specs=[
            ['reported_epi_measure'],
        ],
    ),
    'RATES': Spec(
        offset='',
        axes_primitives=['vaccine_index_type', 'variant_index_type', 'variant_index_type', 'epi_measure_type'],
        field_specs=[
            ['vaccine_status', 'variant', 'variant', 'reported_epi_measure']
        ]
    ),
    'VARIANT_WEIGHTS': Spec(
        offset='',
        axes_primitives=['epi_measure_type'],
        field_specs=[
            ['epi_measure'],
        ],
    ),
    'BETAS': Spec(
        offset='',
        axes_primitives=['epi_measure_type'],
        field_specs=[
            ['epi_measure'],
        ],
    ),
    'ETA': Spec(
        offset='',
        axes_primitives=['vaccine_index_type', 'variant_index_type', 'epi_measure_type'],
        field_specs=[
            ['vaccine_status', 'variant', 'epi_measure']
        ],
    ),
    'CHI': Spec(
        offset='',
        axes_primitives=['variant_index_type', 'variant_index_type', 'epi_measure_type'],
        field_specs=[
            ['variant', 'variant', 'epi_measure'],
        ],
    ),
    'NEW_E': Spec(
        offset='',
        axes_primitives=['vaccine_index_type', 'variant_index_type', 'variant_index_type'],
        field_specs=[
            ['vaccine_status', 'variant', 'variant'],
        ],
    ),
    'EFFECTIVE_SUSCEPTIBLE': Spec(
        offset='',
        axes_primitives=['vaccine_index_type', 'variant_index_type', 'variant_index_type'],
        field_specs=[
            ['vaccine_status', 'variant', 'variant'],
        ],
    ),
    'COMPARTMENTS': Spec(
        offset='',
        axes_primitives=['compartment_type', 'vaccine_index_type', 'variant_index_type'],
        field_specs=[
            ['compartment', 'vaccine_status', 'variant'],
        ]
    ),
    'TRACKING_COMPARTMENTS': Spec(
        offset='COMPARTMENTS',
        axes_primitives=['tracking_compartment_type', 'variant_index_type', 'variant_index_type', 'agg_index_type'],
        field_specs=[
            ['Beta', 'none', 'none', 'all'],
            ['Beta', 'none', 'none', 'death'],
            ['Beta', 'none', 'none', 'admission'],
            ['Beta', 'none', 'none', 'case'],

            ['Infection', 'none', 'variant', 'unvaccinated'],
            ['Infection', 'none', 'all', 'unvaccinated'],
            ['Infection', 'none', 'all', 'all'],
            ['Infection', 'all', 'all', 'all'],
            ['Infection', 'all', 'all', 'vaccine_status'],
            ['Infection', 'all', 'variant', 'all'],

            ['Death', 'none', 'variant', 'unvaccinated'],
            ['Death', 'none', 'all', 'unvaccinated'],
            ['Death', 'none', 'all', 'all'],
            ['Death', 'all', 'all', 'all'],
            ['Death', 'all', 'all', 'vaccine_status'],
            ['Death', 'all', 'variant', 'all'],

            ['Admission', 'none', 'variant', 'unvaccinated'],
            ['Admission', 'none', 'all', 'unvaccinated'],
            ['Admission', 'none', 'all', 'all'],
            ['Admission', 'all', 'all', 'all'],
            ['Admission', 'all', 'all', 'vaccine_status'],
            ['Admission', 'all', 'variant', 'all'],

            ['Case', 'none', 'variant', 'unvaccinated'],
            ['Case', 'none', 'all', 'unvaccinated'],
            ['Case', 'none', 'all', 'all'],
            ['Case', 'all', 'all', 'all'],
            ['Case', 'all', 'all', 'vaccine_status'],
            ['Case', 'all', 'variant', 'all'],

            ['Vaccination', 'all', 'all', 'unvaccinated'],
            ['Booster', 'all', 'all', 'vaccinated'],

            ['EffectiveSusceptible', 'all', 'variant', 'all'],
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

    out += utils.make_import('numpy as np')
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
        out += '), TOMBSTONE, dtype=np.int64)\n'

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


def make_constants() -> str:
    out = make_doctring()
    out += make_imports()
    out += 'TOMBSTONE = -12345\n\n'
    out += make_primitive_types()
    out += make_derived_types()
    out += make_specs()
    return out


if __name__ == '__main__':
    here = Path(__file__).resolve()
    constants_file = here.parent.parent / 'constants.py'
    print(f'Generating {str(constants_file)} from {str(here)}')
    constants_string = make_constants()
    with constants_file.open('w') as f:
        f.write(constants_string)
