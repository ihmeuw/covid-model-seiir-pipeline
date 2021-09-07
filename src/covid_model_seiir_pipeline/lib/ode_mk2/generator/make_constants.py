import itertools
from pathlib import Path
from typing import List
import textwrap

import inflection

from covid_model_seiir_pipeline.lib.ode_mk2.generator import utils

TAB = '    '  # type: str

PRIMITIVES = {
    'base_compartment': [
        'S', 'E', 'I', 'R',
    ],
    'variant': [
        'ancestral',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'omega',
    ],
    'base_parameter': [
        'alpha',
        'sigma',
        'gamma',
        'pi',
        'new_e',
    ],
    'variant_parameter': [
        'beta',
        'kappa',
        'rho',
    ],
    'risk_group': [
        'lr',
        'hr',
    ],
    'protection_status': [
        'unprotected',

        'non_escape_protected',
        'escape_protected',
        'omega_protected',
    ],
    'immune_status': [
        'non_escape_immune',
        'escape_immune',
        'omega_immune',
    ],
    'vaccination_status': [
        'unvaccinated',
        'vaccinated',
    ],
    'removed_vaccination_status': [
        'unvaccinated',
        'vaccinated',
        'newly_vaccinated',
    ],
}
PRIMITIVES['vaccine_type'] = PRIMITIVES['protection_status'] + PRIMITIVES['immune_status']


SPECS = {
    'PARAMETERS': [
        ['base_parameter', 'all'],
        ['variant_parameter', 'variant'],
    ],
    'VACCINE_TYPES': [
        ['vaccine_type'],
    ],
    'COMPARTMENTS': [
        ['S', 'protection_status', 'vaccination_status'],
        ['S', 'immune_status', 'vaccinated'],
        ['E', 'variant', 'vaccination_status'],
        ['I', 'variant', 'vaccination_status'],
        ['R', 'variant', 'removed_vaccination_status'],
    ],
    'AGGREGATES': [
        ['S', 'variant'],
        ['E', 'variant'],
        ['I', 'variant'],
        ['R', 'variant'],
        ['N', 'vaccination_status'],
    ],
    'NEW_E': [
        ['variant'],
    ],
    'FORCE_OF_INFECTION': [
        ['variant'],
    ],
    'NATURAL_IMMUNITY_WANED': [
        ['variant'],
    ],
    'VACCINE_IMMUNITY_WANED': [
        ['immune_status'],
    ],
}


# Susceptible statuses are rank ordered. Here we map susceptible statuses
# onto which variants they're susceptible to.
SUSCEPTIBLE_BY_VARIANT = {
    # Non-immune compartments
    'ancestral': PRIMITIVES['protection_status'],
    'alpha': PRIMITIVES['protection_status'],
    # Non-immune + non-escape immune
    'beta': PRIMITIVES['protection_status'] + PRIMITIVES['immune_status'][:1],
    'gamma': PRIMITIVES['protection_status'] + PRIMITIVES['immune_status'][:1],
    'delta': PRIMITIVES['protection_status'] + PRIMITIVES['immune_status'][:1],
    # Non-immune + non-escape immune + escape immune
    'omega': PRIMITIVES['protection_status'] + PRIMITIVES['immune_status'][:2],
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


def make_primitives() -> str:
    out = textwrap.dedent("""
    #######################
    # Primitive variables #
    #######################
           
    """)
    for primitive_group_name, variables in PRIMITIVES.items():
        content = utils.make_content_array(rows=variables, columns=[''])
        out += utils.make_named_tuple(
            inflection.camelize(primitive_group_name),
            content,
        ) + '\n'

    for primitive_group_name in PRIMITIVES:
        out += utils.make_name_map(inflection.camelize(primitive_group_name), suffix='')
    return out + '\n'


def make_dict(name: str, num_levels: int) -> str:
    if not num_levels:
        raise

    out = f"{name} = Dict.empty(\n"
    out += f"{TAB}types.UniTuple(types.unicode_type, {num_levels}),\n"
    out += f"{TAB}types.int8,\n"
    out += ")\n"
    return out


def make_dict_items(name: str, *levels: str, count: int = 0):
    levels = [PRIMITIVES.get(level, [level]) for level in levels]
    out = ''
    for elements in itertools.product(*levels):
        out += f"{name}[{tuple(elements)}] = np.int8({count})\n"
        count += 1
    return out, count


def validate_spec(name: str, field_groups: List[List[str]]) -> None:
    field_group_lengths = [len(fg) for fg in field_groups]
    if len(set(field_group_lengths)) != 1:
        raise ValueError(f'{name}: All field groups must have the same number of fields.')
    if field_group_lengths[0] == 0:
        raise ValueError(f'{name}: No fields in spec!')


def make_specs() -> str:
    out = ''
    for spec_name, field_groups in SPECS.items():
        validate_spec(spec_name, field_groups)
        out += make_dict(spec_name, len(field_groups[0]))
        count = 0
        for field_group in field_groups:
            group_items, count = make_dict_items(spec_name, *field_group, count=count)
            out += group_items
        out += f"{spec_name}_NAMES = ['_'.join(k) for k in {spec_name}]\n"

        out += '\n'
    return out


def make_compartment_group(compartment: str, variant: str, *levels: List[str]):
    out = f"COMPARTMENT_GROUPS[('{compartment}', '{variant}')] = np.array([\n"
    for elements in itertools.product(*levels):
        out += f"{TAB}COMPARTMENTS[('{compartment}', "
        for element in elements:
            out += f"'{element}', "
        out = out[:-2] + ")],\n"
    out += "], dtype=np.int8)\n"
    return out


def make_compartment_groups() -> str:
    out = ''
    out += 'COMPARTMENT_GROUPS = Dict.empty(\n'
    out += f'{TAB}types.UniTuple(types.unicode_type, 2),\n'
    out += f'{TAB}types.int8[:],\n'
    out += f')\n'
    for variant, protection_groups in SUSCEPTIBLE_BY_VARIANT.items():
        out += make_compartment_group(
            'S', variant,
            protection_groups, PRIMITIVES['vaccination_status'],
        )
    for compartment in ['E', 'I']:
        for variant in PRIMITIVES['variant']:
            out += make_compartment_group(
                compartment, variant,
                [variant], PRIMITIVES['vaccination_status'],
            )
    for variant in PRIMITIVES['variant']:
        out += make_compartment_group(
            'R', variant,
            [variant], PRIMITIVES['removed_vaccination_status'],
        )
    for vaccination_status in PRIMITIVES['vaccination_status']:
        out += f"COMPARTMENT_GROUPS[('N', '{vaccination_status}')] = np.array([\n"
        out += f"{TAB}v for k, v in COMPARTMENTS.items() if k[2] == '{vaccination_status}'\n"
        out += "])\n"
    out += f"COMPARTMENT_GROUPS[('N', 'vaccine_eligible')] = np.array([\n"
    out += f"{TAB}v for k, v in COMPARTMENTS.items() if k[2] == 'unvaccinated' and k[0] not in ['E', 'I']\n"
    out += "])\n"
    out += f"COMPARTMENT_GROUPS[('N', 'total')] = np.array([\n"
    out += f"{TAB}v for k, v in COMPARTMENTS.items()\n"
    out += "])\n"

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
    out += make_primitives()
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
