from argparse import ArgumentParser
import os

from seiir_model_pipeline.diagnostics.versions_comparison import VersionsComparator
from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.utils import load_locations


def get_args():

    parser = ArgumentParser()
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():

    args = get_args()
    os.makedirs(args.output_dir)

    directories = [Directories(regression_version=rv) for rv in args.versions]
    locations = [set(load_locations(d)) for d in directories]
    locs = locations[0]
    for loc in locations[1:]:
        locs = loc & locs
    vc = VersionsComparator(
        list_of_directories=directories,
        groups=list(locs)
    )
    for group in locs:
        vc.compare_coefficients_by_location_plot(group=group, output_dir=args.output_dir)

    for i, version in enumerate(args.versions):
        if i == 0:
            continue
        output_dir = f'{args.output_dir}/{args.versions[0]}-{version}'
        os.makedirs(output_dir)

        vc = VersionsComparator(
            list_of_directories=[directories[0], directories[i]],
            groups=list(locs)
        )
        vc.compare_coefficients_scatterplot(output_dir=output_dir)


if __name__ == '__main__':
    main()
