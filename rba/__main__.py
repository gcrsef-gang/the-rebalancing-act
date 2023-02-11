"""
usage: python3 -m rba <command> [args] [-v]

Commands
--------
 - communitygen [--graph_file] [--num_thresholds]
    Generates `num_thresholds` community maps based on the precinct graph `graph`, and writes to a
    file storing a list of individual communities, containing data on constituent precincts, birth 
    and death times.
 - districtgen
 - ensemblegen
 - quantify
 - draw

Use -v or --verbose for verbosity.
"""

import argparse
import os

import rba


if __name__ == "__main__":
    package_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers()

    communitygen_parser = subparsers.add_parser("communitygen")
    communitygen_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
    communitygen_parser.add_argument("--num_thresholds", type=int, default=50)
    communitygen_parser.add_argument("--output_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
    communitygen_parser.set_defaults(func=rba.community_generation.create_communities)
    
    quantify = subparsers.add_parser("quantify")
    quantify.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
    quantify.add_argument("--district_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_districts.json"))
    quantify.add_argument("--community_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
    quantify.set_defaults(func=rba.district_quantification.quantify_districts)
    args = parser.parse_args()
    args.func(**{key: val for key, val in vars(args).items() if key != "func"})