"""
usage: python3 -m rba <command> [args] [-v]

Commands
--------
 - communitygen [--graph_file] [--num_thresholds]
    Generates `num_thresholds` community maps based on the precinct graph `graph`, and writes to a
    file storing a list of individual communities, containing data on constituent precincts, birth 
    and death times.
 - optimize [--graph_file] [--communitygen_out_file] [--vra_config_file] [--num_steps]
            [--num_districts] [--initial_plan_file] [--output_dir]
    Runs simulated annealing algorithm, saves the ten best maps, as well as a dataframe keeping
    track of various statistics for each state of the chain. `vra_config` is a JSON file
    containing information about minority-opportunity district constraints.
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
    communitygen_parser.add_argument("--state", type=str, default="new_hampshire")
   #  communitygen_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
    communitygen_parser.add_argument("--num_thresholds", type=int, default=50)
   #  communitygen_parser.add_argument("--output_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
    communitygen_parser.set_defaults(func=rba.community_generation.create_communities)
    
    quantify_parser = subparsers.add_parser("quantify")
    quantify_parser.add_argument("--state", type=str, default="new_hampshire")
   #  quantify_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
   #  quantify_parser.add_argument("--district_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_districts.json"))
   #  quantify_parser.add_argument("--community_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
    quantify_parser.set_defaults(func=rba.district_quantification.quantify_districts)

    draw_parser = subparsers.add_parser("draw")
    draw_parser.add_argument("--state", type=str, default="new_hampshire")
    draw_parser.add_argument("--output_file", type=str)
   #  draw_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
    draw_parser.add_argument("--edge_lifetime_file", type=str, default=None)
    draw_parser.add_argument("--num_frames", type=int, default=50)
    draw_parser.add_argument("--partition_file", type=str, default=None)
    draw_parser.set_defaults(func=rba.visualization.visualize)

    ensemble_parser = subparsers.add_parser("ensemble")
    ensemble_parser.add_argument("--state", type=str, default="new_hampshire")
   #  ensemble_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
   #  ensemble_parser.add_argument("--community_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
   #  ensemble_parser.add_argument("--vra_config_file", type=str, default=os.path.join(package_dir, "data/2010/vra_nh.json"))
    ensemble_parser.add_argument("--num_steps", type=int, default=100)
    ensemble_parser.add_argument("--num_districts", type=int, default=2)
    ensemble_parser.add_argument("--initial_plan_file", type=str, default=None)
   #  ensemble_parser.add_argument("--district_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_districts.json"))
    ensemble_parser.add_argument("-o", "--output_dir", type=str)
    ensemble_parser.set_defaults(func=rba.ensemble.ensemble_analysis)

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--state", type=str, default="new_hampshire")
   #  optimize_parser.add_argument("--graph_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_geodata_merged.json"))
   #  optimize_parser.add_argument("--communitygen_out_file", type=str, default=os.path.join(package_dir, "data/2010/new_hampshire_communities.json"))
   #  optimize_parser.add_argument("--vra_config_file", type=str, default=os.path.join(package_dir, "data/2010/vra_nh.json"))
    optimize_parser.add_argument("--num_steps", type=int, default=100)
    optimize_parser.add_argument("--num_districts", type=int, default=2)
    optimize_parser.add_argument("--initial_plan_file", type=str, default=None)
    optimize_parser.add_argument("-o", "--output_dir", type=str)
    optimize_parser.set_defaults(func=rba.optimization.optimize)

    args = parser.parse_args()
    arguments = {key: val for key, val in vars(args).items() if key != "func" and key != "state"}
    state = vars(args)["state"]
    if args.func.__name__ == "create_communities":
        arguments["graph_file"] = os.path.join(package_dir, f"data/2010/{state}_geodata_merged.json")
        arguments["output_file"] = os.path.join(package_dir, f"data/2010/{state}_communities.json")
    elif args.func.__name__ == "quantify_districtss":
        arguments["graph_file"] = os.path.join(package_dir, f"data/2010/{state}_geodata_merged.json")
        arguments["community_file"] = os.path.join(package_dir, f"data/2010/{state}_communities.json")
        arguments["district_file"] = os.path.join(package_dir, f"data/2010/{state}_districts.json")
    elif args.func.__name__ == "visualize":
        arguments["graph_file"] = os.path.join(package_dir, f"data/2010/{state}_geodata_merged.json")
    elif args.func.__name__ == "ensemble_analysis":
        arguments["graph_file"] = os.path.join(package_dir, f"data/2010/{state}_geodata_merged.json")
        arguments["community_file"] = os.path.join(package_dir, f"data/2010/{state}_communities.json")
        arguments["district_file"] = os.path.join(package_dir, f"data/2010/{state}_districts.json")
        arguments["vra_config_file"] = os.path.join(package_dir, f"data/2010/vra_{state}.json")
    elif args.func.__name__ == "optimize":
        arguments["graph_file"] = os.path.join(package_dir, f"data/2010/{state}_geodata_merged.json")
        arguments["community_file"] = os.path.join(package_dir, f"data/2010/{state}_communities.json")
        arguments["vra_config_file"] = os.path.join(package_dir, f"data/2010/vra_{state}.json")
   #  if arguments["func"] 
   #  args.func(**{key: val for key, val in vars(args).items() if key != "func"})
    args.func(**arguments)