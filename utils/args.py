import argparse


def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--local",
        type=bool,
        default=False,
        help="If True, init ray in local mode.",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to use WanDB logging.",
    )
    parser.add_argument(
        "--num_agent",
        type=int,
        default=2,
        help="Number of agents in the environment.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=1e8,
        help="Number of timesteps to run the experiment for.",
    )

    args, remaining = parser.parse_known_args()
    print("Running trails with the following arguments: ", args)
    return args, remaining


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)