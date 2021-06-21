#!/bin/python3

import sys
import yaml
import pandas as pd
from schema import Schema, And, Use
from multiprocessing.pool import ThreadPool
import subprocess, time

class ProgramEnv:
    def __init__(self, name : str, options : list[str]):
        self.name = name
        self.options = options

class BenchContext:
    def __init__(self, env : dict[str, ProgramEnv], input_files : list[str]):
        self.env = env
        self.input_files = input_files

class Schedule:
    def __init__(self, program_env: BenchContext):
        self.program_env = program_env

    # TODO: make a function to launch x program with a certain schedule algo
    def run_all_program(self):
        # Run for each input file all the program on thread pool
        result_time = {}

        # TODO: refacto this shit
        for name, p_env in self.program_env.env.items():

            for option in p_env.options:
                program_result = {}
                for input_file in self.program_env.input_files:
                    time = self._run_batch([name, *option.split(" "), input_file])
                    print(
                        f"Exection time for {name} of {input_file.split('/')[-1]} is {time}"
                    )
                    program_result[input_file.split("/")[-1]] = time

                key_name = f"{name.split('/')[-1]} {option}"
                result_time[key_name] = program_result

        return result_time

    def _run_batch(self, args_list):
        start_time = time.time()
        subprocess.run(args_list)
        end_time = time.time()

        return end_time - start_time

def print_usage():
    print("""Usage: Tigerbench config.yaml
    This YAML tend to give the configuration for the program""")

schema = Schema(
    {
        "input_files": [Use(str)],
        "programs": [{"program_name": Use(str), "parameters": [Use(str)]}],
    }
)


def gen_environment(config_path: str) -> BenchContext:
    """This function take config_path as path of the Yaml configuration
    and schema to validate the YAML file"""

    # TODO: refacto if possible check for automatic yaml parsing ??
    input_files: list[str] = []
    program_env: dict[str, ProgramEnv] = {}

    with open(config_path) as config:
        # Load the YAML file
        yaml_loaded = yaml.safe_load(config)

        # Validate it
        schema.validate(yaml_loaded)

        # Get all inputs file
        input_files = yaml_loaded["input_files"]

        # build our environment
        for program_info in yaml_loaded["programs"]:
            program_name = program_info["program_name"]
            program_env[program_name] = ProgramEnv(
                program_name, program_info["parameters"]
            )

    return BenchContext(program_env, input_files)

def main(config_file):
    env = gen_environment(config_file)
    result_times = Schedule(env).run_all_program()
    df = pd.DataFrame(result_times)

    plot_log = df.plot(kind="bar", logy=True, figsize=(10,10))
    plot_normal = df.plot(kind="bar", figsize=(10,10))

    fig = plot_log.get_figure()
    fig.savefig("bench_logscale.jpg")

    fig = plot_normal.get_figure()
    fig.savefig("bench.jpg")

    print(result_times)

if __name__ == "__main__":
    main(sys.argv[1])
