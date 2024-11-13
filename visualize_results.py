"""Perform grid search over many parameters."""

import os
import re
from datetime import datetime

import click
import zeno_client
import pandas as pd

from data_utils import (
    load_data,
    load_data_aider_bench,
    get_model_name_aider_bench,
    load_swe_bench_dataset,
)
from models import SWEBenchPath


def visualise_swe_bench(
    benchmark_paths: list[SWEBenchPath],
    project_title: str | None = None,
    zeno_api_key: str | None = None,
):
    """
    Visualize data from multiple input files.

    Args:
        input_files (list[str]): A list of filepaths representing SWE-bench trajectories generated
        by OpenHands.

        project_title (str | None, default=None): Title to use for the generated ZenoML project. If
        not provided, generate a title based on the current date.

        zeno_api_key (str | None, default=None): An optional API key for ZenoML. If not provided,
        will attempt to read from the `ZENO_API_KEY` environment variable.

    Raises:
        AssertionError: If no API key can be found in the parameters or environment.
    """
    data = [load_data(input_file.trajectories) for input_file in benchmark_paths]
    ids = [trajectory[0] for dataset in data for trajectory in dataset]
    id_map = {x: i for (i, x) in enumerate(ids)}

    # Find all duplicate values in "ids"
    seen = set()
    duplicates = set()
    for x in ids:
        if x in seen:
            duplicates.add(x)
        seen.add(x)
    print(duplicates)

    # Find an API key and build a client.
    if zeno_api_key is None:
        zeno_api_key = os.environ.get("ZENO_API_KEY")

    assert zeno_api_key, "Can't find ZenoML API key"

    vis_client = zeno_client.ZenoClient(zeno_api_key)

    # Set the title if not given.
    if project_title is None:
        project_title = f"SWE-bench Performance: {datetime.now()}"

    vis_project = vis_client.create_project(
        name=project_title,
        view={
            "data": {"type": "markdown"},
            "label": {"type": "text"},
            "output": {"type": "text"},
        },
        description="OpenHands agent performance comparisons on SWE-bench",
        public=False,
        metrics=[
            zeno_client.ZenoMetric(name="resolved", type="mean", columns=["resolved"]),
        ],
    )

    # Upload the structure of the dataset.
    vis_project.upload_dataset(
        load_swe_bench_dataset(benchmark_paths[0]),
        id_column="id",
        data_column="problem_statement",
    )

    # Do evaluation
    for input_file, data_entry in zip(benchmark_paths, data):
        resolved = [0] * len(data[0])
        for entry in data_entry:
            resolved[id_map[entry[0]]] = entry[2]
        df_system = pd.DataFrame(
            {
                "id": ids,
                "resolved": resolved,
            },
            index=ids,
        )
        model_name = re.sub(r"data/.*lite/", "", str(input_file.trajectories))
        model_name = re.sub(r"(od_output|output).jsonl", "", model_name)
        model_name = model_name.replace("/", "_")
        vis_project.upload_system(
            df_system, name=model_name, id_column="id", output_column="resolved"
        )


def visualize_aider_bench(input_files: list[str]):
    """Visualize data from multiple input files."""
    data = [load_data_aider_bench(input_file) for input_file in input_files]
    ids = [x[0] for x in data[0]]
    id_map = {x: i for (i, x) in enumerate(ids)}

    # Find all duplicate values in "ids"
    seen = set()
    duplicates = set()
    for x in ids:
        if x in seen:
            duplicates.add(x)
        seen.add(x)
    print(duplicates)

    vis_client, vis_project = None, None
    print(os.environ.get("ZENO_API_KEY"))
    vis_client = zeno_client.ZenoClient(os.environ.get("ZENO_API_KEY"))

    # use zeno to visualize
    df_data = pd.DataFrame(
        {
            "id": ids,
            "instruction": [x[1] for x in data[0]],
        },
        index=ids,
    )
    df_data["instruction_length"] = df_data["instruction"].apply(len)
    # df_data["repo"] = df_data["id"].str.rsplit("-", n=1).str[0]
    vis_project = vis_client.create_project(
        name="Aider Bench Code Editing Visualization",
        view={
            "data": {"type": "markdown"},
            "label": {"type": "text"},
            "output": {"type": "markdown"},
        },
        description="Aider Bench Code Editing",
        public=False,
        metrics=[
            zeno_client.ZenoMetric(name="resolved", type="mean", columns=["resolved"]),
        ],
    )
    vis_project.upload_dataset(df_data, id_column="id", data_column="instruction")

    # Do evaluation
    for input_file, data_entry in zip(input_files, data):
        output = [""] * len(data[0])
        resolved = [0] * len(data[0])
        for entry in data_entry:
            resolved[id_map[entry[0]]] = entry[2]
            output[
                id_map[entry[0]]
            ] += f"## Resolved\n {entry[2]} \n ## Test Cases\n {entry[3]}\n ## Agent Trajectory\n"
            for i in range(len(entry[4])):
                output[id_map[entry[0]]] += f"### Step {i+1} \n"
                output[id_map[entry[0]]] += f'Action: {entry[4][i]["action"]}\n'
                output[id_map[entry[0]]] += f'Code: {entry[4][i]["code"]}\n'
                output[id_map[entry[0]]] += f'Thought: {entry[4][i]["thought"]}\n'
                output[
                    id_map[entry[0]]
                ] += f'Observation: {entry[4][i]["observation"]}\n'

        df_system = pd.DataFrame(
            {"id": ids, "agent output": output, "resolved": resolved},
            index=ids,
        )

        vis_project.upload_system(
            df_system,
            name=get_model_name_aider_bench(input_file),
            id_column="id",
            output_column="agent output",
        )


@click.command()
@click.argument("benchmarks", type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option("--report-title", type=str)
@click.option(
    "--benchmark", type=click.Choice(["swe-bench", "aider-bench"]), default="swe-bench"
)
@click.option("--zeno-api-key", type=str, envvar="ZENO_API_KEY")
def cli(benchmarks, report_title, benchmark, zeno_api_key):
    """
    Generate a ZenoML performance report over the provided files.
    """
    match benchmark:
        case "swe-bench":
            visualise_swe_bench(
                [SWEBenchPath.from_directory(benchmark) for benchmark in benchmarks],
                report_title,
                zeno_api_key,
            )
        case "aider-bench":
            visualize_aider_bench(benchmarks)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
