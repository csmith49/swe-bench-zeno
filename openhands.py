from datetime import datetime
from typing import Callable
import pandas as pd

import click
import zeno_client

from models.openhands import BenchmarkPath


# pylint: disable=unspecified-encoding


def load_dataset(benchmark_path: BenchmarkPath) -> pd.DataFrame:
    """
    Load the dataset (the IDs and problem defs) from a benchmark.
    """
    rows: list[pd.DataFrame] = []

    for trajectory in benchmark_path.load_trajectories():
        row = pd.DataFrame(
            [
                {
                    "id": trajectory.instance_id,
                    "problem_statement": trajectory.instance.problem_statement,
                }
            ]
        )
        rows.append(row)

    dataset = pd.concat(rows, ignore_index=True)
    dataset["statement_length"] = dataset["problem_statement"].apply(len)
    dataset["repo"] = dataset["id"].str.rsplit("-", n=1).str[0]

    return dataset


def load_system(benchmark_path: BenchmarkPath) -> pd.DataFrame:
    """
    Load a system (the agents performance over each instance in the dataset) from a benchmark.
    """
    # Load the results and metadata. When we convert a trajectory we'll flatten some of the
    # relevant info stored in these objects.
    results = benchmark_path.load_results()
    metadata = benchmark_path.load_metadata()

    rows: list[pd.DataFrame] = []
    for trajectory in benchmark_path.load_trajectories():
        test_result = results[trajectory.instance_id].test_result

        # Convert the trajectory to a dataframe.
        row = pd.DataFrame(
            [
                {
                    "id": trajectory.instance_id,
                    "history": trajectory.history,
                    # The test results contain the outputs and Boolean summaries of agent
                    # performance.
                    **test_result.dump(),
                    # The metadata contains information about agent/LLM configuration. Should be
                    # consistent across all trajectories in this system.
                    **metadata.dump(),
                }
            ]
        )
        rows.append(row)

    return pd.concat(rows, ignore_index=True)


def compute_metrics(system: pd.DataFrame, metrics: dict[str, Callable]) -> pd.DataFrame:
    """
    Extend a system with additional metrics.

    Modifies the system in-place.
    """
    for key, metric in metrics.items():
        system[key] = system.apply(metric, axis=1)

    return system


@click.command()
@click.argument("benchmarks", type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option("--project-title", type=str)
@click.option("--zeno-api-key", type=str, envvar="ZENO_API_KEY")
def cli(benchmarks, project_title, zeno_api_key) -> None:
    """
    Generate a ZenoML performance report for the given OpenHands benchmarks.
    """
    # If we have a key we can build the viz client.
    assert zeno_api_key, "Can't find ZenoML API key"

    viz_client = zeno_client.ZenoClient(zeno_api_key)

    # Set the title, if not given.
    if project_title is None:
        project_title = f"SWE-bench Performance: {datetime.now()}"

    # Initialize a new project.
    viz_project = viz_client.create_project(
        name=project_title,
        view={
            "data": {"type": "markdown"},
            "label": {"type": "text"},
            "output": {"type": "code"},
        },
        description="OpenHands agent performance comparisons on SWE-bench",
        public=False,
        metrics=[
            zeno_client.ZenoMetric(name="resolved", type="mean", columns=["resolved"]),
        ],
    )

    uploaded_dataset: bool = False

    for benchmark in benchmarks:
        benchmark_path = BenchmarkPath.from_directory(benchmark)

        # If we haven't yet sent the dataset to the project, do so before building the system. We
        # should only have to do this once as it's common across all benchmarks.
        if not uploaded_dataset:
            viz_project.upload_dataset(
                load_dataset(benchmark_path),
                id_column="id",
                data_column="problem_statement",
            )
            uploaded_dataset = True

        # Then convert the benchmark to a ZenoML system.
        viz_project.upload_system(
            compute_metrics(
                load_system(benchmark_path),
                {
                    "history_length": lambda row: len(row["history"]),
                    "git_patch_length": lambda row: len(row["git_patch"]),
                    "git_patch_insertions": lambda row: len(
                        [
                            line
                            for line in row["git_patch"].split("\n")
                            if line.startswith("+")
                        ]
                    ),
                    "git_patch_deletions": lambda row: len(
                        [
                            line
                            for line in row["git_patch"].split("\n")
                            if line.startswith("-")
                        ]
                    ),
                },
            ),
            name=benchmark_path.name,
            id_column="id",
            output_column="git_patch",
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
