from datetime import datetime
import json
import pandas as pd

import click
import zeno_client


from models import SWEBenchInstance
from models.openhands import BenchmarkPath


# pylint: disable=unspecified-encoding


def load_dataset(benchmark_path: BenchmarkPath) -> pd.DataFrame:
    """
    Load the dataset (the IDs and problem defs) from a benchmark.
    """
    rows: list[pd.DataFrame] = []

    with open(benchmark_path.trajectories, "r") as file:
        for line in file.readlines():
            data = json.loads(line)
            row = pd.DataFrame(
                [
                    {
                        "id": data.get("instance_id"),
                        "problem_statement": data.get("instance", {}).get(
                            "problem_statement"
                        ),
                    }
                ]
            )
            rows.append(row)

    dataset = pd.concat(rows, ignore_index=True)
    dataset["statement_length"] = dataset["problem_statement"].apply(len)
    dataset["repo"] = dataset["id"].str.rsplit("-", n=1).str[0]

    return dataset


def load_system(benchmark_path: BenchmarkPath) -> pd.DataFrame:
    # Load the results. When we load a trajectory we'll find the associated instance object and
    # flatten the needed parameters.
    results: dict[str, SWEBenchInstance] = {}
    with open(benchmark_path.results, "r") as file:
        for line in file.readlines():
            instance = SWEBenchInstance.model_validate_json(line)
            results[instance.instance_id] = instance

    # Load the trajectories.
    trajectories: list[pd.DataFrame] = []
    with open(benchmark_path.trajectories, "r") as file:
        for line in file.readlines():
            data = json.loads(line)
            instance = results[data["instance_id"]]
            trajectory = pd.DataFrame(
                [
                    {
                        "id": data["instance_id"],
                        "history": data["history"],
                        "resolved": instance.test_result.report.resolved,
                        "git_patch": instance.test_result.git_patch,
                    }
                ]
            )
            trajectories.append(trajectory)

    return pd.concat(trajectories, ignore_index=True)


@click.command()
@click.argument("benchmarks", type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option("--project-title", type=str)
@click.option("--zeno-api-key", type=str, envvar="ZENO_API_KEY")
def cli(benchmarks, project_title, zeno_api_key):
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
            load_system(benchmark_path),
            name=benchmark_path.name,
            id_column="id",
            output_column="git_patch",
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
