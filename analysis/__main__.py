from pathlib import Path
from datetime import datetime
import click

import pandas as pd
import zeno_client

from swe_bench.models import Split
from analysis.models.data import Data
from analysis.features import compute_features as compute_features_dataframe
from analysis.performance_gap import top_performers, unresolved_instances

@click.group()
def cli(): ...


@cli.command()
@click.option(
    "--split",
    type=Split,
    default="verified",
    callback=lambda _ctx, _, value: Split.from_str(value),
)
@click.option("--output", "-o", type=str, default="data.json")
def download(split: Split, output: str) -> None:
    """Download and store SWE-bench data locally."""
    data = Data.download(split)
    with open(output, "w") as f:
        f.write(data.model_dump_json())

    # Compute size of downloaded file
    file_size = Path(output).stat().st_size
    click.echo(f"Downloaded {file_size} bytes to {output}")


@cli.command()
@click.option("--input", "-i", type=str, default="data.json")
@click.option("--output", "-o", type=str, default="features.csv")
def compute_features(input: str, output: str) -> None:
    """Compute features for the downloaded data."""
    with open(input) as f:
        data = Data.model_validate_json(f.read())

    df = compute_features_dataframe(data.dataset.instances)
    df.to_csv(output, index=False)

@cli.command()
@click.option("--data", "-d", type=str, default="data.json")
@click.option("--features", "-f", type=str, default="features.csv")
@click.option("--zeno-api-key", type=str, envvar="ZENO_API_KEY")
@click.option("--top-k", type=int, default=5, help="Only include top k systems (and OH)")
def upload(data: str, features: str, zeno_api_key: str, top_k: int) -> None:
    """Upload data and features to Zeno."""
    assert zeno_api_key, "No Zeno API key found."
    viz_client = zeno_client.ZenoClient(zeno_api_key)
    
    with open(data) as f:
        data = Data.model_validate_json(f.read())

    df = pd.read_csv(features)
    
    source = data.systems[data.closest_system("OpenHands")]
    targets = top_performers(data.systems.values(), k=top_k)

    # Create a new project.
    current_time = datetime.now()
    viz_project = viz_client.create_project(
        name="SWE-bench Leaderboard",
        view={
            "data": {"type": "markdown"},
            "label": {"type": "text"},
            "output": {
                "type": "vstack",
                "keys": {
                    "status": {"type": "text", "label": "Status"},
                    "patch": {"type": "code"},
                }
            },
        },
        description=f"SWE-bench leaderboard (as of {current_time}) performance analysis, by entry.",
        public=True,
        metrics=[
            zeno_client.ZenoMetric(name="resolved", type="mean", columns=["resolved"])
        ],
    )

    # Build and upload the dataset.
    viz_project.upload_dataset(df, id_column="instance_id", data_column="instance/problem_statement")

    systems = {name: evaluation for name, evaluation in data.systems.items() if evaluation in [source, *targets]}

    for name, system in systems.items():
        data = pd.DataFrame(
            [
                {
                    "instance_id": prediction.instance_id,
                    "resolved": system.results.is_resolved(prediction.instance_id),
                    "output": {
                        "status": "✅ Success" if system.results.is_resolved(prediction.instance_id)
                                else "❌ Failed" if prediction.patch
                                else "Not attempted",
                        "patch": prediction.patch or "No patch generated",
                    }
                }
                for prediction in system.predictions
            ]
        )
        for key, value in {"any": 1, "majority": top_k // 2, "all": top_k}.items():
            gap = unresolved_instances(source, targets, threshold=value)
            data[f"performance_gap_{key}"] = data["instance_id"].apply(lambda instance_id: instance_id in gap)

        # Some systems have duplicated entries, which Zeno doesn't like.
        if len(data["instance_id"].unique()) != len(data["instance_id"]):
            data.drop_duplicates("instance_id", inplace=True)

        viz_project.upload_system(
            data,
            name=name,
            id_column="instance_id",
            output_column="output",
        )


if __name__ == "__main__":
    cli()
