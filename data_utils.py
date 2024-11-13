import json
import os
import re

import pandas as pd

from models import SWEBenchPath, SWEBenchInstance


def extract_conversation(history):
    conversation = []
    if isinstance(history, list):
        for step in history:
            if isinstance(step, dict):
                # Handle the case where each step is a dictionary
                if step.get("source") == "user":
                    conversation.append(
                        {"role": "user", "content": step.get("message", "")}
                    )
                elif step.get("source") == "agent":
                    conversation.append(
                        {"role": "assistant", "content": step.get("message", "")}
                    )
            elif isinstance(step, list) and len(step) == 2:
                # Handle the case where each step is a list of two elements
                source, message = step
                if isinstance(source, dict) and "source" in source:
                    if source["source"] == "user":
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    message.get("message", "")
                                    if isinstance(message, dict)
                                    else str(message)
                                ),
                            }
                        )
                    elif source["source"] == "agent":
                        conversation.append(
                            {
                                "role": "assistant",
                                "content": (
                                    message.get("message", "")
                                    if isinstance(message, dict)
                                    else str(message)
                                ),
                            }
                        )
    return conversation


def load_swe_bench_dataset(benchmark_path: SWEBenchPath) -> pd.DataFrame:
    """
    Load the SWE-bench dataset from an OpenHands-generated output trajectory file.

    The resulting dataframe can be passed to ZenoML projects as a dataset.
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


def load_swe_bench_trajectories(benchmark_path: SWEBenchPath) -> pd.DataFrame:
    """
    ...
    """

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
                        "history": extract_conversation(data["history"]),
                        "resolved": instance.test_result.report.resolved,
                        "git_patch": instance.test_result.git_patch
                    }
                ]
            )
            trajectories.append(trajectory)

    return pd.concat(trajectories, ignore_index=True)


def load_data(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).split(".")[0]  # Get the 'test' part
    print("Directory name: ", directory_name)
    print("Base name: ", base_name)

    # Construct paths for report.json and .md file
    report_json_path = os.path.join(directory_name, f"{base_name}.swebench_eval.jsonl")
    report_md_path = os.path.join(directory_name, f"{base_name}.swebench_eval.md")
    resolved_map = {}

    # Try to load report.json first
    if os.path.exists(report_json_path):
        with open(report_json_path, "r") as report_file:
            for line in report_file:
                instance = SWEBenchInstance.model_validate_json(line)
                resolved_map[instance.instance_id] = (
                    instance.test_result.report.resolved
                )
    elif os.path.exists(report_md_path):
        # If report.json doesn't exist, parse the markdown file
        with open(report_md_path, "r") as md_file:
            content = md_file.read()
            resolved_instances = re.findall(
                r"- \[(.*?)\]", content.split("## Resolved Instances")[1].split("##")[0]
            )
            for instance in resolved_instances:
                resolved_map[instance] = True
    else:
        print(f"Warning: No report file found for {base_name}")

    # Load conversation data
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            problem_statement = data.get("instance", {}).get("problem_statement")

            # Get resolved status from the report.json or markdown file
            resolved = 1 if resolved_map.get(instance_id, False) else 0

            # Extract conversation history, ensure it's a list of dictionaries
            conversation = extract_conversation(data.get("history", []))
            if not isinstance(conversation, list):
                conversation = [{"role": "assistant", "content": str(conversation)}]
            else:
                conversation = [
                    (
                        msg
                        if isinstance(msg, dict)
                        else {"role": "assistant", "content": str(msg)}
                    )
                    for msg in conversation
                ]

            # Append instance data with resolved status
            data_list.append((instance_id, problem_statement, resolved, conversation))

    return data_list


def load_data_aider_bench(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    print("Directory name: ", directory_name)
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            test_result = data.get("test_result", {})
            resolved = (
                1
                if test_result.get("exit_code") == 0
                and bool(re.fullmatch(r"\.+", test_result.get("test_cases")))
                else 0
            )
            test_cases = test_result.get("test_cases")
            instruction = data.get("instruction")
            agent_trajectory = []
            for step in data.get("history", []):
                if step[0]["source"] != "agent":
                    continue
                agent_trajectory.append(
                    {
                        "action": step[0].get("action"),
                        "code": step[0].get("args", {}).get("code"),
                        "thought": step[0].get("args", {}).get("thought"),
                        "observation": step[1].get("message"),
                    }
                )
            data_list.append(
                (instance_id, instruction, resolved, test_cases, agent_trajectory)
            )

    return data_list


def get_model_name_aider_bench(file_path):
    with open(file_path, "r") as file:
        first_line = file.readline()
        data = json.loads(first_line)
        return (
            data.get("metadata", {}).get("llm_config", {}).get("model").split("/")[-1]
        )
