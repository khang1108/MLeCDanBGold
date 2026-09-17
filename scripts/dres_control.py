#!/usr/bin/env python3
"""DRES Evaluation Operator Control CLI.

Use this script to inspect, start, switch, and advance tasks in the active DRES
evaluation run without having to open the DRES web admin panel manually.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control active DRES evaluation tasks")
    parser.add_argument(
        "action",
        choices=["status", "start", "next", "previous", "switch", "list", "terminate"],
        help="Action to perform",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="Task index (0-based) or task name for 'switch'",
    )
    parser.add_argument(
        "--dres-url",
        default=os.getenv("HCMAI_DRES_BASE_URL", "https://dres.iamphuckhang.dev"),
        help="DRES root URL",
    )
    parser.add_argument(
        "--evaluation-id",
        default=os.getenv("HCMAI_DRES_EVALUATION_ID", ""),
        help="DRES Evaluation ID",
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="DRES admin username",
    )
    parser.add_argument(
        "--password",
        default="admin1234",
        help="DRES admin password",
    )
    return parser.parse_args()


def get_client(base_url: str, u: str, p: str) -> tuple[httpx.Client, dict[str, Any], str]:
    """Login and return authenticated client and session."""
    clean_base = base_url.rstrip("/").removesuffix("/api/v2")
    api_url = f"{clean_base}/api/v2"
    client = httpx.Client(timeout=15.0, verify=False)
    resp = client.post(f"{api_url}/login", json={"username": u, "password": p})
    if resp.status_code != 200:
        print(f"Failed to login to DRES: {resp.text}", file=sys.stderr)
        sys.exit(1)
    user = resp.json()
    return client, user, api_url


def resolve_eval_id(client: httpx.Client, api_url: str, eval_id: str) -> str:
    """Ensure evaluation ID is valid and active."""
    evals = client.get(f"{api_url}/evaluation/info/list").json()
    active = [e for e in evals if e.get("status") == "ACTIVE"]

    if eval_id:
        match = next((e for e in evals if e["id"] == eval_id), None)
        if match and match.get("status") == "ACTIVE":
            return eval_id
        if match:
            print(f"Warning: Evaluation {eval_id} is in status '{match.get('status')}'.")

    if len(active) == 1:
        return active[0]["id"]
    elif len(active) > 1:
        print(f"Multiple active evaluations found: {[e['id'] for e in active]}")
        return active[0]["id"]
    else:
        print("Error: No ACTIVE evaluation run found on DRES.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    client, user, api_url = get_client(args.dres_url, args.username, args.password)
    eval_id = resolve_eval_id(client, api_url, args.evaluation_id)

    # Fetch evaluation details
    eval_info_resp = client.get(f"{api_url}/evaluation/admin/{eval_id}")
    if eval_info_resp.status_code != 200:
        print(f"Error fetching evaluation info: {eval_info_resp.text}", file=sys.stderr)
        sys.exit(1)
    eval_data = eval_info_resp.json()
    template = eval_data.get("template", {})
    task_templates = template.get("tasks", [])

    # Fetch current evaluation state
    state_resp = client.get(f"{api_url}/evaluation/{eval_id}/state")
    state = state_resp.json() if state_resp.status_code == 200 else {}

    def ensure_task_stopped():
        s_resp = client.get(f"{api_url}/evaluation/{eval_id}/state")
        if s_resp.status_code == 200 and s_resp.json().get("taskStatus") == "RUNNING":
            client.post(f"{api_url}/evaluation/admin/{eval_id}/task/abort")

    action = args.action

    if action == "status":
        print("=" * 60)
        print(f" DRES EVALUATION STATUS: {eval_data.get('name')}")
        print("=" * 60)
        print(f"Evaluation ID:   {eval_id}")
        print(f"Evaluation Type: {eval_data.get('type')}")
        print(f"Run Status:      {eval_data.get('status') or state.get('evaluationStatus')}")
        print(f"Active Task ID:  {state.get('taskId')}")
        print(f"Task Status:     {state.get('taskStatus')}")
        print(f"Time Left:       {state.get('timeLeft')}s")

        cur_tmpl_id = state.get("taskTemplateId")
        cur_idx = None
        for i, t in enumerate(task_templates):
            if t.get("id") == cur_tmpl_id:
                cur_idx = i
                print(f"Current Task:    [{i}] {t.get('name')} (Duration: {t.get('duration')}s)")
                break

        print(f"Total Tasks:     {len(task_templates)}")
        print("=" * 60)

    elif action == "list":
        print("=" * 60)
        print(f" TASKS LIST FOR EVALUATION: {eval_data.get('name')}")
        print("=" * 60)
        cur_tmpl_id = state.get("taskTemplateId")
        for i, t in enumerate(task_templates):
            marker = " -> " if t.get("id") == cur_tmpl_id else "    "
            print(f"{marker}[{i:02d}] {t.get('name'):<20} Duration: {t.get('duration')}s  ({t.get('comment', '')})")
        print("=" * 60)

    elif action == "start":
        print(f"Starting current task for evaluation {eval_id}...")
        resp = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/start")
        if resp.status_code == 200:
            print("Successfully started task! Status is now RUNNING.")
        else:
            print(f"Failed to start task: HTTP {resp.status_code} - {resp.text}")

    elif action == "next":
        ensure_task_stopped()
        print(f"Moving to next task for evaluation {eval_id}...")
        resp = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/next")
        if resp.status_code == 200:
            print("Successfully moved to next task.")
            start_r = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/start")
            if start_r.status_code == 200:
                print("Task started and accepting submissions!")
            else:
                print(f"Note: Start task returned: {start_r.text}")
        else:
            print(f"Failed to move to next task: HTTP {resp.status_code} - {resp.text}")

    elif action == "previous":
        ensure_task_stopped()
        print(f"Moving to previous task for evaluation {eval_id}...")
        resp = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/previous")
        if resp.status_code == 200:
            print("Successfully moved to previous task.")
            start_r = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/start")
            if start_r.status_code == 200:
                print("Task started and accepting submissions!")
        else:
            print(f"Failed to move to previous task: HTTP {resp.status_code} - {resp.text}")

    elif action == "switch":
        target = args.target.strip()
        if not target:
            print("Error: Please specify task index or name to switch to. Example: switch 2", file=sys.stderr)
            sys.exit(1)

        target_idx = None
        if target.isdigit():
            target_idx = int(target)
        else:
            for i, t in enumerate(task_templates):
                if t.get("name") == target:
                    target_idx = i
                    break

        if target_idx is None or target_idx < 0 or target_idx >= len(task_templates):
            print(f"Error: Could not find task '{target}'. Use 'list' to view available tasks.", file=sys.stderr)
            sys.exit(1)

        ensure_task_stopped()
        task_name = task_templates[target_idx].get("name")
        print(f"Switching to task [{target_idx}] '{task_name}'...")
        resp = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/switch/{target_idx}")
        if resp.status_code == 200:
            print(f"Successfully switched to task [{target_idx}].")
            start_r = client.post(f"{api_url}/evaluation/admin/{eval_id}/task/start")
            if start_r.status_code == 200:
                print(f"Task '{task_name}' is now RUNNING and accepting submissions!")
            else:
                print(f"Note: Start task returned: {start_r.text}")
        else:
            print(f"Failed to switch task: HTTP {resp.status_code} - {resp.text}")

    elif action == "terminate":
        print(f"Terminating evaluation {eval_id}...")
        resp = client.post(f"{api_url}/evaluation/admin/{eval_id}/terminate")
        print("Response:", resp.status_code, resp.text)


if __name__ == "__main__":
    main()
