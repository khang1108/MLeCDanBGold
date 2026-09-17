#!/usr/bin/env python3
"""Import groundtruth evaluation files into DRES server.

This script converts HCMAI evaluation groundtruth JSON files (e.g.,
artifacts/evaluation/query_001_windows.json) into a DRES MediaCollection,
EvaluationTemplate, and active Evaluation Run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import groundtruth evaluation into DRES")
    parser.add_argument(
        "--input",
        "-i",
        default="artifacts/evaluation/query_001_windows.json",
        help="Path to evaluation JSON file with groundtruth (default: artifacts/evaluation/query_001_windows.json)",
    )
    parser.add_argument(
        "--dres-url",
        default="https://dres.iamphuckhang.dev/api/v2",
        help="DRES API v2 base URL (default: https://dres.iamphuckhang.dev/api/v2)",
    )
    parser.add_argument(
        "--username",
        "-u",
        default="admin",
        help="DRES admin username (default: admin)",
    )
    parser.add_argument(
        "--password",
        "-p",
        default="admin1234",
        help="DRES admin password (default: admin1234)",
    )
    parser.add_argument(
        "--ssh-host",
        default="azureuser@20.43.83.34",
        help="SSH user@host for Azure VM to create video and cache stubs (default: azureuser@20.43.83.34)",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Template and Evaluation run name (default: derived from input file)",
    )
    parser.add_argument(
        "--task-duration",
        type=int,
        default=86400,
        help="Duration in seconds per task (default: 86400 / 24 hours so tasks stay active for testing)",
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        default=True,
        help="Automatically update HCMAI_DRES_EVALUATION_ID in .env (default: True)",
    )
    return parser.parse_args()


def load_media_info() -> dict[str, dict[str, Any]]:
    """Load video metadata from artifacts/media-info if present."""
    media_info = {}
    for p in glob.glob("artifacts/media-info/*.json"):
        video_id = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, encoding="utf-8") as f:
                media_info[video_id] = json.load(f)
        except Exception:
            pass
    return media_info


def update_env_file(eval_id: str, base_url: str) -> None:
    """Update HCMAI_DRES_EVALUATION_ID and HCMAI_DRES_BASE_URL in .env."""
    env_path = ".env"
    if not os.path.exists(env_path):
        print(f"Note: {env_path} not found, skipping update.")
        return

    # HCMAI DresClient appends /api/v2 to routes, so root URL is needed
    clean_base_url = base_url.removesuffix("/api/v2").removesuffix("/")

    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    has_eval_id = False
    new_lines = []
    for line in lines:
        if line.startswith("HCMAI_DRES_EVALUATION_ID="):
            new_lines.append(f"HCMAI_DRES_EVALUATION_ID={eval_id}\n")
            has_eval_id = True
        elif line.startswith("HCMAI_DRES_BASE_URL="):
            new_lines.append(f"HCMAI_DRES_BASE_URL={clean_base_url}\n")
        else:
            new_lines.append(line)

    if not has_eval_id:
        # Find where DRES section is or append at end
        inserted = False
        for i, line in enumerate(new_lines):
            if line.startswith("HCMAI_DRES_BASE_URL="):
                new_lines.insert(i + 1, f"HCMAI_DRES_EVALUATION_ID={eval_id}\n")
                inserted = True
                break
        if not inserted:
            new_lines.append(f"HCMAI_DRES_EVALUATION_ID={eval_id}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Updated .env with HCMAI_DRES_EVALUATION_ID={eval_id}")


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    cases = eval_data.get("cases", [])
    if not cases:
        print(f"Error: No cases found in {input_path}", file=sys.stderr)
        sys.exit(1)

    template_name = args.name or f"HCMAI_{eval_data.get('split', '001')}_KIS"
    print(f"Loaded {len(cases)} cases from {input_path}")
    print(f"Target Template Name: {template_name}")

    # Step 1: Login to DRES
    base_url = args.dres_url.rstrip("/")
    client = httpx.Client(timeout=30.0, verify=False)

    login_resp = client.post(
        f"{base_url}/login",
        json={"username": args.username, "password": args.password},
    )
    if login_resp.status_code != 200:
        print(f"Failed to login to DRES: HTTP {login_resp.status_code} - {login_resp.text}", file=sys.stderr)
        sys.exit(1)

    admin_user = login_resp.json()
    print(f"Logged in as admin ({admin_user.get('id')})")

    # Step 2: Get or Create Media Collection
    collections_resp = client.get(f"{base_url}/collection/list")
    collection_id = None
    collection_name = "HCMAI_Videos"

    if collections_resp.status_code == 200:
        for col in collections_resp.json():
            if col.get("name") == collection_name:
                collection_id = col.get("id")
                break

    if not collection_id:
        print(f"Creating Media Collection '{collection_name}'...")
        client.post(
            f"{base_url}/collection",
            json={
                "name": collection_name,
                "description": "HCMAI Multimodal Video Collection",
                "basePath": "/opt/dres/data",
                "itemCount": 0,
            },
        )
        for col in client.get(f"{base_url}/collection/list").json():
            if col.get("name") == collection_name:
                collection_id = col.get("id")
                break

    print(f"Media Collection '{collection_name}' ID: {collection_id}")

    # Step 3: Ensure media items exist in collection
    needed_videos = set()
    for case in cases:
        for gt in case.get("ground_truth", []):
            needed_videos.add(gt["video_id"])

    media_info = load_media_info()
    print(f"Ground truth references {len(needed_videos)} unique videos.")

    # Check which videos already exist in collection
    resolve_resp = client.post(
        f"{base_url}/collection/{collection_id}/resolve",
        json=list(needed_videos),
    )
    existing_items = resolve_resp.json() if resolve_resp.status_code == 200 else []
    video_to_item = {it["name"]: it for it in existing_items}

    # Add missing videos
    added_count = 0
    for vid in needed_videos:
        if vid not in video_to_item:
            info = media_info.get(vid, {})
            duration_s = info.get("length", 1800)
            duration_ms = int(duration_s * 1000)
            item_payload = {
                "mediaItemId": "",
                "name": vid,
                "type": "VIDEO",
                "collectionId": collection_id,
                "location": f"{vid}.mp4",
                "durationMs": duration_ms,
                "fps": 25.0,
                "metadata": [],
            }
            add_resp = client.post(f"{base_url}/mediaItem", json=item_payload)
            if add_resp.status_code == 200:
                added_count += 1
            else:
                print(f"Warning: Failed to add video {vid}: {add_resp.text}")

    if added_count > 0:
        print(f"Registered {added_count} new media items into '{collection_name}'.")

    # Re-resolve to get all item UUIDs
    resolve_resp = client.post(
        f"{base_url}/collection/{collection_id}/resolve",
        json=list(needed_videos),
    )
    video_to_item = {it["name"]: it for it in resolve_resp.json()}
    print(f"Resolved {len(video_to_item)}/{len(needed_videos)} video UUIDs in collection.")

    # Step 4: Build tasks and prepare target ranges
    task_type = {
        "name": "Textual Known Item Search",
        "duration": args.task_duration,
        "targetOption": "SINGLE_MEDIA_SEGMENT",
        "hintOptions": ["TEXT"],
        "submissionOptions": [
            "NO_DUPLICATES",
            "LIMIT_CORRECT_PER_TEAM",
            "TEMPORAL_SUBMISSION",
        ],
        "taskOptions": ["HIDDEN_RESULTS"],
        "scoreOption": "KIS",
        "configuration": {"LIMIT_CORRECT_PER_TEAM.limit": "1"},
    }

    task_group = {
        "id": None,
        "name": "KIS",
        "type": "Textual Known Item Search",
    }

    # Fetch all registered DRES users to populate teams
    all_users = client.get(f"{base_url}/user/list").json()
    teams = []

    # Map participant users into teams
    p_users = [u for u in all_users if u.get("role") in ("PARTICIPANT", "ADMIN")]
    if p_users:
        # Create team with all participants
        teams.append({
            "id": None,
            "name": "HCMAI_Competitors",
            "color": "#2ecc71",
            "users": [{"id": u["id"], "username": u["username"], "role": u["role"]} for u in p_users],
            "logoData": None,
        })
    else:
        teams.append({
            "id": None,
            "name": "HCMAI_Team",
            "color": "#2ecc71",
            "users": [{"id": admin_user["id"], "username": admin_user["username"], "role": admin_user["role"]}],
            "logoData": None,
        })

    tasks = []
    stub_files = []

    for idx, case in enumerate(cases):
        query_id = case.get("query_id", f"task-{idx+1}")
        raw_query = case.get("raw_query", "")

        targets = []
        for gt in case.get("ground_truth", []):
            vid = gt.get("video_id")
            media_item = video_to_item.get(vid)
            if not media_item:
                print(f"Warning: Video {vid} not found in collection, skipping target.")
                continue

            item_id = media_item["mediaItemId"]
            time_windows = gt.get("time_windows_ms", [])
            start_ms = time_windows[0][0] if time_windows else 0
            end_ms = time_windows[0][1] if time_windows else 0
            duration_ms = media_item.get("durationMs") or 1800000
            clamped_end_ms = min(end_ms, duration_ms) if duration_ms else end_ms

            targets.append({
                "type": "MEDIA_ITEM_TEMPORAL_RANGE",
                "target": item_id,
                "range": {
                    "start": {"value": str(start_ms), "unit": "MILLISECONDS"},
                    "end": {"value": str(end_ms), "unit": "MILLISECONDS"},
                },
            })

            # Video file stub and cache preview stub
            stub_files.append(f"/opt/dres/data/{vid}.mp4")
            stub_files.append(f"/opt/dres/dist/dres-dist/lib/cache/{item_id}_{start_ms}-{clamped_end_ms}.mp4")

        hints = [
            {
                "type": "TEXT",
                "start": 0,
                "end": args.task_duration,
                "description": raw_query,
            }
        ]

        tasks.append({
            "id": None,
            "name": query_id,
            "taskGroup": "KIS",
            "taskType": "Textual Known Item Search",
            "duration": args.task_duration,
            "collectionId": collection_id,
            "targets": targets,
            "hints": hints,
            "comment": f"Generated from {case.get('source_query_file', '')}",
        })

    # Step 5: Create dummy video and cache stubs on server via SSH
    if args.ssh_host and stub_files:
        print(f"Preparing {len(stub_files)} server file and preview cache stubs via SSH...")
        unique_stubs = sorted(list(set(stub_files)))
        # Batch commands
        touch_cmd = "sudo mkdir -p /opt/dres/data /opt/dres/dist/dres-dist/lib/cache && sudo touch " + " ".join(unique_stubs)
        try:
            ssh_proc = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", args.ssh_host, touch_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if ssh_proc.returncode == 0:
                print("Server file and cache stubs successfully prepared.")
            else:
                print(f"Warning: SSH stub creation returned {ssh_proc.returncode}: {ssh_proc.stderr}")
        except Exception as e:
            print(f"Warning: Could not connect to SSH host: {e}")

    # Step 6: Create or update Evaluation Template
    # Clean up existing template with same name if exists
    existing_tmpls = client.get(f"{base_url}/template/list").json()
    for t in existing_tmpls:
        if t.get("name") == template_name:
            print(f"Cleaning up previous template '{template_name}' ({t['id']})...")
            client.delete(f"{base_url}/template/{t['id']}")

    print(f"Creating evaluation template '{template_name}'...")
    create_t_resp = client.post(
        f"{base_url}/template",
        json={"name": template_name, "description": f"Imported from {os.path.basename(input_path)} with {len(tasks)} tasks"},
    )
    if create_t_resp.status_code != 200:
        print(f"Failed to create template: {create_t_resp.text}", file=sys.stderr)
        sys.exit(1)

    # Get created template ID
    updated_tmpls = client.get(f"{base_url}/template/list").json()
    template_id = next(t["id"] for t in updated_tmpls if t["name"] == template_name)
    print(f"Created Template ID: {template_id}")

    # Fetch template object to update
    tmpl_obj = client.get(f"{base_url}/template/{template_id}").json()
    tmpl_obj["taskTypes"] = [task_type]
    tmpl_obj["taskGroups"] = [task_group]
    tmpl_obj["teams"] = teams
    tmpl_obj["tasks"] = tasks

    print(f"Uploading {len(tasks)} tasks to template...")
    patch_resp = client.patch(f"{base_url}/template/{template_id}", json=tmpl_obj)
    if patch_resp.status_code != 200:
        print(f"Failed to update template: HTTP {patch_resp.status_code} - {patch_resp.text}", file=sys.stderr)
        sys.exit(1)
    print("Template tasks successfully configured!")

    # Step 7: Create and Start Evaluation Run
    run_name = f"{template_name}_Run"
    print(f"Creating Evaluation Run '{run_name}'...")

    # Terminate any existing run with this name
    info_list_resp = client.get(f"{base_url}/evaluation/info/list")
    if info_list_resp.status_code == 200:
        for ev in info_list_resp.json():
            if ev.get("name") == run_name and ev.get("status") in ("ACTIVE", "CREATED"):
                print(f"Stopping existing run {ev['id']}...")
                client.post(f"{base_url}/evaluation/admin/{ev['id']}/terminate")

    run_payload = {
        "templateId": template_id,
        "name": run_name,
        "type": "SYNCHRONOUS",
        "properties": {
            "participantCanView": True,
            "shuffleTasks": False,
            "allowRepeatedTasks": True,
            "limitSubmissionPreviews": -1,
        },
    }

    create_run_resp = client.post(
        f"{base_url}/evaluation/admin/create",
        json=run_payload,
    )
    if create_run_resp.status_code != 200:
        print(f"Failed to create evaluation run: HTTP {create_run_resp.status_code} - {create_run_resp.text}", file=sys.stderr)
        sys.exit(1)

    # Find the created evaluation ID from info list
    active_evals = client.get(f"{base_url}/evaluation/info/list").json()
    evaluation_id = None
    for ev in active_evals:
        if ev.get("name") == run_name:
            evaluation_id = ev.get("id")
            break

    if not evaluation_id:
        print("Error: Could not locate created evaluation ID.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluation Run ID: {evaluation_id}")

    # Start evaluation run
    start_resp = client.post(f"{base_url}/evaluation/admin/{evaluation_id}/start")
    if start_resp.status_code == 200:
        print(f"Evaluation Run '{run_name}' is now ACTIVE!")
    else:
        print(f"Start evaluation returned HTTP {start_resp.status_code}: {start_resp.text}")

    # Start the first task
    start_task_resp = client.post(f"{base_url}/evaluation/admin/{evaluation_id}/task/start")
    if start_task_resp.status_code == 200:
        print("Task 1 is now RUNNING and accepting submissions!")
    else:
        print(f"Start task returned: {start_task_resp.text}")

    # Step 8: Update .env
    if args.update_env:
        update_env_file(evaluation_id, base_url)

    print("\n========================================================")
    print(" DRES EVALUATION IMPORTED & RUNNING SUCCESSFULLY!")
    print(f" Template:       {template_name} ({template_id})")
    print(f" Evaluation Run: {run_name} ({evaluation_id})")
    print(f" Tasks Count:    {len(tasks)}")
    print(f" Base URL:       {base_url}")
    print(f" Web UI:         https://dres.iamphuckhang.dev/#/client/run/{evaluation_id}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
