from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "thundercompute" / "launch.sh"


def make_auth_tnr(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "auth-bin"
    fake_bin.mkdir()
    fake_tnr = fake_bin / "tnr"
    fake_tnr.write_text(
        "#!/usr/bin/env bash\n"
        'printf "call\\n" >> "$TNR_CALLS"\n'
        '[[ "${TNR_LOGIN_ACTIVE:-false}" == "true" ]] && exit 0\n'
        '[[ "${TNR_API_TOKEN:-}" == "accepted-token" ]] && exit 0\n'
        "exit 1\n"
    )
    fake_tnr.chmod(0o700)
    return fake_bin


def run_auth_check(
    tmp_path: Path,
    *,
    login_active: bool = False,
    cli_token: str = "",
    env_token: str = "",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = make_auth_tnr(tmp_path)
    calls = tmp_path / "tnr-calls"
    command = f'source "{LAUNCHER}"; TOKEN="{cli_token}"; check_authentication'
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TNR_CALLS": str(calls),
        "TNR_LOGIN_ACTIVE": "true" if login_active else "false",
        "TNR_API_TOKEN": env_token,
    }
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    return result, calls


def test_authentication_uses_existing_tnr_login(tmp_path: Path) -> None:
    result, calls = run_auth_check(tmp_path, login_active=True)

    assert result.returncode == 0, result.stderr
    assert "already authenticated" in result.stdout
    assert calls.read_text().splitlines() == ["call"]


def test_authentication_uses_supplied_token_when_logged_out(tmp_path: Path) -> None:
    result, calls = run_auth_check(tmp_path, cli_token="accepted-token")

    assert result.returncode == 0, result.stderr
    assert "API token accepted" in result.stdout
    assert "accepted-token" not in result.stdout
    assert "accepted-token" not in result.stderr
    assert calls.read_text().splitlines() == ["call", "call"]


def test_authentication_uses_token_environment_variable(tmp_path: Path) -> None:
    result, calls = run_auth_check(tmp_path, env_token="accepted-token")

    assert result.returncode == 0, result.stderr
    assert "already authenticated" in result.stdout
    assert "accepted-token" not in result.stdout
    assert "accepted-token" not in result.stderr
    assert calls.read_text().splitlines() == ["call"]


def test_authentication_alerts_when_logged_out_without_token(tmp_path: Path) -> None:
    result, calls = run_auth_check(tmp_path)

    assert result.returncode != 0
    assert "Thunder CLI is not logged in" in result.stderr
    assert "--token TOKEN" in result.stderr
    assert "TNR_API_TOKEN" in result.stderr
    assert calls.read_text().splitlines() == ["call"]


def test_launcher_never_uses_interactive_tnr_connect() -> None:
    source = LAUNCHER.read_text()

    assert "tnr connect" not in source


def test_status_json_provides_noninteractive_connection_metadata() -> None:
    payload = [
        {
            "id": 0,
            "uuid": "otojoj1s",
            "status": "RUNNING",
            "ip": "216.81.200.237",
            "port": 32213,
        }
    ]
    command = f'source "{LAUNCHER}"; json_instance_connection 0'

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["otojoj1s", "216.81.200.237", "32213"]


def test_connection_resolution_uses_key_cached_by_tnr_scp(tmp_path: Path) -> None:
    fake_bin = tmp_path / "status-bin"
    fake_bin.mkdir()
    fake_tnr = fake_bin / "tnr"
    fake_tnr.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'[{\"id\":0,\"uuid\":\"otojoj1s\",\"status\":\"RUNNING\","
        "\"ip\":\"216.81.200.237\",\"port\":32213}]'\n"
    )
    fake_tnr.chmod(0o700)
    tnr_home = tmp_path / "thunder"
    key_dir = tnr_home / "keys"
    key_dir.mkdir(parents=True)
    key_file = key_dir / "otojoj1s"
    key_file.write_text("test key")
    connection_file = tmp_path / "connection"
    connection_file.touch()
    command = (
        f'source "{LAUNCHER}"; INSTANCE_ID=0; '
        'resolve_ssh_connection; '
        'printf "%s\\n" "$SSH_HOST" "$SSH_PORT" "$SSH_KEY_FILE"'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TNR_HOME": str(tnr_home),
        "HCMAI_THUNDER_CONNECTION_FILE": str(connection_file),
    }

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    expected = ["216.81.200.237", "32213", str(key_file)]
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected
    assert connection_file.read_text().splitlines() == expected


def test_remote_deployment_uses_noninteractive_ubuntu_ssh(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$SSH_ARGS_CAPTURE"\n'
    )
    fake_ssh.chmod(0o700)
    args_capture = tmp_path / "ssh-args"
    key_file = tmp_path / "thunder-key"
    key_file.write_text("test key")
    command = (
        f'source "{LAUNCHER}"; '
        'INSTANCE_ID="0"; '
        'SSH_HOST="216.81.200.237"; SSH_PORT="32213"; '
        f'SSH_KEY_FILE="{key_file}"; '
        'DEPLOY_ARGS=(--asr true); start_remote_tmux'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SSH_ARGS_CAPTURE": str(args_capture),
    }

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    ssh_args = args_capture.read_text().splitlines()
    assert "-T" in ssh_args
    assert "BatchMode=yes" in ssh_args
    assert "IdentitiesOnly=yes" in ssh_args
    assert "PasswordAuthentication=no" in ssh_args
    assert "KbdInteractiveAuthentication=no" in ssh_args
    assert "ubuntu@216.81.200.237" in ssh_args
    assert not any(argument.startswith("root@") for argument in ssh_args)
    assert "tmux new-session" in ssh_args[-1]
    assert "--asr\\ true" in ssh_args[-1]


def test_existing_instance_option_preserves_model_arguments() -> None:
    command = (
        f'source "{LAUNCHER}"; '
        'parse_args --instance 0 -- --ocr true --asr true --vqa true; '
        'printf "%s\\n" "$INSTANCE_ID" "${DEPLOY_ARGS[@]}"'
    )

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "0",
        "--ocr",
        "true",
        "--asr",
        "true",
        "--vqa",
        "true",
    ]


def test_cleanup_deletes_created_instance_and_clears_state(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "tnr-calls"
    fake_tnr = fake_bin / "tnr"
    fake_tnr.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$TNR_CALLS"\n'
    )
    fake_tnr.chmod(0o700)
    state_file = tmp_path / "instance-id"
    state_file.write_text("7\n")
    command = (
        f'source "{LAUNCHER}"; '
        'INSTANCE_ID="7"; INSTANCE_CREATED=true; DELETE_ON_EXIT=true; '
        f'INSTANCE_ID_FILE="{state_file}"; exit 0'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TNR_CALLS": str(calls),
    }

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ["delete --yes 7"]
    assert state_file.read_text() == ""


def test_upload_target_keeps_absolute_home_path(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "tnr-calls"
    fake_tnr = fake_bin / "tnr"
    fake_tnr.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$TNR_CALLS"\n'
    )
    fake_tnr.chmod(0o700)
    bootstrap = tmp_path / "deploy_cloudflared_private.sh"
    bootstrap.write_text("#!/usr/bin/env bash\n")
    command = (
        f'source "{LAUNCHER}"; '
        f'DEPLOY_SCRIPT="{bootstrap}"; INSTANCE_ID="0"; upload_bootstrap'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TNR_CALLS": str(calls),
    }

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        f"scp {bootstrap} 0:/home/ubuntu/ --yes"
    ]
