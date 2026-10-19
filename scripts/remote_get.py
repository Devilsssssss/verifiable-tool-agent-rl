"""Download a file from the training server via SFTP (paramiko).

Usage:
  python scripts/remote_get.py <remote_path> <local_path>

Credentials come from the QKT_SERVER_PASSWORD environment variable.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import paramiko


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: remote_get.py <remote> <local>", file=sys.stderr)
        return 2
    remote, local = sys.argv[1], sys.argv[2]
    password = os.environ.get("QKT_SERVER_PASSWORD")
    if not password:
        print("ERROR: set QKT_SERVER_PASSWORD first", file=sys.stderr)
        return 2
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "10.199.161.82",
        port=22,
        username="qianketong",
        password=password,
        timeout=15,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    sftp.get(remote, local)
    sftp.close()
    client.close()
    print(f"downloaded {remote} -> {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
