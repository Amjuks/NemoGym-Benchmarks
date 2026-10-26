# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision only our benchmark's API credentials from private local configuration."""

import argparse
import base64
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("env.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.env_file)
    credentials = {}
    for field, key in (("policy_api_key", "policy-api-key"), ("judge_api_key", "judge-api-key")):
        value = config[field]
        if not isinstance(value, str) or not value.strip() or value == "???":
            raise ValueError(f"Provide a nonempty {field} in {args.env_file}")
        credentials[key] = base64.b64encode(value.encode()).decode()
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "aman-owasp-api-keys",
            "namespace": "aman-owasp-bench",
            "labels": {"app.kubernetes.io/part-of": "aman-owasp-bench"},
        },
        "type": "Opaque",
        "data": credentials,
    }
    command = ["kubectl", "apply", "--server-side", "--field-manager=aman-owasp", "-f", "-"]
    if args.dry_run:
        command = ["kubectl", "apply", "--dry-run=client", "-o", "name", "-f", "-"]
    # Credentials go through stdin, never command-line arguments or a temporary manifest.
    subprocess.run(command, input=json.dumps(secret), text=True, check=True)


if __name__ == "__main__":
    main()
