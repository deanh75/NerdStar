#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -eu

# Needed for vscode, otherwise the debugger is not started
echo "Listening on port"

qemu-aarch64 "$@"
