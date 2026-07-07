<!--
SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

This folder contains configuration for a Visual Studio Code Dev Container
https://code.visualstudio.com/docs/devcontainers/containers.

Please note that this dev container configuration is not intended to
form part of the KleidiCV product. The dev container configuration is
only provided for development convenience and is not held to the same
quality standards as the rest of KleidiCV.

To use this Dev Container you must first log in to the Arm GitLab
Docker registry. Instructions are at
https://docs.gitlab.com/ee/user/packages/container_registry/authenticate_with_container_registry.html
In summary:
1. In the GitLab UI create an access token.
2. Run `docker login registry.gitlab.arm.com -u FirstnameLastname`
3. Paste the access token when prompted.
