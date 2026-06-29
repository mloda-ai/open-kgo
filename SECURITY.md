<!--
NOTE: This policy is generic to every mloda plugin, not specific to open-kgo.
SECURITY.md is currently absent from mloda, mloda-registry, and
mloda-plugin-template. The durable fix is a plugin-facing default in the
template that survives scaffolding (tracked upstream); until then this lives here.
-->

# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public
GitHub issue. Either:

- Email **security@mloda.ai** with a description, affected version, and
  reproduction steps; or
- Use GitHub's private vulnerability reporting: the repo **Security** tab →
  **"Report a vulnerability"**.

We aim to acknowledge reports within a few business days and will keep you
updated on remediation. Once a fix is released, we're happy to credit you.

## Supported versions

open-kgo is pre-1.0; security fixes land on the latest release only.

## Scope

This project ships connectors that run against in-memory libraries and local
file fixtures (no Docker, no network by policy). Reports about dependency CVEs
(surfaced by GitHub's Dependabot alerts or an on-demand `tox -e security` /
`pip-audit` run) are welcome but are report-only and not release-blocking.
