<!--
NOTE: This guide is generic to every mloda plugin, not specific to open-kgo.
The mloda-plugin-template ships a CONTRIBUTING.md that is template-only and is
deleted on "Use this template", with no plugin-facing replacement — so every
generated plugin starts without one. The durable fix is a plugin-facing
community-health default in the template that survives scaffolding (tracked
upstream); until then this lives here.
-->

# Contributing to open-kgo

Thanks for your interest! open-kgo is a knowledge-graph connector plugin for
[mloda](https://github.com/mloda-ai/mloda).

## Development setup

```bash
uv venv && source .venv/bin/activate
uv sync --all-extras
```

## Before you open a PR

`tox` is the merge gate. It must pass:

```bash
tox
```

It runs pytest, `ruff format --check`, `ruff check`, `mypy --strict`, and bandit.

## Ground rules

- **Tests required.** Every feature or fix ships with tests — follow the
  patterns in the existing `open_kgo/.../tests/` trees.
- **No-Docker policy.** Connector tests run against in-memory libraries
  (rdflib, networkx, kuzu) or file fixtures — no Docker, no network, no
  external services. See `open_kgo/feature_groups/kg/README.md`.
- **Conventional Commits.** Use `feat:` / `fix:` / `chore:` / `docs:` / etc.;
  release tooling parses them. No `Co-Authored-By` / AI-agent trailers.

## Reporting issues

Use the issue template (Bug report / Feature request). One-sentence summary,
reproduction or motivation, code pointers (`file:line`), and a definition of
done if scoped.
