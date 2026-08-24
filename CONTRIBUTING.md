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

The same checks run on your PR through the test workflow; see
[docs/github-workflows.md](docs/github-workflows.md) for what CI does with them
and how releases are cut.

## Ground rules

- **Tests required.** Every feature or fix ships with tests — follow the
  patterns in the existing `open_kgo/.../tests/` trees.
- **No-Docker policy.** Connector tests run against in-memory libraries
  (rdflib, networkx, kuzu) or file fixtures — no Docker, no network, no
  external services. See `open_kgo/feature_groups/kg/README.md`.
- **Conventional Commits.** Use `feat:` / `fix:` / `chore:` / `docs:` / etc.;
  release tooling parses them. No `Co-Authored-By` / AI-agent trailers.

## Reporting issues

There is a single **Issue** form: it covers bugs, feature requests, and small
tasks alike. Blank issues are disabled for anyone without write access, so
that form is the way in.

It asks for:

- **Summary**: one sentence describing the bug, request, or task.
- **Reproduction or motivation**: minimal steps plus expected vs. actual for
  bugs; motivation, current workaround, and desired behavior for features and
  tasks.
- **Code pointers** (optional): `file:line` references, so a newcomer can find
  the starting point quickly.
- **Definition of done** (optional): behavior, tests, and docs needed to call
  it complete.
- **Environment** (optional, bugs only): Python version, OS, mloda version.

Security vulnerabilities are the exception: report those privately per
[SECURITY.md](SECURITY.md), not through the public Issue form.

For questions that are not issues, the issue template chooser also links the
[mloda documentation](https://mloda-ai.github.io/mloda/) and the
[mloda framework](https://github.com/mloda-ai/mloda) repository, which is the
place to start a broader conversation.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License, Version 2.0](LICENSE).
