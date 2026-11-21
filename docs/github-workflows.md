# GitHub Workflows

This project uses three GitHub Actions workflows to automate testing and releases.

Dependency CVE scanning is left to GitHub's native Dependabot security alerts (enable them under **Settings > Code security**), with `tox -e security` (`pip-audit`) available for on-demand local scans. Pulling later template changes is a deliberate, manual step (fetch the template remote and merge), not an automated job.

## Test Workflow

**File:** `.github/workflows/test.yml`

**Triggers:**
- Push to any branch
- Pull request to `main`

**Jobs:** Two, each across the same Python matrix (3.11, 3.12, 3.13, 3.14).

- **`test`** runs `tox -e py<version>` (e.g. `py312`), the default env's
  checks pinned to the leg's Python: pytest (excluding the `notebooks`
  marker), ruff formatting and linting, mypy type checking, and bandit security
  analysis. Locally, plain `tox` runs the default `py311` env.
- **`notebooks`** runs `tox -e notebooks` with `base_python` set to the leg's
  Python. It executes the demo marimo notebooks end-to-end (`pytest -m notebooks`). That env additionally installs
  the `demo` extra, which pulls in marimo. End-to-end execution is slow, so it
  is kept out of the default `tox` run to keep local iteration and the main
  matrix fast, but it is its own CI job, so the notebook suite still gates
  pull requests.

**Requirements:** None. This workflow uses only public GitHub Actions and requires no secrets.

## Release Workflow

**File:** `.github/workflows/release.yaml`

**Triggers:**
- Manual dispatch only

**Purpose:** Automates semantic versioning and publishing. The workflow:
1. Analyzes commits using [semantic-release](https://semantic-release.gitbook.io/) to determine the next version
2. Creates a GitHub release with changelog
3. Updates version in `pyproject.toml`
4. Builds the package once from the release tag and publishes it to TestPyPI
5. Installs the wheel TestPyPI serves and imports it (smoke test)
6. Attaches the smoke-tested files to the GitHub release

**Requirements:**
- `SEMANTIC_RELEASE_TOKEN` - GitHub Personal Access Token with `repo` write permissions
- TestPyPI Trusted Publisher (no secret), see below

**Prerequisites:** Commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) format for semantic-release to determine version bumps:
- `fix:` - Patch release (0.0.x)
- `feat:` - Minor release (0.x.0)
- `feat!:` or `BREAKING CHANGE:` - Major release (x.0.0)

## Publish to PyPI Workflow

**File:** `.github/workflows/publish-pypi.yaml`

**Triggers:**
- Manual dispatch from `main`, with the released `version` as input

**Purpose:** Downloads the files the Release workflow attached to the GitHub release, checks they match what TestPyPI serves, and uploads them to PyPI. Nothing reaches PyPI without passing the TestPyPI smoke test first.

**Requirements:**
- PyPI Trusted Publisher (no secret), see below

## Setting Up Secrets

To configure the required secrets for the release workflow:

### SEMANTIC_RELEASE_TOKEN

1. Go to GitHub **Settings** > **Developer settings** > **Personal access tokens** > **Tokens (classic)**
2. Click **Generate new token (classic)**
3. Give it a descriptive name (e.g., "semantic-release")
4. Select the `repo` scope (full control of private repositories)
5. Click **Generate token** and copy the token
6. In your repository, go to **Settings** > **Secrets and variables** > **Actions**
7. Click **New repository secret**
8. Name: `SEMANTIC_RELEASE_TOKEN`
9. Value: Paste the token
10. Click **Add secret**

### Trusted Publishing (PyPI and TestPyPI)

Uploads use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no API token is stored.

1. In the repository, go to **Settings** > **Environments** and create `testpypi` and `pypi` (optionally restrict both to the `main` branch and add required reviewers to `pypi`)
2. On [TestPyPI](https://test.pypi.org/manage/project/open-kgo/settings/publishing/) and [PyPI](https://pypi.org/manage/project/open-kgo/settings/publishing/), add a GitHub publisher with owner `mloda-ai` and repository `open-kgo`: on TestPyPI workflow `release.yaml` with environment `testpypi`, on PyPI workflow `publish-pypi.yaml` with environment `pypi`
3. Delete the old `PYPI_API_TOKEN` repository secret and revoke the token on PyPI
