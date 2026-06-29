# GitHub Workflows

This project uses two GitHub Actions workflows to automate testing and releases.

Dependency CVE scanning is left to GitHub's native Dependabot security alerts (enable them under **Settings > Code security**), with `tox -e security` (`pip-audit`) available for on-demand local scans. Pulling later template changes is a deliberate, manual step (fetch the template remote and merge), not an automated job.

## Test Workflow

**File:** `.github/workflows/test.yml`

**Triggers:**
- Push to any branch
- Pull request to `main`

**Purpose:** Runs the full test suite using tox across multiple Python versions (3.10, 3.11, 3.12, 3.13). This includes pytest, ruff linting, mypy type checking, and bandit security analysis.

**Requirements:** None. This workflow uses only public GitHub Actions and requires no secrets.

## Release Workflow

**File:** `.github/workflows/release.yaml`

**Triggers:**
- Manual dispatch only

**Purpose:** Automates semantic versioning and publishing. The workflow:
1. Analyzes commits using [semantic-release](https://semantic-release.gitbook.io/) to determine the next version
2. Creates a GitHub release with changelog
3. Updates version in `pyproject.toml`
4. Builds and publishes the package to PyPI

**Requirements:**
- `SEMANTIC_RELEASE_TOKEN` - GitHub Personal Access Token with `repo` write permissions
- `PYPI_API_TOKEN` - PyPI API token for package publishing

**Prerequisites:** Commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) format for semantic-release to determine version bumps:
- `fix:` - Patch release (0.0.x)
- `feat:` - Minor release (0.x.0)
- `feat!:` or `BREAKING CHANGE:` - Major release (x.0.0)

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

### PYPI_API_TOKEN

1. Log in to [PyPI](https://pypi.org/)
2. Go to **Account settings** > **API tokens**
3. Click **Add API token**
4. Give it a descriptive name and scope it to your project (recommended) or all projects
5. Click **Create token** and copy the token
6. In your repository, go to **Settings** > **Secrets and variables** > **Actions**
7. Click **New repository secret**
8. Name: `PYPI_API_TOKEN`
9. Value: Paste the token (starts with `pypi-`)
10. Click **Add secret**
