# Set up CI/CD with develop/main workflow

CI/CD pipeline with a two-branch workflow to keep our main branch and Docker images stable.

## Changes

### Branch Structure
- `develop`: All development PRs go here first
- `main`: Post-test merge destination

### CI Pipeline (on develop)
- Runs on Python 3.9-3.11
- Runs and Requires Ruff pass for code quality:
  - Basic Python errors (F)
  - Style conventions (E, W)
  - Import organization (I)
- Pytest & Docker build verification

### Release (happens on main)
- Builds and pushes Docker images:
  - GPU: `latest` and `v#.#.#`
  - CPU: `latest-cpu` and `v#.#.#-cpu`
  - GUI: `latest-gui` and `v#.#.#-gui`
- Auto creates GitHub releases

### Dependencies
- Updated numpy to 2.0.2 for Python 3.9+ compatibility
- Added test dependencies for coverage and async testing

### Documentation
- Added CONTRIBUTING.md with workflow explanation
- Added VERSION file for release tracking
