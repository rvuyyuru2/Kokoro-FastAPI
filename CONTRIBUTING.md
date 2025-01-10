# Contributing to Kokoro-FastAPI
## Development Setup

1. Install Docker and Docker Compose
2. Clone the repository
3. Start the development environment:
```bash
# For GPU support
docker compose up

# For CPU-only
docker compose -f docker-compose.cpu.yml up
```

## Code Quality & Testing

Our CI pipeline runs these checks on your PR:

1. Code Quality (ruff):
- Basic Python errors (F)
- Style conventions (E, W)
- Import organization (I)

2. Tests:
- Runs in Docker with CPU PyTorch
- Includes coverage reporting
- Has some timeout protection so I don't burn out too many build minutes

## Branch Structure

Two-branch system to keep our main branch + build images stable:
- `develop`: All development PRs go here, it'll autotest the changes
- `main`: Stable or cool code only

## Pull Request Process

1. Fork & clone
2. Create feature branch from `develop`
3. Make changes, go wild
4. Test in Docker, run ruff and pytest till green
```bash
# Run tests
docker compose run --rm api pytest --asyncio-mode=auto --cov=api

# Run linting
docker compose run --rm api ruff check .
```
5. Submit PR to `develop` branch
6. CI will robo-test your changes
7. Commit any fixes needed, or open a issue, etc
8. Once approved, it'll merge onto main

## Releases

After getting through `develop`, it'll pend on main:
1. Version number in `VERSION` file
2. Docker images with tags:
   - GPU: `latest` and `v#.#.#`
   - CPU: `latest-cpu` and `v#.#.#-cpu`
   - GUI: `latest-gui` and `v#.#.#-gui`
3. Creates a GitHub release with version tag
.

## Questions?

Open an issue
