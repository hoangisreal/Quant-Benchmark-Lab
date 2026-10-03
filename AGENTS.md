# Repository Guidelines

## Project Structure & Module Organization

The current working directory contains no application source, tests, assets, or build manifests. Keep future code organized by responsibility: place production code under a clearly named source directory (such as `src/`), tests under `tests/` or alongside modules, and static files under `assets/`. Update this guide when the actual project layout is added; do not assume a framework-specific structure before its configuration is present.

## Build, Test, and Development Commands

No build, test, or local-run commands are defined in the current checkout. Once the project is initialized, add its canonical commands here and document their purpose. Prefer commands declared by the repository itself (for example, package scripts, a `Makefile`, or documented task runner targets) so contributors use the same workflow as CI.

## Coding Style & Naming Conventions

Follow the formatter, linter, and language conventions selected by the project when they are introduced. Use descriptive names that match each language’s standard casing, keep modules focused, and avoid adding a second formatter or linter without a project need. Format changed files and resolve lint errors before submitting changes.

## Testing Guidelines

There is no test framework or test suite in the current checkout. Add tests with the project’s chosen framework as features are introduced, and document the exact test command here. Name tests after the behavior or module they cover, and include regression coverage for bug fixes.

## Commit & Pull Request Guidelines

No Git history is available in this checkout to establish an existing commit convention. Use short, imperative commit subjects that describe one change (for example, `Add input validation`). Pull requests should explain the change and its motivation, list relevant verification commands and results, and link related issues. Include screenshots when a change affects a user interface.

## Configuration & Secrets

Keep credentials and machine-specific settings out of version control. Provide safe example configuration files where needed, and document required environment variables without including real secret values.
