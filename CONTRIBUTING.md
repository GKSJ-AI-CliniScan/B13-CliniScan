# Contributing to CliniScan

Thank you for your interest in contributing to CliniScan! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions with other contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/B13-CliniScan.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use type hints for function signatures
- Write docstrings for all public functions and classes

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chest_xray_analysis

# Run specific test file
pytest tests/test_annotation_converter.py
```

### Formatting Code

```bash
# Format code with black
black chest_xray_analysis tests

# Sort imports with isort
isort chest_xray_analysis tests

# Check code style with flake8
flake8 chest_xray_analysis tests
```

## Commit Messages

Use clear and descriptive commit messages:

```
[Component] Brief description of changes

- Detailed explanation of what changed
- Why the change was necessary
- Any related issues or PRs
```

Example:
```
[preprocessing] Add photometric correction for DICOM images

- Handle MONOCHROME1 vs MONOCHROME2 interpretation
- Invert pixel values when necessary
- Add unit tests for photometric correction
- Fixes #42
```

## Pull Request Process

1. Ensure all tests pass: `pytest`
2. Ensure code is properly formatted: `black`, `isort`, `flake8`
3. Update documentation if needed
4. Create a descriptive pull request with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Any breaking changes

## Documentation

- Add docstrings to all public functions and classes
- Update README.md if adding new features
- Include examples for new functionality

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment information (Python version, OS, etc.)
- Relevant code snippets or error messages

## Feature Requests

When requesting features:

- Describe the use case
- Explain why it's needed
- Provide examples if possible
- Consider backward compatibility

## Questions?

Feel free to open an issue or discussion for questions about contributing.

Thank you for contributing to CliniScan!
