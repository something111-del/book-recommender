# Contributing to Book Recommender

Thank you for your interest in contributing to the Book Recommender project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check if the enhancement has already been suggested
- Provide a clear description of the proposed feature
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Run tests to ensure nothing breaks
5. Commit your changes (`git commit -m 'Add YourFeature'`)
6. Push to your branch (`git push origin feature/YourFeature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Testing

- Add tests for new features
- Ensure all existing tests pass
- Aim for good test coverage

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Questions?

Feel free to open an issue for any questions about contributing!
