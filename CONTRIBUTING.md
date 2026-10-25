# Contributing to dfmacro

First off, thank you for considering contributing to `dfmacro`! It's people like you that make open source such a great community.

## How to contribute

### 1. Report bugs
If you find a bug, please open an issue. Include:
- Your Python version
- Your pandas/polars version
- A minimal reproducible example

### 2. Suggest features
We welcome new ideas! Open an issue and describe:
- What you want to do
- Why the current API doesn't support it
- How you imagine the syntax should look

### 3. Submit a Pull Request
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-new-feature`
3. Make your changes.
4. Add tests for your changes.
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m "feat: add my new feature"`
7. Push to the branch: `git push origin feature/my-new-feature`
8. Open a Pull Request.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Tony123-tech/dfmacro.git
cd dfmacro

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install in editable mode with test dependencies
pip install -e ".[test]"

# Run the tests
pytest