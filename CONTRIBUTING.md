# How to contribute to this project.

1. Install the development environment

   We use [uv](https://docs.astral.sh/uv/) for fast and reliable dependency management. First, install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then, install the project in development mode with all dependencies:

   ```bash
   uv sync --dev
   ```

2. Initialize the pre-commit hooks

   Please initialize the pre-commit hooks, which will automatically run black formatter before each commit.

   ```bash
   uv run pre-commit install
   ```

3. Open a new branch for your changes

   Please open a new branch for your changes, and make sure that your changes are isolated to that branch.

   ```bash
   git checkout -b <branch-name>
   ```

4. Make your changes, format your code, and run the tests

   Make your changes (VSCode is strongly recommended as an editor)
   Please, format your code with a formatter (e.g. black).
   If you are using VSCode as your editor, you can install the Python extension, and set the formatter to black ([https://code.visualstudio.com/docs/python/formatting](https://code.visualstudio.com/docs/python/formatting)).
   The pre-commit hooks will automatically run the formatter before each commit.
   If some code needs to be formatted, the pre-commit hooks stop the commit and format the code. You can then commit again (so better to already have the code well formatted before committing to avoid re-doing the commit).
   Then, make sure that the tests are passing. You can manually run the tests with pytest:

   ```bash
   uv run pytest
   ```

   If you are using VSCode as your editor, you can also install the [Python extension](https://code.visualstudio.com/docs/python/testing), and run the tests from the editor.

5. Push your changes to the remote repository

   Please push your changes to the remote repository, and open a pull request.

   ```bash
   git push origin <branch-name>
   ```

6. Open a pull request

   Please open a pull request on GitHub to the `dev` branch, and make sure that the tests are passing.
