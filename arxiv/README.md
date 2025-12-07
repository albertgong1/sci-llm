## Arxiv Web Tools

### Data Setup

Place files in `arxiv/artifacts`. These are ignored by version control. Generally, inputs and outputs are `.csv` files and are placed here.

### Environment Setup

If not installed, install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repo root, navigate to the `/arxiv/` folder (`cd ./arxiv`) and run:
```bash
uv sync
```

This will create a virtual environment with all Python dependencies.

### Captcha

We need Docker to automate requests. Run the following to pull a specific container:
```bash
docker pull lwthiker/curl-impersonate:0.6-chrome
```

Then ensure the Docker daemon is running. 

### Code Editing

#### Run Code Formatting

```bash
sh run_format.sh
```

#### Run Unit Tests

```bash
sh run_tests.sh
```