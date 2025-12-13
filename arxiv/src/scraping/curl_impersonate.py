from dataclasses import dataclass
import tempfile
import subprocess


@dataclass
class CurlResponse:
    stdout: str
    stderr: str
    text: str


def _get_runner_script(browser_name: str) -> str:
    if browser_name == "chrome":
        return "curl_chrome110"
    elif browser_name == "safari":
        return "safari15_5"
    else:
        raise ValueError(f"Unsupported Browser:{browser_name}")


def get_webpage(curl_command: str, browser_name: str = "chrome") -> CurlResponse:
    """See more: https://hub.docker.com/r/lwthiker/curl-impersonate"""
    runner_script = _get_runner_script(browser_name)
    with tempfile.NamedTemporaryFile(mode="w+b") as temp_file:
        # requires this docker image of chrome impersonator
        # run: `docker pull lwthiker/curl-impersonate:0.6-chrome`
        cmd = [
            f"docker run --rm lwthiker/curl-impersonate:0.6-chrome {runner_script} {curl_command} > {temp_file.name}",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
        )
        content = temp_file.read()
        api_resp_str = content.decode("utf-8")
        return CurlResponse(
            stdout=result.stdout, stderr=result.stderr, text=api_resp_str
        )
