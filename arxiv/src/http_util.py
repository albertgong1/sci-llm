from pathlib import Path
import asyncio

import aiohttp
from aiohttp import ClientSession

# not exhaustive
_CONTENT_TYPE_TO_F_EXT = {"application/pdf": "pdf"}


async def _fetch(
    url: str,
    session: ClientSession,
    allow_redirects: bool,
    param: dict[str, str] | None,
    header: dict[str, str] | None,
    cookie: dict[str, str] | None,
    download_to_folder: Path | None,
) -> str:
    args = {"allow_redirects": allow_redirects}

    # trying not to thrash defaults if they exist
    if param:
        args["params"] = param
    if header:
        args["headers"] = header
    if cookie:
        args["cookies"] = cookie

    if download_to_folder is not None:
        assert download_to_folder.exists()
        assert download_to_folder.is_dir(follow_symlinks=False), "No symlinks!"
        assert url

        # downloading a file
        async with session.get(url) as response:
            content_type: str = response.content_type
            sp = url.split("/")[-1]
            if not sp:
                raise ValueError(f"{url} could not be split...")
            # append the file extension if it is obvious
            f_ext = "." + _CONTENT_TYPE_TO_F_EXT.get(content_type, "")
            sp += f_ext
            save_to_path = download_to_folder / sp

            # download it
            is_incomplete = False
            with open(save_to_path, "wb") as f:
                while True:
                    try:
                        c = await response.content.read(1 << 20)
                        if not c:
                            break
                        f.write(c)
                    except aiohttp.ClientPayloadError as e:
                        if "Not enough data to satisfy content length header." in str(
                            e.__cause__
                        ):
                            # ok, just stop pulling
                            is_incomplete = True
                        else:
                            # some other error, throw it
                            raise

            if is_incomplete:
                save_to_path.rename(save_to_path.stem + "_incomplete" + f_ext)

            return str(save_to_path.absolute())
    else:
        # no download, return the text content
        async with session.get(url, **args) as response:
            t = await response.text()
            return t


async def _bound_fetch(
    sem: asyncio.Semaphore,
    url: str,
    session: ClientSession,
    allow_redirects: bool,
    param: dict[str, str] | None,
    header: dict[str, str] | None,
    cookie: dict[str, str] | None,
    download_to_folder: Path | None,
) -> str:
    async with sem:
        return await _fetch(
            url, session, allow_redirects, param, header, cookie, download_to_folder
        )


async def _run(
    urls: list[str],
    max_concurrent_coro: int,
    allow_redirects: bool,
    params: list[dict[str, str] | None],
    headers: list[dict[str, str] | None],
    cookies: list[dict[str, str] | None],
    download_to_folder: Path | None,
) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent_coro)
    async with ClientSession() as session:
        tasks = [
            _bound_fetch(
                sem,
                url,
                session,
                allow_redirects,
                param,
                header,
                cookie,
                download_to_folder,
            )
            for url, param, header, cookie in zip(urls, params, headers, cookies)
        ]
        return await asyncio.gather(*tasks)


def get_responses(
    urls: list[str],
    params: list[dict[str, str]] | None = None,
    headers: list[dict[str, str]] | None = None,
    cookies: list[dict[str, str]] | None = None,
    # connection options
    max_concurrent_coro: int = 2_000,
    allow_redirects: bool = True,
    # download options
    download_to_folder: Path | None = None,
) -> list[str]:
    """Make several async requests at once.

    The (urls, params, headers, cookies) objects must be aligned and the same length if provided.

    Only supports GET requests.

    :param urls: URLs to use
    :param params: optional URL params for each url
    :param headers: optional headers for request
    :param cookies: optional cookies for request.
    :param max_concurrent_coro: Maximum number of concurrent coroutines that can be in event
        loop at once.
    :param allow_redirects: Whether to follow redirects on http 30X
    :param download_to_folder: if provided, a subfolder to where to save files. If this is non-null,
        then the response will be streamed in smaller chunks and saved to disk.
    :return: text of the http response or the path to the file saved on download.
    """
    params = params or ([None] * len(urls))
    headers = headers or ([None] * len(urls))
    cookies = cookies or ([None] * len(urls))
    assert len(params) == len(headers) == len(cookies) == len(urls), "Size mismatch."

    if download_to_folder is not None:
        # better for downloading
        max_concurrent_coro = min(max_concurrent_coro, 150)

    return asyncio.run(
        _run(
            urls,
            max_concurrent_coro,
            allow_redirects,
            params,
            headers,
            cookies,
            download_to_folder,
        )
    )
