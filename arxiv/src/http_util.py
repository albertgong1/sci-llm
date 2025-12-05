from typing import Optional
import asyncio
from aiohttp import ClientSession


async def _fetch(
    url: str,
    session: ClientSession,
    allow_redirects: bool,
    param: Optional[dict[str, str]],
    header: Optional[dict[str, str]],
    cookie: Optional[dict[str, str]],
) -> str:
    args = {"allow_redirects": allow_redirects}

    # trying not to thrash defaults if they exist
    if param:
        args["params"] = param
    if header:
        args["headers"] = header
    if cookie:
        args["cookies"] = cookie
    async with session.get(url, **args) as response:
        t = await response.text()
        return t


async def _bound_fetch(
    sem: asyncio.Semaphore,
    url: str,
    session: ClientSession,
    allow_redirects: bool,
    param: Optional[dict[str, str]],
    header: Optional[dict[str, str]],
    cookie: Optional[dict[str, str]],
) -> str:
    async with sem:
        return await _fetch(url, session, allow_redirects, param, header, cookie)


async def _run(
    urls: list[str],
    max_concurrent_coro: int,
    allow_redirects: bool,
    params: list[dict[str, str] | None],
    headers: list[dict[str, str] | None],
    cookies: list[dict[str, str] | None],
) -> list[str]:
    sem = asyncio.Semaphore(max_concurrent_coro)
    async with ClientSession() as session:
        tasks = [
            _bound_fetch(sem, url, session, allow_redirects, param, header, cookie)
            for url, param, header, cookie in zip(urls, params, headers, cookies)
        ]
        return await asyncio.gather(*tasks)


def get_responses(
    urls: list[str],
    params: Optional[list[dict[str, str]]] = None,
    headers: Optional[list[dict[str, str]]] = None,
    cookies: Optional[list[dict[str, str]]] = None,
    # connection options
    max_concurrent_coro: int = 2_000,
    allow_redirects: bool = True,
) -> list[str]:
    params = params or ([None] * len(urls))
    headers = headers or ([None] * len(urls))
    cookies = cookies or ([None] * len(urls))
    assert len(params) == len(headers) == len(cookies) == len(urls), "Size mismatch."
    return asyncio.run(
        _run(urls, max_concurrent_coro, allow_redirects, params, headers, cookies)
    )
