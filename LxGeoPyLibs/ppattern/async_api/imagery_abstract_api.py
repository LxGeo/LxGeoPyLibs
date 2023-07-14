import httpx
import json
import logging
from LxGeoPyLibs.ppattern.async_api.exceptions import BadRequest, Unauthorized, NotFoundError, \
    RatelimitError, ServerError, UnexpectedError, NotResponding, NetworkError


logger = logging.getLogger('ImageryAbstractApi.client')


class ImageryAbstractApi:


    def __init__(self, auth = None, session=None, is_async=False, **options):
        
        self.auth = auth

        self.is_async = is_async
        self.session = (session or (httpx.AsyncClient() if is_async
                        else httpx.Client()))
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.session.headers.update(headers)

        self.api_url = options.get("api_url")

        self.timeout = options.get('timeout', 10)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.aclose()

    def close(self):
        if self.is_async:
            logger.warning("Please use aclose() method to close a async client.")
        return self.session.close()

    async def aclose(self):
        await self.session.aclose()

    def _raise_for_status(self, resp, text, *, method=None, return_raw=True):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = text
        code = getattr(resp, 'status', None) or getattr(resp, 'status_code')

        logger.debug(self.REQUEST_LOG.format(
            method=method or resp.request_info.method, url=resp.url,
            text=text, status=code
        ))

        if 300 > code >= 200:  # Request was successful
            if not return_raw:
                return data
            return data, resp  # value, response 
        elif code == 400:
            raise BadRequest(resp, data)
        elif code in (401, 403):  # Unauthorized request - Invalid credentials
            raise Unauthorized(resp, data)
        elif code == 404:  # not found
            raise NotFoundError(resp, data)
        elif code == 429:
            raise RatelimitError(resp, data)
        elif code == 503:  # Maintainence
            raise ServerError(resp, data)
        else:
            raise UnexpectedError(resp, data)

    async def _arequest(self, method, url, **kwargs):
        timeout = kwargs.pop('timeout', self.timeout)
        return_raw = kwargs.pop("return_raw", True)

        try:
            resp = await self.session.request(
                method, url, timeout=timeout, auth=self.auth, **kwargs
            )
            return self._raise_for_status(
                resp, resp.text, method=method, return_raw=return_raw
            )
        except (httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout):
            raise NotResponding
        except httpx.NetworkError:
            raise NetworkError
        finally:
            try:
                await resp.aclose()
            except UnboundLocalError:
                pass

    def _request(self, method, url, **kwargs):
        if self.is_async:  # return a coroutine
            return self._arequest(method, url, **kwargs)
        timeout = kwargs.pop('timeout', self.timeout)
        return_raw = kwargs.pop("return_raw", True)

        try:
            resp = self.session.request(
                method, url, timeout=timeout, auth=self.auth, **kwargs
            )
            return self._raise_for_status(
                resp, resp.text, method=method, return_raw=return_raw
            )
        except (httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout):
            raise NotResponding
        except httpx.NetworkError:
            raise NetworkError
        finally:
            try:
                resp.close()
            except UnboundLocalError:
                pass

    def _split_kwargs(self, **kwargs):
        protected_kwargs = [
            "method", "url", "params", "data", "json", "headers", "cookies",
            "files", "auth", "timeout", "allow_redirects", "proxies", "verify",
            "stream", "cert", "return_raw"
        ]
        params = kwargs.pop("params", {})
        for key in list(kwargs.keys()):
            if key not in protected_kwargs:
                params[key] = kwargs.pop(key)

        return params, kwargs

    def get(self, **kwargs):
        params, kwargs = self._split_kwargs(**kwargs)
        return self._request("GET", self.api_url, params=params, **kwargs)

    def post(self, data=None, json=None, **kwargs):
        params, kwargs = self._split_kwargs(**kwargs)
        return self._request("POST", self.api_url, params=params, data=data, json=json, **kwargs)

    def delete(self, **kwargs):
        params, kwargs = self._split_kwargs(**kwargs)
        return self._request("DELETE", self.api_url, params=params, **kwargs)