# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import os
import inspect
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(self._uri, **self._connect_kwargs(headers))
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    @staticmethod
    def _connect_kwargs(headers: Optional[Dict]) -> Dict:
        """Return kwargs supported by the installed websockets version.

        ``websockets.sync.client.connect`` changed keyword support across
        releases.  Some versions accept ``additional_headers`` and keepalive
        ping kwargs, while older sync clients reject them.  Build the richest
        compatible kwargs instead of pinning the whole environment to one
        websockets release.
        """
        preferred = {
            "compression": None,
            "max_size": None,
            "open_timeout": 150,
            "ping_interval": 20,
            "ping_timeout": 20,
        }

        params = inspect.signature(websockets.sync.client.connect).parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            kwargs = dict(preferred)
            if headers:
                kwargs["additional_headers"] = headers
            return kwargs

        kwargs = {key: value for key, value in preferred.items() if key in params}
        if headers:
            if "additional_headers" in params:
                kwargs["additional_headers"] = headers
            elif "extra_headers" in params:
                kwargs["extra_headers"] = headers
        return kwargs

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
