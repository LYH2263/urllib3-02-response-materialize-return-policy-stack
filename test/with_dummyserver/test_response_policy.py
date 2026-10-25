from __future__ import annotations

import gzip
import socket
import threading
from collections.abc import Callable
from typing import cast

import pytest

from dummyserver.testcase import SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, HTTPResponse, Retry
from urllib3.response import (
    BodyMaterialize,
    ConnReturn,
    DecodeProfile,
    response_policy,
)

BODY = b"x" * 100
GZIP_BODY = gzip.compress(b"foo" * 50)


def body_of(resp: object) -> bytes | None:
    body = cast(HTTPResponse, resp)._body
    assert isinstance(body, bytes) or body is None
    return body


def make_keepalive_handler(
    quit_event: threading.Event,
) -> Callable[[socket.socket], None]:
    """
    Accept loop serving keep-alive HTTP/1.1 requests on each accepted
    socket. Responses:

      /body  -> 200, plain BODY
      /gzip  -> 200, GZIP_BODY with content-encoding: gzip
      /red   -> 303 redirect to /body
    """

    def keepalive_handler(listener: socket.socket) -> None:
        def serve(sock: socket.socket) -> None:
            sock.settimeout(0.25)
            buffer = b""
            try:
                while not quit_event.is_set():
                    while b"\r\n\r\n" not in buffer:
                        chunk = sock.recv(65536)
                        if not chunk:
                            return
                        buffer += chunk
                    head, _, rest = buffer.partition(b"\r\n\r\n")
                    request_line = head.split(b"\r\n", 1)[0]
                    parts = request_line.split(b" ")
                    path = parts[1] if len(parts) >= 2 else b"/"

                    if path == b"/red":
                        payload = (
                            b"HTTP/1.1 303 See Other\r\n"
                            b"Location: /body\r\n"
                            b"Content-Length: 0\r\n"
                            b"\r\n"
                        )
                    elif path == b"/gzip":
                        payload = (
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Encoding: gzip\r\n"
                            + f"Content-Length: {len(GZIP_BODY)}\r\n\r\n".encode()
                            + GZIP_BODY
                        )
                    else:
                        payload = (
                            b"HTTP/1.1 200 OK\r\n"
                            + f"Content-Length: {len(BODY)}\r\n\r\n".encode()
                            + BODY
                        )
                    sock.sendall(payload)
                    buffer = rest
            except (ConnectionError, OSError):
                return
            finally:
                sock.close()

        listener.settimeout(0.25)
        while not quit_event.is_set():
            try:
                sock = listener.accept()[0]
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=serve, args=(sock,), daemon=True).start()

    return keepalive_handler


def idle_connections(pool: HTTPConnectionPool) -> list[object]:
    return [c for c in pool.pool.queue if c is not None]  # type: ignore[union-attr]


class TestResponsePolicySocket(SocketDummyServerTestCase):
    def make_pool(self, maxsize: int = 2) -> HTTPConnectionPool:
        return HTTPConnectionPool(self.host, self.port, maxsize=maxsize, retries=False)

    def start_server(self) -> None:
        self._quit_event = threading.Event()
        self._start_server(
            make_keepalive_handler(self._quit_event),
            quit_event=self._quit_event,
        )

    def test_empty_stack_pin_parity(self) -> None:
        self.start_server()
        pool = self.make_pool()
        resp = pool.urlopen("GET", "/body")
        assert body_of(resp) == BODY
        assert resp.connection is None
        assert len(idle_connections(pool)) == 1

    def test_hold_until_close_streaming(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(conn_return=ConnReturn.HOLD_UNTIL_CLOSE):
            resp = pool.urlopen("GET", "/body", preload_content=False)
            assert resp.connection is not None
            assert idle_connections(pool) == []
            data = b"".join(resp.stream(32))
            assert data == BODY
            # Exhausted fp does not auto-release under hold.
            assert resp.connection is not None
            resp.release_conn()
            assert resp.connection is None
            assert len(idle_connections(pool)) == 1

    def test_hold_until_close_eager_preload(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(conn_return=ConnReturn.HOLD_UNTIL_CLOSE):
            resp = pool.urlopen("GET", "/body")
        assert body_of(resp) == BODY
        assert resp.connection is not None
        resp.release_conn()
        assert resp.connection is None

    def test_hold_survives_frame_pop(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(conn_return="hold_until_close"):
            resp = pool.urlopen("GET", "/body", preload_content=False)
        assert resp.connection is not None
        assert resp.read() == BODY
        assert resp.connection is not None
        resp.release_conn()
        assert resp.connection is None

    def test_hold_does_not_occupy_intermediate_connection(self) -> None:
        self.start_server()
        # maxsize=1: the recursion must reuse the intermediate response's
        # connection after draining it.
        pool = self.make_pool(maxsize=1)
        with response_policy(conn_return=ConnReturn.HOLD_UNTIL_CLOSE):
            resp = pool.urlopen(
                "GET",
                "/red",
                preload_content=False,
                retries=Retry(redirect=5),
            )
            assert resp.status == 200
            assert resp.read() == BODY
            assert resp.connection is not None
        assert pool.num_connections == 1
        resp.release_conn()

    def test_drain_then_release_redirect(self) -> None:
        self.start_server()
        pool = self.make_pool(maxsize=1)
        with response_policy(conn_return="drain_then_release"):
            resp = pool.urlopen("GET", "/red", retries=Retry(redirect=5))
            assert body_of(resp) == BODY
            # Final response follows the pin default (released with preload).
            assert resp.connection is None
            assert pool.num_connections == 1

    def test_drain_then_release_user_drain(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(conn_return="drain_then_release"):
            resp = pool.urlopen(
                "GET", "/body", preload_content=False, release_conn=False
            )
            assert resp.connection is not None
            resp.drain_conn()
            assert resp.connection is None
            assert len(idle_connections(pool)) == 1

    def test_follow_preload_frame_matches_pin(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(conn_return=ConnReturn.FOLLOW_PRELOAD):
            # Default call: preload=True -> release defaults True (pin).
            resp = pool.urlopen("GET", "/body")
            assert body_of(resp) == BODY
            assert resp.connection is None
            assert len(idle_connections(pool)) == 1
            # Streaming call: release follows preload and stays False.
            resp = pool.urlopen("GET", "/body", preload_content=False)
            assert resp.connection is not None
            assert idle_connections(pool) == []
            resp.read()
            resp.release_conn()

    def test_stream_only_through_urlopen(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(body_materialize=BodyMaterialize.STREAM_ONLY):
            resp = pool.urlopen("GET", "/body")  # default preload_content=True
            assert body_of(resp) is None
            with pytest.raises(RuntimeError, match="stream_only"):
                resp.data
            assert resp.read() == BODY
        assert resp.connection is None

    def test_force_raw_through_urlopen(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(decode_profile=DecodeProfile.FORCE_RAW):
            resp = pool.urlopen("GET", "/gzip")
            assert body_of(resp) == GZIP_BODY
            assert gzip.decompress(body_of(resp) or b"") == b"foo" * 50

    def test_opaque_cached_through_urlopen(self) -> None:
        self.start_server()
        pool = self.make_pool()
        with response_policy(body_materialize=BodyMaterialize.OPAQUE_CACHED):
            resp = pool.urlopen("GET", "/body", preload_content=False)
            assert resp.read(10) == BODY[:10]
            with pytest.raises(RuntimeError, match="opaque_cached"):
                resp.data

    def test_concurrent_urlopen_frames_do_not_cross(self) -> None:
        self.start_server()
        pool = HTTPConnectionPool(self.host, self.port, maxsize=4, retries=False)
        results: dict[str, list[bytes | None]] = {}
        results_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def stream_only_worker() -> None:
            with response_policy(body_materialize=BodyMaterialize.STREAM_ONLY):
                barrier.wait(timeout=10)
                for _ in range(5):
                    resp = pool.urlopen("GET", "/body")
                    with results_lock:
                        results.setdefault("stream_bodies", []).append(body_of(resp))
                    # Streaming read returns the conn to the pool.
                    assert resp.read() == BODY

        def pin_worker() -> None:
            barrier.wait(timeout=10)
            for _ in range(5):
                resp = pool.urlopen("GET", "/body")
                with results_lock:
                    results.setdefault("pin_bodies", []).append(body_of(resp))

        threads = [
            threading.Thread(target=stream_only_worker),
            threading.Thread(target=pin_worker),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert all(not thread.is_alive() for thread in threads)
        assert all(body is None for body in results["stream_bodies"])
        assert all(body == BODY for body in results["pin_bodies"])

    def test_frame_popped_after_exception(self) -> None:
        self.start_server()
        pool = self.make_pool()
        try:
            with response_policy(body_materialize=BodyMaterialize.STREAM_ONLY):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        resp = pool.urlopen("GET", "/body")
        assert body_of(resp) == BODY
