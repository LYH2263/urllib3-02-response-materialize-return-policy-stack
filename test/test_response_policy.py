from __future__ import annotations

import gzip
import http.client as httplib
import threading
import zlib
from io import BytesIO

import pytest

import urllib3
from urllib3.response import (
    BodyMaterialize,
    ConnReturn,
    DecodeProfile,
    HTTPResponse,
    ResponsePolicyFrame,
    current_response_policy,
    pop_response_policy,
    push_response_policy,
    response_policy,
)

from .test_response import MockChunkedEncodingResponse, MockSock


class TestPolicyStack:
    def test_empty_stack_is_none(self) -> None:
        assert current_response_policy() is None

    def test_push_pop(self) -> None:
        frame = push_response_policy(body_materialize="stream_only")
        try:
            assert current_response_policy() is frame
            assert frame.body_materialize is BodyMaterialize.STREAM_ONLY
            assert frame.decode_profile is DecodeProfile.HONOR_DEFAULT
            assert frame.conn_return is ConnReturn.FOLLOW_PRELOAD
        finally:
            pop_response_policy(frame)
        assert current_response_policy() is None

    def test_nested_frames_restore_outer_on_pop(self) -> None:
        outer = push_response_policy(decode_profile="force_raw")
        try:
            inner = push_response_policy(body_materialize="stream_only")
            try:
                assert current_response_policy() is inner
            finally:
                pop_response_policy(inner)
            assert current_response_policy() is outer
        finally:
            pop_response_policy(outer)
        assert current_response_policy() is None

    def test_context_manager_pops_on_exception(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with response_policy(body_materialize="stream_only"):
                raise ValueError("boom")
        assert current_response_policy() is None

    def test_frame_context_manager_protocol(self) -> None:
        frame = push_response_policy(conn_return="hold_until_close")
        with frame:
            assert current_response_policy() is frame
        assert current_response_policy() is None

    def test_double_pop_raises(self) -> None:
        frame = push_response_policy()
        pop_response_policy(frame)
        with pytest.raises(RuntimeError):
            pop_response_policy(frame)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"body_materialize": "nope"},
            {"decode_profile": "nope"},
            {"conn_return": "nope"},
        ],
    )
    def test_invalid_enum_value_rejected(self, kwargs: dict[str, str]) -> None:
        with pytest.raises(ValueError):
            push_response_policy(**kwargs)
        assert current_response_policy() is None

    def test_enum_and_string_equivalent(self) -> None:
        frame = push_response_policy(
            body_materialize=BodyMaterialize.STREAM_ONLY,
            decode_profile=DecodeProfile.FORCE_RAW,
            conn_return=ConnReturn.HOLD_UNTIL_CLOSE,
        )
        pop_response_policy(frame)

    def test_stack_is_isolated_between_threads(self) -> None:
        seen = {}

        def worker() -> None:
            with response_policy(body_materialize="stream_only"):
                seen[threading.get_ident()] = current_response_policy()
                barrier.wait(timeout=5)

        barrier = threading.Barrier(2)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert current_response_policy() is None
        assert len(seen) == 2
        assert len({id(f) for f in seen.values()}) == 2


class TestBodyMaterializePolicy:
    def test_unstacked_construction_pins(self) -> None:
        resp = HTTPResponse(BytesIO(b"foo"), preload_content=True)
        assert resp._body == b"foo"

    def test_stream_only_forbids_eager_preload(self) -> None:
        fp = BytesIO(b"foo")
        with response_policy(body_materialize="stream_only"):
            resp = HTTPResponse(fp, preload_content=True)
            assert resp._body is None
            assert fp.tell() == 0

    def test_stream_only_data_does_not_cache(self) -> None:
        with response_policy(body_materialize="stream_only"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            with pytest.raises(RuntimeError, match="stream_only"):
                resp.data
            with pytest.raises(RuntimeError, match="stream_only"):
                resp.read(cache_content=True)
            # Plain streaming reads still work and never set _body.
            assert resp.read() == b"foo"
            assert resp._body is None

    def test_stream_only_binding_governs_response_after_pop(self) -> None:
        with response_policy(body_materialize="stream_only"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
        assert current_response_policy() is None
        assert resp._bound_response_policy is not None
        with pytest.raises(RuntimeError, match="stream_only"):
            resp.data

    def test_stream_only_does_not_affect_unbound_response(self) -> None:
        with response_policy(body_materialize="stream_only"):
            pass
        resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
        assert resp.data == b"foo"
        assert resp._body == b"foo"

    def test_opaque_cached_allows_fresh_full_cache(self) -> None:
        with response_policy(body_materialize="opaque_cached"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            assert resp.data == b"foo"
            assert resp._body == b"foo"

    def test_opaque_cached_eager_preload_complete(self) -> None:
        with response_policy(body_materialize="opaque_cached"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=True)
            assert resp._body == b"foo"

    def test_opaque_cached_rejects_partial_then_data(self) -> None:
        with response_policy(body_materialize="opaque_cached"):
            resp = HTTPResponse(BytesIO(b"foobar"), preload_content=False)
            assert resp.read(3) == b"foo"
            with pytest.raises(RuntimeError, match="opaque_cached"):
                resp.data
            with pytest.raises(RuntimeError, match="opaque_cached"):
                resp.read(cache_content=True)
            # The remainder is still streamable.
            assert resp.read() == b"bar"
            assert resp._body is None

    def test_opaque_cached_chunked_half_read_then_data(self) -> None:
        with response_policy(body_materialize="opaque_cached"):
            httplib_resp = httplib.HTTPResponse(MockSock)  # type: ignore[arg-type]
            httplib_resp.chunked = True
            httplib_resp.fp = MockChunkedEncodingResponse([b"chunk"])  # type: ignore[assignment]
            resp = HTTPResponse(
                httplib_resp,
                preload_content=False,
                headers={"transfer-encoding": "chunked"},
            )
            gen = resp.read_chunked(2)
            next(gen)
            with pytest.raises(RuntimeError, match="opaque_cached"):
                resp.data

    def test_preload_eager_aligns_pin(self) -> None:
        with response_policy(body_materialize="preload_eager"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=True)
            assert resp._body == b"foo"
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            assert resp._body is None

    def test_drain_conn_ignores_materialize_cache_rules(self) -> None:
        # stream_only forbids caching elsewhere, but draining must still be
        # possible and must never raise.
        with response_policy(body_materialize="stream_only"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            resp.drain_conn()
            assert resp.read() == b""

    def test_drain_conn_does_not_invent_body_cache(self) -> None:
        with response_policy(body_materialize="opaque_cached"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            resp.drain_conn()
            assert resp._body is None


class TestDecodeProfilePolicy:
    def test_honor_default_decodes_like_pin(self) -> None:
        data = zlib.compress(b"foo")
        with response_policy(decode_profile="honor_default"):
            resp = HTTPResponse(BytesIO(data), headers={"content-encoding": "deflate"})
            assert resp._body == b"foo"

    def test_force_raw_eager_preload_is_raw(self) -> None:
        data = gzip.compress(b"foo")
        with response_policy(decode_profile="force_raw"):
            resp = HTTPResponse(BytesIO(data), headers={"content-encoding": "gzip"})
            assert resp._body == data

    def test_force_raw_read_and_data(self) -> None:
        data = gzip.compress(b"foo")
        with response_policy(decode_profile="force_raw"):
            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "gzip"},
            )
            assert resp.read() == data

            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "gzip"},
            )
            assert resp.data == data

    def test_force_raw_does_not_silently_flip_after_decode(self) -> None:
        data = zlib.compress(b"foo")
        resp = HTTPResponse(
            BytesIO(data),
            preload_content=False,
            headers={"content-encoding": "deflate"},
        )
        assert resp.read(1, decode_content=True) == b"f"
        with response_policy(decode_profile="force_raw"):
            with pytest.raises(RuntimeError, match="not supported after"):
                resp.read(1, decode_content=False)

    def test_force_raw_preserves_head_and_204_zero_length(self) -> None:
        with response_policy(decode_profile="force_raw"):
            resp = HTTPResponse(
                BytesIO(b""), preload_content=False, request_method="HEAD"
            )
            assert resp.length_remaining == 0
            resp = HTTPResponse(BytesIO(b""), status=204, preload_content=False)
            assert resp.length_remaining == 0
            assert resp.enforce_content_length is True

    def test_forbid_flip_bans_reverse_flip(self) -> None:
        data = zlib.compress(b"foo")
        with response_policy(decode_profile="forbid_flip"):
            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "deflate"},
            )
            assert resp.read(2, decode_content=False) is not None
            with pytest.raises(RuntimeError, match="not supported after"):
                resp.read(1, decode_content=True)

    def test_forbid_flip_keeps_classic_flip_error(self) -> None:
        data = zlib.compress(b"foo")
        with response_policy(decode_profile="forbid_flip"):
            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "deflate"},
            )
            assert resp.read(1, decode_content=True) == b"f"
            with pytest.raises(
                RuntimeError,
                match=r"Calling read\(decode_content=False\) is not supported",
            ):
                resp.read(1, decode_content=False)

    def test_forbid_flip_allows_flip_without_encoding(self) -> None:
        # No content-encoding means flipping cannot corrupt anything.
        with response_policy(decode_profile="forbid_flip"):
            resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
            assert resp.read(1, decode_content=False) == b"f"
            assert resp.read(decode_content=True) == b"oo"

    def test_force_raw_read1(self) -> None:
        data = zlib.compress(b"foo")
        with response_policy(decode_profile="force_raw"):
            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "deflate"},
            )
            assert resp.read1(2) is not None
            assert resp._has_decoded_content is False

    def test_forbid_flip_read1_reverse(self) -> None:
        data = zlib.compress(b"foo")
        with response_policy(decode_profile="forbid_flip"):
            resp = HTTPResponse(
                BytesIO(data),
                preload_content=False,
                headers={"content-encoding": "deflate"},
            )
            resp.read1(2, decode_content=False)
            with pytest.raises(RuntimeError, match="not supported after"):
                resp.read1(decode_content=True)

    def test_force_raw_read_chunked(self) -> None:
        with response_policy(decode_profile="force_raw"):
            httplib_resp = httplib.HTTPResponse(MockSock)  # type: ignore[arg-type]
            httplib_resp.chunked = True
            httplib_resp.fp = MockChunkedEncodingResponse([b"chunk"])  # type: ignore[assignment]
            resp = HTTPResponse(
                httplib_resp,
                preload_content=False,
                headers={"transfer-encoding": "chunked"},
            )
            chunks = list(resp.stream())
            assert b"chunk" in b"".join(chunks)
            assert resp._has_decoded_content is False

    def test_inner_decode_frame_overrides_outer(self) -> None:
        data = gzip.compress(b"foo")
        with response_policy(decode_profile="force_raw"):
            resp = HTTPResponse(BytesIO(data), headers={"content-encoding": "gzip"})
            assert resp._body == data
            with response_policy(decode_profile="honor_default"):
                resp = HTTPResponse(BytesIO(data), headers={"content-encoding": "gzip"})
                assert resp._body == b"foo"
            resp = HTTPResponse(BytesIO(data), headers={"content-encoding": "gzip"})
            assert resp._body == data


class TestConnReturnUnitFork:
    class FakeConn:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakePool:
        def __init__(self) -> None:
            self.returned: list[object] = []

        def _put_conn(self, conn: object) -> None:
            self.returned.append(conn)

    def _make_response(self) -> tuple[HTTPResponse, FakePool, FakeConn]:
        resp = HTTPResponse(BytesIO(b"foo"), preload_content=False)
        pool = self.FakePool()
        conn = self.FakeConn()
        resp._pool = pool  # type: ignore[assignment]
        resp._connection = conn  # type: ignore[assignment]
        return resp, pool, conn

    def test_unstacked_drain_does_not_release(self) -> None:
        resp, pool, _conn = self._make_response()
        resp.drain_conn()
        assert pool.returned == []
        assert resp._connection is not None

    def test_drain_then_release_drain_returns_conn(self) -> None:
        resp, pool, conn = self._make_response()
        with response_policy(conn_return="drain_then_release"):
            resp.drain_conn()
        assert pool.returned == [conn]
        assert resp._connection is None

    def test_hold_drain_does_not_return(self) -> None:
        resp, pool, _conn = self._make_response()
        with response_policy(conn_return="hold_until_close"):
            resp.drain_conn()
        assert pool.returned == []
        assert resp._connection is not None

    def test_hold_explicit_release_still_works(self) -> None:
        resp, pool, conn = self._make_response()
        with response_policy(conn_return="hold_until_close"):
            resp.release_conn()
        assert pool.returned == [conn]
        assert resp._connection is None

    def test_bound_frame_governs_drain_after_pop(self) -> None:
        resp, pool, _conn = self._make_response()
        with response_policy(conn_return="drain_then_release"):
            pass
        # The response itself was not created inside the frame, so it keeps
        # pin semantics: only the live frame applies to an unbound response
        # constructed before the frame was pushed and popped.
        resp.drain_conn()
        assert pool.returned == []


def test_public_api_exports() -> None:
    assert urllib3.BodyMaterialize is BodyMaterialize
    assert urllib3.DecodeProfile is DecodeProfile
    assert urllib3.ConnReturn is ConnReturn
    assert issubclass(ResponsePolicyFrame, object)
    assert callable(urllib3.push_response_policy)
    assert callable(urllib3.pop_response_policy)
    assert callable(urllib3.response_policy)
    assert urllib3.current_response_policy() is None
