"""Pure unit tests for the object-key layout (no network)."""

from __future__ import annotations

from langgraph.checkpoint.tigris import _keys


def test_default_ns_encoding() -> None:
    key = _keys.manifest_key("", "thread-1", "", "cp-1")
    assert key == "checkpoints/thread-1/__default__/cp-1/manifest.json"


def test_prefix_applied() -> None:
    key = _keys.blob_key("agents/", "t", "ns", "cp")
    assert key == "agents/checkpoints/t/ns/cp/checkpoint.bin"


def test_special_chars_encoded_and_recoverable() -> None:
    cp_id = "cp/with/slashes"
    key = _keys.manifest_key("", "t", "", cp_id)
    assert "cp/with/slashes" not in key  # slashes were encoded
    assert _keys.checkpoint_id_from_manifest_key(key) == cp_id


def test_ns_round_trip() -> None:
    key = _keys.manifest_key("", "t", "my-ns", "cp")
    assert _keys.ns_from_manifest_key(key) == "my-ns"
    default_key = _keys.manifest_key("", "t", "", "cp")
    assert _keys.ns_from_manifest_key(default_key) == ""


def test_is_manifest_key() -> None:
    assert _keys.is_manifest_key(_keys.manifest_key("", "t", "", "cp"))
    assert not _keys.is_manifest_key(_keys.blob_key("", "t", "", "cp"))


def test_write_key_round_trip_and_negative_idx() -> None:
    for idx in (-4, -1, 0, 5, 999):
        key = _keys.write_key("", "t", "", "cp", "task-1", idx)
        task_id, parsed = _keys.parse_write_key(key)
        assert task_id == "task-1"
        assert parsed == idx


def test_write_keys_sort_numerically() -> None:
    keys = [_keys.write_key("", "t", "", "cp", "task", i) for i in (10, 2, 0, -1)]
    ordered = sorted(keys, key=_keys.parse_write_key)
    assert [_keys.parse_write_key(k)[1] for k in ordered] == [-1, 0, 2, 10]


def test_checkpoint_ids_sort_lexically() -> None:
    # Sortable ids must produce sortable manifest keys (latest == max).
    ids = ["00000000-a", "00000001-a", "00000002-a"]
    keys = [_keys.manifest_key("", "t", "", i) for i in ids]
    assert _keys.checkpoint_id_from_manifest_key(max(keys)) == ids[-1]
