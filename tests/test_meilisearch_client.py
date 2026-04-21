"""Unit tests for Meilisearch client helpers."""

from src.tools import meilisearch_client as mc


def test_list_index_uids_sorted(mocker):
    mocker.patch(
        "src.tools.meilisearch_client.requests.get",
        return_value=mocker.Mock(
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"uid": "zebra"}, {"uid": "alpha"}]},
        ),
    )
    uids = mc.list_index_uids()
    assert uids == ["alpha", "zebra"]


def test_list_index_uids_fallback_to_default(mocker):
    mocker.patch(
        "src.tools.meilisearch_client.requests.get",
        side_effect=ConnectionError("down"),
    )
    uids = mc.list_index_uids()
    assert uids == [mc.settings.meili_index_name]
