from __future__ import annotations

from pathlib import Path

from review_app import create_app


def test_dashboard_session_queue_filters_and_safe_source_routes(repository_root: Path, tmp_path: Path):
    app = create_app(repository_root, workspace_root=tmp_path / "review-workspace", testing=True)
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 302
    response = client.post("/session", data={"reviewer": "مراجع Élodie"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"35" in response.data
    assert client.get("/?coverage=full&extraction=success").status_code == 200
    workspace = app.extensions["review_workspace"]
    state, _, _ = workspace.load(next(iter(workspace.entries)))
    source_id = state["sources"][0]["source_id"]
    assert client.get(f"/source/{source_id}").status_code == 200
    assert client.get("/source/not_an_inventoried_source").status_code == 404
    assert client.get("/source/..%2F..%2FREADME.md").status_code == 404


def test_invalid_dashboard_edit_has_no_workspace_mutation(repository_root: Path, tmp_path: Path):
    app = create_app(repository_root, workspace_root=tmp_path / "review-workspace", testing=True)
    client = app.test_client()
    client.post("/session", data={"reviewer": "Reviewer"})
    workspace = app.extensions["review_workspace"]
    review_id = next(row["review_record_id"] for row in workspace.list_records() if row["document_kind"] == "contract_template")
    state_path, canonical_path, _ = workspace._paths(review_id)
    before = state_path.read_bytes(), canonical_path.read_bytes()
    response = client.post(
        f"/review/{review_id}/correction",
        data={"correction": '{"source_files": [{"relative_path": "evil"}]}'},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Unsupported correction properties" in response.data
    assert (state_path.read_bytes(), canonical_path.read_bytes()) == before

