"""Local-only Flask entry point for the Hannibal pilot review dashboard."""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from review_workspace import ReviewError, ReviewWorkspace  # noqa: E402


def create_app(
    repository_root: Path = ROOT, *, workspace_root: Path | None = None,
    testing: bool = False,
) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "review_ui"))
    app.config.update(
        SECRET_KEY=secrets.token_hex(32),
        TESTING=testing,
        MAX_CONTENT_LENGTH=1_000_000,
    )
    workspace = ReviewWorkspace(repository_root, workspace_root=workspace_root)
    workspace.initialize()
    app.extensions["review_workspace"] = workspace

    def reviewer() -> str:
        value = session.get("reviewer", "").strip()
        if not value:
            raise ReviewError("Start a reviewer session first")
        return value

    @app.before_request
    def require_reviewer():
        if request.endpoint not in {"start_session", "static"} and not session.get("reviewer"):
            return redirect(url_for("start_session", next=request.path))
        return None

    @app.route("/session", methods=["GET", "POST"])
    def start_session():
        if request.method == "POST":
            name = request.form.get("reviewer", "").strip()
            if not name:
                flash("Reviewer name is required.", "error")
            elif len(name) > 200:
                flash("Reviewer name is too long.", "error")
            else:
                session["reviewer"] = name
                target = request.form.get("next", "")
                return redirect(target if target.startswith("/") and not target.startswith("//") else url_for("queue"))
        return render_template("session.html", next=request.args.get("next", ""))

    @app.get("/")
    def queue():
        rows = workspace.list_records()
        kind = request.args.get("kind", "")
        extraction = request.args.get("extraction", "")
        coverage = request.args.get("coverage", "")
        disposition = request.args.get("disposition", "")
        query = request.args.get("q", "").casefold().strip()
        visible = [row for row in rows if (
            (not kind or (row["document_kind"] or "none") == kind)
            and (not extraction or row["extraction_status"] == extraction)
            and (not coverage or row["review_coverage"] == coverage)
            and (not disposition or row["final_disposition"] == disposition)
            and (not query or query in " ".join(row["source_paths"]).casefold())
        )]
        required = sum(len(row["required_reviews"]) for row in rows)
        completed = sum(row["completed_reviews"] for row in rows)
        summary = {
            "groups": len(rows), "visible": len(visible), "required": required,
            "completed": completed, "approved_templates": sum(row["training_eligibility"] == "eligible" for row in rows),
        }
        return render_template("queue.html", rows=visible, all_rows=rows, summary=summary)

    @app.get("/review/<review_id>")
    def review(review_id: str):
        try:
            state, canonical, extraction = workspace.load(review_id)
        except ReviewError:
            abort(404)
        return render_template(
            "review.html", state=state, canonical=canonical, extraction=extraction,
            categories=("extraction", "classification", "structure", "privacy", "sanitization", "legal"),
            canonical_json=json.dumps(canonical, ensure_ascii=False, indent=2) if canonical else "",
        )

    def run_action(review_id: str, action):
        try:
            action()
            flash("Change saved and revalidated.", "success")
        except (ReviewError, ValueError, json.JSONDecodeError) as exc:
            flash(str(exc), "error")
        return redirect(url_for("review", review_id=review_id))

    @app.post("/review/<review_id>/classification")
    def correct_classification(review_id: str):
        languages = [item.strip() for item in request.form.get("languages", "").split(",") if item.strip()]
        correction = {
            "document": {
                "document_kind": request.form.get("document_kind"),
                "template_state": request.form.get("template_state"),
                "contract_family": request.form.get("contract_family"),
                "contract_type": request.form.get("contract_type", "").strip() or None,
            },
            "languages": languages,
        }
        return run_action(review_id, lambda: workspace.apply_correction(review_id, reviewer(), correction))

    @app.post("/review/<review_id>/correction")
    def correct_structure(review_id: str):
        return run_action(
            review_id,
            lambda: workspace.apply_correction(
                review_id, reviewer(), json.loads(request.form.get("correction", "{}")),
            ),
        )

    @app.post("/review/<review_id>/sanitize")
    def sanitize(review_id: str):
        return run_action(
            review_id,
            lambda: workspace.sanitize(
                review_id, reviewer(), block_id=request.form.get("block_id", ""),
                start=int(request.form.get("start", "")), end=int(request.form.get("end", "")),
                variable_id=request.form.get("variable_id", ""),
                value_type=request.form.get("value_type", "text"),
            ),
        )

    @app.post("/review/<review_id>/decision")
    def decision(review_id: str):
        return run_action(
            review_id,
            lambda: workspace.decide(
                review_id, reviewer(), request.form.get("category", ""),
                request.form.get("status", ""), request.form.get("notes", ""),
            ),
        )

    @app.post("/review/<review_id>/final")
    def final(review_id: str):
        return run_action(
            review_id,
            lambda: workspace.finalize(
                review_id, reviewer(), request.form.get("disposition", ""), request.form.get("notes", ""),
            ),
        )

    @app.get("/source/<source_id>")
    def source(source_id: str):
        try:
            path, extension = workspace.source_path(source_id)
        except ReviewError:
            abort(404)
        return send_file(path, as_attachment=extension != ".pdf", download_name=path.name)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

