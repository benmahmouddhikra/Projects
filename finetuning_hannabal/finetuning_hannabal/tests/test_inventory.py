from __future__ import annotations

from inventory import EXPECTED_COUNTS, assert_expected_counts, extension_counts, inventory_corpus


def test_inventory_counts_and_known_duplicate(repository_root):
    records = inventory_corpus(repository_root / "templates", repository_root)
    assert len(records) == 1276
    assert extension_counts(records) == EXPECTED_COUNTS
    assert_expected_counts(records)
    duplicate_groups = {}
    for record in records:
        group_id = record["exact_duplicate"]["group_id"]
        if group_id:
            duplicate_groups.setdefault(group_id, set()).add(record["relative_path"])
    assert list(duplicate_groups.values()) == [{
        "templates/Émirats arabes unis/CSP_Agreement_Final (1).pdf",
        "templates/Émirats arabes unis/CSP_Agreement_Final.pdf",
    }]


def test_pilot_manifest_is_fixed_and_covers_all_folders(feature_root, repository_root):
    import json
    from pilot import validate_fixed_pilot

    manifest = json.loads((feature_root / "config" / "pilot-manifest.json").read_text(encoding="utf-8"))
    validate_fixed_pilot(manifest, repository_root)

