import hashlib


def evaluate_nonce_roundtrip(expected, downloaded, remote_sha256=None, commit=None):
    expected_sha256 = hashlib.sha256(expected).hexdigest()
    restored_sha256 = hashlib.sha256(downloaded).hexdigest()
    exact = len(expected) == len(downloaded) and expected_sha256 == restored_sha256
    metadata_matches = remote_sha256 is None or remote_sha256 == expected_sha256
    verified = bool(commit) and exact and metadata_matches
    return {
        "status": "verified" if verified else "failed",
        "round_trip_verified": verified,
        "commit_pinned": bool(commit),
        "expected_sha256": expected_sha256,
        "restored_sha256": restored_sha256,
        "bytes": len(expected),
        "restored_bytes": len(downloaded),
        "remote_sha256": remote_sha256,
        "remote_kind": "git" if remote_sha256 is None else "lfs",
        "remote_metadata_matches": metadata_matches,
        "exact_downloaded_bytes": exact,
    }
