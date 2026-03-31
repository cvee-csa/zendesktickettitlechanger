#!/usr/bin/env python3
"""
upload_to_drive.py - Upload a file to Google Drive.

Supports OAuth2 (refresh token) credentials stored as a JSON string
in the GDRIVE_CREDENTIALS_JSON environment variable.

Usage:
    python upload_to_drive.py <file_path>

Required env vars:
    GDRIVE_CREDENTIALS_JSON  - JSON string with OAuth2 credentials:
        {
            "type": "authorized_user",
            "client_id": "...",
            "client_secret": "...",
            "refresh_token": "..."
        }
    GDRIVE_FOLDER_ID         - Target Google Drive folder ID

Optional env vars:
    GDRIVE_FILENAME          - Override the uploaded file name
"""

import json
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# -- Mime type mapping --------------------------------------------------------
MIME_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}


def get_credentials():
    """Build Google OAuth2 credentials from the GDRIVE_CREDENTIALS_JSON env var."""
    raw = (os.environ.get("GDRIVE_CREDENTIALS_JSON") or "").strip()
    if not raw:
        print("[Drive] ERROR: GDRIVE_CREDENTIALS_JSON is not set or empty.")
        sys.exit(1)

    try:
        creds_info = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Drive] ERROR: GDRIVE_CREDENTIALS_JSON is not valid JSON: {e}")
        sys.exit(1)

    required = ["client_id", "client_secret", "refresh_token"]
    missing = [k for k in required if not creds_info.get(k)]
    if missing:
        print(f"[Drive] ERROR: Missing fields in credentials JSON: {', '.join(missing)}")
        sys.exit(1)

    return Credentials(
        token=None,
        refresh_token=creds_info["refresh_token"],
        client_id=creds_info["client_id"],
        client_secret=creds_info["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )


def upload_file(file_path):
    """Upload a file to Google Drive and return the API response."""
    folder_id = (os.environ.get("GDRIVE_FOLDER_ID") or "").strip()
    if not folder_id:
        print("[Drive] ERROR: GDRIVE_FOLDER_ID is not set or empty.")
        sys.exit(1)

    if not os.path.isfile(file_path):
        print(f"[Drive] ERROR: File not found: {file_path}")
        sys.exit(1)

    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    file_name = os.environ.get("GDRIVE_FILENAME", "").strip() or os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = MIME_TYPES.get(ext, "application/octet-stream")

    file_metadata = {
        "name": file_name,
        "parents": [folder_id],
    }

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

    print(f"[Drive] Uploading '{file_name}' to folder {folder_id}...")
    uploaded = (
        service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )

    print(f"[Drive] Success! File ID: {uploaded['id']}")
    print(f"[Drive] Name:    {uploaded['name']}")
    print(f"[Drive] View at: {uploaded.get('webViewLink', '(no link)')}")
    return uploaded


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_drive.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    upload_file(file_path)


if __name__ == "__main__":
    main()
