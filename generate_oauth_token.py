#!/usr/bin/env python3
"""
generate_oauth_token.py - One-time helper to generate OAuth2 refresh token
for Google Drive API access.

Prerequisites:
    1. Go to https://console.cloud.google.com/
    2. Create a project (or use an existing one)
    3. Enable the Google Drive API
    4. Go to APIs & Services > Credentials
    5. Create an OAuth 2.0 Client ID (type: Desktop app)
    6. Download the client secret JSON file

Usage:
    pip install google-auth-oauthlib
    python generate_oauth_token.py /path/to/client_secret.json

This will open a browser for you to authorize the app, then print
the credentials JSON to store as the GDRIVE_CREDENTIALS_JSON GitHub secret.
"""

import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_oauth_token.py <client_secret.json>")
        print()
        print("Download the client secret JSON from:")
        print("  https://console.cloud.google.com/apis/credentials")
        sys.exit(1)

    client_secret_file = sys.argv[1]

    print("Starting OAuth2 authorization flow...")
    print("A browser window will open for you to authorize access.")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    creds = flow.run_local_server(port=0)

    # Build the credentials JSON for the GitHub secret
    creds_json = {
        "type": "authorized_user",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
    }

    print()
    print("=" * 60)
    print("SUCCESS! Copy the JSON below and save it as a GitHub secret")
    print("named GDRIVE_CREDENTIALS_JSON:")
    print("=" * 60)
    print()
    print(json.dumps(creds_json, indent=2))
    print()
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Go to your repo Settings > Secrets and variables > Actions")
    print("  2. Add a new secret: GDRIVE_CREDENTIALS_JSON")
    print("  3. Paste the JSON above as the value")
    print("  4. Also add GDRIVE_FOLDER_ID with your target folder ID")
    print("     (find it in the Google Drive folder URL after /folders/)")


if __name__ == "__main__":
    main()
