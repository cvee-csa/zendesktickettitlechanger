"""
Zendesk Ticket Title Suggester

Queries Zendesk for open tickets, analyzes their content using the Claude API,
and suggests more meaningful titles based on ticket context.

Guardrails:
- Rate limiting for both Zendesk and Claude API calls
- PII redaction before sending ticket content to Claude
- Retry logic with exponential backoff
- Configurable max ticket cap to control costs
- Log-only mode by default (no ticket modifications)
- Title length and content validation on suggestions
"""

import csv
import os
import re
import sys
import json
import time
import logging
from datetime import datetime
from functools import wraps

import requests
import anthropic

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZENDESK_SUBDOMAIN = (os.environ.get("ZENDESK_SUBDOMAIN") or "").strip()
ZENDESK_EMAIL = (os.environ.get("ZENDESK_EMAIL") or "").strip()
ZENDESK_API_TOKEN = (os.environ.get("ZENDESK_API_TOKEN") or "").strip()
ANTHROPIC_API_KEY = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

# Google Drive upload (optional)
GDRIVE_SA_JSON   = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")  # full JSON key string
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")            # folder or Shared Drive folder ID

ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"

# How many tickets to process per run (to control API costs)
MAX_TICKETS = int(os.environ.get("MAX_TICKETS", "50"))

# Claude model to use
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Rate limiting: seconds to wait between API calls
ZENDESK_RATE_LIMIT_DELAY = float(os.environ.get("ZENDESK_RATE_LIMIT_DELAY", "0.5"))
CLAUDE_RATE_LIMIT_DELAY = float(os.environ.get("CLAUDE_RATE_LIMIT_DELAY", "1.0"))

# Retry configuration
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", "2.0"))

# Maximum allowed title length for suggestions
MAX_TITLE_LENGTH = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Redaction
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL_REDACTED]"),
    # Phone numbers (various formats)
    (re.compile(r"\b(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Credit card numbers (basic pattern)
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CC_REDACTED]"),
    # IP addresses
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
    # API keys / tokens (long hex or base64 strings)
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), "[TOKEN_REDACTED]"),
]


def redact_pii(text: str) -> str:
    """Remove personally identifiable information from text before sending to Claude."""
    if not text:
        return text
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
    """Decorator that retries a function with exponential backoff on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, anthropic.APIError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1, max_retries + 1, func.__name__, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_retries + 1, func.__name__, e,
                        )
            raise last_exception
        return wrapper
    return decorator



# ---------------------------------------------------------------------------
# Claude API limit detection
# ---------------------------------------------------------------------------

# Error messages from the Anthropic API that indicate a non-retryable billing
# or token limit problem.  When one of these is detected the script stops
# early instead of burning through every remaining ticket.
CLAUDE_LIMIT_PATTERNS = [
    "credit balance is too low",
    "monthly spend limit",
    "rate limit exceeded",
    "token limit",
    "billing",
    "exceeded your current quota",
    "insufficient_quota",
]


class ClaudeTokenLimitError(Exception):
    """Raised when the Claude API reports a billing or token limit issue."""
    pass


def is_claude_limit_error(error: anthropic.APIError) -> bool:
    """Check whether an Anthropic API error indicates a token/credit limit."""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in CLAUDE_LIMIT_PATTERNS)

# ---------------------------------------------------------------------------
# Zendesk helpers
# ---------------------------------------------------------------------------


def zendesk_auth():
    """Return the (email/token, api_token) tuple for Zendesk basic auth."""
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)


def handle_zendesk_rate_limit(response: requests.Response):
    """Check for Zendesk 429 rate limit and wait if needed."""
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logger.warning("Zendesk rate limit hit. Waiting %d seconds...", retry_after)
        time.sleep(retry_after)
        return True
    return False


@retry_with_backoff()
def fetch_open_tickets() -> list[dict]:
    """Fetch open (and new) tickets from Zendesk using the Search API."""
    tickets = []
    query = "type:ticket status<solved"
    url = f"{ZENDESK_BASE_URL}/search.json"
    params = {"query": query, "sort_by": "created_at", "sort_order": "desc", "per_page": 100}

    while url and len(tickets) < MAX_TICKETS:
        logger.info("Fetching tickets from: %s", url)
        resp = requests.get(url, auth=zendesk_auth(), params=params, timeout=30)

        if handle_zendesk_rate_limit(resp):
            continue  # retry the same request

        resp.raise_for_status()
        data = resp.json()
        tickets.extend(data.get("results", []))
        url = data.get("next_page")
        params = None  # next_page URL already contains query params
        time.sleep(ZENDESK_RATE_LIMIT_DELAY)

    return tickets[:MAX_TICKETS]


@retry_with_backoff()
def fetch_ticket_comments(ticket_id: int) -> list[dict]:
    """Fetch the first few comments on a ticket to provide context."""
    url = f"{ZENDESK_BASE_URL}/tickets/{ticket_id}/comments.json"
    resp = requests.get(url, auth=zendesk_auth(), params={"per_page": 5}, timeout=30)

    if handle_zendesk_rate_limit(resp):
        # Retry once after rate limit
        time.sleep(int(resp.headers.get("Retry-After", 60)))
        resp = requests.get(url, auth=zendesk_auth(), params={"per_page": 5}, timeout=30)

    resp.raise_for_status()
    return resp.json().get("comments", [])


# ---------------------------------------------------------------------------
# Claude title suggestion
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Zendesk ticket title optimizer. Your job is to read the current title \
and body of a support ticket and suggest a clearer, more descriptive title that \
will help support agents quickly understand what the ticket is about.

Rules:
- Keep the suggested title under 100 characters.
- Be specific: include the product, feature, or error if mentioned.
- Use sentence case.
- Do not add ticket IDs or status to the title.
- Do not include any personal information (names, emails, account numbers) in the title.
- If the current title is already clear and descriptive, respond with "KEEP" and nothing else.
- Respond with ONLY the suggested title (or "KEEP"). No explanation, no quotes.
"""


def validate_suggestion(suggestion: str, ticket_id: int) -> str | None:
    """Validate a suggested title before accepting it."""
    if not suggestion or not suggestion.strip():
        logger.warning("Ticket #%s: Empty suggestion received, skipping.", ticket_id)
        return None

    suggestion = suggestion.strip().strip('"').strip("'")

    if len(suggestion) > MAX_TITLE_LENGTH:
        logger.warning(
            "Ticket #%s: Suggestion too long (%d chars), skipping: %s",
            ticket_id, len(suggestion), suggestion[:80] + "...",
        )
        return None

    # Reject suggestions that look like they contain PII the model leaked back
    pii_leak_patterns = [
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),  # CC
    ]
    for pattern in pii_leak_patterns:
        if pattern.search(suggestion):
            logger.warning(
                "Ticket #%s: Suggestion appears to contain PII, skipping.", ticket_id,
            )
            return None

    return suggestion


def suggest_title(client: anthropic.Anthropic, ticket: dict, comments: list[dict]) -> dict:
    """Use Claude to suggest a better title for the given ticket.

    Returns a dict with keys:
        suggested_title: str or empty string
        status: "Suggestion" | "Keep Current" | "Error"
        reason: short explanation
    """
    current_title = ticket.get("subject", ticket.get("raw_subject", ""))
    description = ticket.get("description", "")

    # Redact PII from content before sending to Claude
    redacted_title = redact_pii(current_title)
    redacted_description = redact_pii(description[:2000])

    # Build context from first few comments (redacted)
    comment_texts = []
    for c in comments[:3]:
        body = c.get("plain_body") or c.get("body", "")
        if body:
            comment_texts.append(redact_pii(body[:1000]))

    user_message = f"""Current title: {redacted_title}

Ticket description:
{redacted_description}

Additional comments:
{chr(10).join(comment_texts) if comment_texts else "(none)"}"""

    try:
        # Rate limit Claude API calls
        time.sleep(CLAUDE_RATE_LIMIT_DELAY)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        suggestion = response.content[0].text.strip()
        if suggestion.upper() == "KEEP":
            return {"suggested_title": "", "status": "Keep Current", "reason": "Title is already clear and descriptive"}
        validated = validate_suggestion(suggestion, ticket["id"])
        if validated:
            return {"suggested_title": validated, "status": "Suggestion", "reason": "Title could be more descriptive"}
        return {"suggested_title": "", "status": "Keep Current", "reason": "Suggestion failed validation"}
    except anthropic.APIError as e:
        if is_claude_limit_error(e):
            logger.error("Claude API limit reached for ticket #%s: %s", ticket["id"], e)
            raise ClaudeTokenLimitError(str(e))
        logger.error("Claude API error for ticket #%s: %s", ticket["id"], e)
        return {"suggested_title": "", "status": "Error", "reason": str(e)[:120]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def format_date(iso_string: str | None) -> str:
    """Convert an ISO date string to MM/DD/YYYY format."""
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return str(iso_string)[:10]


# ---------------------------------------------------------------------------
# CSV Report columns (inspired by ESC/RARC report structure)
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "Ticket #",
    "Status",
    "Current Title",
    "Suggested Title",
    "Recommendation",
    "Reason",
    "Ticket URL",
    "Ticket Status",
    "Priority",
    "Created",
    "Last Updated",
]


def write_csv_report(rows: list[dict], output_path: str, run_meta: dict):
    """Write the title suggestion report as a CSV file.

    Includes a summary header block followed by one row per ticket.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # --- Summary header block (mirrors the ESC/RARC report style) ---
        writer.writerow(["Title Suggestion Report"])
        writer.writerow(["Run Date", run_meta["run_date"]])
        writer.writerow(["Tickets Scanned", run_meta["tickets_scanned"]])
        writer.writerow(["Suggestions Made", run_meta["suggestions_made"]])
        writer.writerow(["Titles Kept", run_meta["titles_kept"]])
        writer.writerow(["Errors", run_meta["errors"]])
        writer.writerow(["Skipped (API Limit)", run_meta.get("skipped", 0)])
        writer.writerow(["PII Redaction", "Enabled"])
        writer.writerow(["Mode", "Log Only"])
        writer.writerow([])  # blank separator row

        # --- Column headers ---
        writer.writerow(CSV_COLUMNS)

        # --- Data rows ---
        for row in rows:
            writer.writerow([row.get(col, "") for col in CSV_COLUMNS])

    logger.info("CSV report written to %s (%d data rows)", output_path, len(rows))



# -- Google Drive upload -----------------------------------------------------

def upload_to_gdrive(file_path):
    """
    Upload file_path to Google Drive (works with both My Drive and Shared Drives).
    Requires GDRIVE_SERVICE_ACCOUNT_JSON and GDRIVE_FOLDER_ID env vars.
    Skips silently if either is missing or google-auth libs are not installed.
    """
    if not GDRIVE_AVAILABLE:
        print("  [Drive] google-auth libraries not installed -- skipping upload.")
        return
    if not GDRIVE_SA_JSON or not GDRIVE_FOLDER_ID:
        print("  [Drive] GDRIVE_SERVICE_ACCOUNT_JSON or GDRIVE_FOLDER_ID not set -- skipping.")
        return

    try:
        sa_json = GDRIVE_SA_JSON.strip()
        if not sa_json:
            print("  [Drive] GDRIVE_SERVICE_ACCOUNT_JSON is blank after stripping whitespace -- skipping.")
            return
        creds_info = json.loads(sa_json)

        # Support both service account keys and OAuth user credentials
        if creds_info.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        else:
            # OAuth user credentials (from generate_oauth_token.py / InstalledAppFlow)
            from google.oauth2.credentials import Credentials
            creds = Credentials(
                token=creds_info.get("token"),
                refresh_token=creds_info["refresh_token"],
                token_uri=creds_info.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=creds_info["client_id"],
                client_secret=creds_info["client_secret"],
                scopes=creds_info.get("scopes"),
            )

        service = build("drive", "v3", credentials=creds)

        file_name = os.path.basename(file_path)
        file_metadata = {"name": file_name, "parents": [GDRIVE_FOLDER_ID]}
        media = MediaFileUpload(
            file_path,
            mimetype="text/csv",
            resumable=True,
        )

        # supportsAllDrives=True makes it work for both Shared Drives and My Drive
        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        ).execute()

        print(f"  [Drive] Uploaded: {uploaded['name']}")
        print(f"  [Drive] View at : {uploaded.get('webViewLink', '(no link)')}")

    except json.JSONDecodeError as e:
        print(f"  [Drive] GDRIVE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
        print(f"  [Drive] Secret starts with: {repr(GDRIVE_SA_JSON[:80])}")
        print("  [Drive] Check that the secret contains the raw JSON (not base64 or a file path).")
    except Exception as e:
        print(f"  [Drive] Upload failed: {e}")


def main():
    # Validate required env vars
    missing = []
    for var in ("ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Zendesk Ticket Title Suggester")
    logger.info("=" * 60)
    logger.info("Mode: LOG ONLY (no tickets will be modified)")
    logger.info("Max tickets: %d", MAX_TICKETS)
    logger.info("Claude model: %s", CLAUDE_MODEL)
    logger.info("Rate limits: Zendesk=%.1fs, Claude=%.1fs", ZENDESK_RATE_LIMIT_DELAY, CLAUDE_RATE_LIMIT_DELAY)
    logger.info("Retry config: max_retries=%d, base_delay=%.1fs", MAX_RETRIES, RETRY_BASE_DELAY)
    logger.info("PII redaction: ENABLED")
    logger.info("=" * 60)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    logger.info("Fetching open tickets from Zendesk (%s)...", ZENDESK_SUBDOMAIN)
    try:
        tickets = fetch_open_tickets()
    except requests.RequestException as e:
        logger.error("Failed to fetch tickets from Zendesk after retries: %s", e)
        sys.exit(1)

    logger.info("Found %d open tickets to analyze.", len(tickets))

    report_rows: list[dict] = []
    suggestion_count = 0
    keep_count = 0
    errors = 0

    for i, ticket in enumerate(tickets, 1):
        ticket_id = ticket["id"]
        current_title = ticket.get("subject", ticket.get("raw_subject", ""))
        ticket_status = ticket.get("status", "")
        ticket_priority = ticket.get("priority", "") or ""
        created_at = ticket.get("created_at", "")
        updated_at = ticket.get("updated_at", "")
        ticket_url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}"

        logger.info("[%d/%d] Analyzing ticket #%s: %s", i, len(tickets), ticket_id, current_title)

        try:
            comments = fetch_ticket_comments(ticket_id)
        except requests.RequestException as e:
            logger.error("  \u2192 Failed to fetch comments for ticket #%s: %s", ticket_id, e)
            errors += 1
            report_rows.append({
                "Ticket #": ticket_id,
                "Status": "Error",
                "Current Title": current_title,
                "Suggested Title": "",
                "Recommendation": "Review Manually",
                "Reason": f"Failed to fetch comments: {str(e)[:80]}",
                "Ticket URL": ticket_url,
                "Ticket Status": ticket_status.capitalize(),
                "Priority": ticket_priority.capitalize(),
                "Created": format_date(created_at),
                "Last Updated": format_date(updated_at),
            })
            continue

        try:
            result = suggest_title(client, ticket, comments)
        except ClaudeTokenLimitError as e:
            logger.error("=" * 60)
            logger.error("CLAUDE API LIMIT REACHED — stopping early.")
            logger.error("Reason: %s", e)
            logger.error("Processed %d/%d tickets before limit was hit.", i - 1, len(tickets))
            logger.error("Add credits at https://console.anthropic.com/settings/billing")
            logger.error("=" * 60)
            # Mark remaining tickets (including this one) as skipped
            for remaining_ticket in tickets[i - 1:]:
                rt_id = remaining_ticket["id"]
                rt_title = remaining_ticket.get("subject", remaining_ticket.get("raw_subject", ""))
                rt_url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{rt_id}"
                report_rows.append({
                    "Ticket #": rt_id,
                    "Status": "Skipped",
                    "Current Title": rt_title,
                    "Suggested Title": "",
                    "Recommendation": "Re-run After Adding Credits",
                    "Reason": "Claude API limit reached",
                    "Ticket URL": rt_url,
                    "Ticket Status": remaining_ticket.get("status", "").capitalize(),
                    "Priority": (remaining_ticket.get("priority", "") or "").capitalize(),
                    "Created": format_date(remaining_ticket.get("created_at", "")),
                    "Last Updated": format_date(remaining_ticket.get("updated_at", "")),
                })
            break

        suggested_title = result["suggested_title"]
        status = result["status"]
        reason = result["reason"]

        if status == "Suggestion":
            suggestion_count += 1
            recommendation = "Update Title"
            logger.info("  \u2192 Suggested: %s", suggested_title)
        elif status == "Error":
            errors += 1
            recommendation = "Review Manually"
            logger.info("  \u2192 Error analyzing title.")
        else:
            keep_count += 1
            recommendation = "No Action Needed"
            logger.info("  \u2192 Title is fine, no change suggested.")

        report_rows.append({
            "Ticket #": ticket_id,
            "Status": status,
            "Current Title": current_title,
            "Suggested Title": suggested_title,
            "Recommendation": recommendation,
            "Reason": reason,
            "Ticket URL": ticket_url,
            "Ticket Status": ticket_status.capitalize(),
            "Priority": ticket_priority.capitalize(),
            "Created": format_date(created_at),
            "Last Updated": format_date(updated_at),
        })

    # Print summary to stdout
    print("\n" + "=" * 80)
    print(f"TITLE SUGGESTION REPORT \u2014 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    skipped = sum(1 for r in report_rows if r["Status"] == "Skipped")
    print(f"Tickets scanned: {len(tickets)}")
    print(f"Suggestions made: {suggestion_count}")
    print(f"Titles kept: {keep_count}")
    print(f"Errors encountered: {errors}")
    print(f"Skipped (API limit): {skipped}")
    print(f"PII redaction: enabled")
    print("=" * 80)

    for row in report_rows:
        if row["Status"] == "Suggestion":
            print(f"\nTicket #{row['Ticket #']}  {row['Ticket URL']}")
            print(f"  Current:   {row['Current Title']}")
            print(f"  Suggested: {row['Suggested Title']}")

    print("\n" + "=" * 80)

    # Build run metadata for the report header
    run_meta = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickets_scanned": len(tickets),
        "suggestions_made": suggestion_count,
        "titles_kept": keep_count,
        "errors": errors,
        "skipped": skipped,
    }

    # Write CSV report
    csv_path = os.environ.get("OUTPUT_FILE", "title_suggestions.csv")
    write_csv_report(report_rows, csv_path, run_meta)

    # Upload CSV to Google Drive (skips silently if not configured)
    upload_to_gdrive(csv_path)

    if suggestion_count == 0:
        logger.info("All ticket titles look good \u2014 nothing to suggest!")

    # Exit with error code only if real errors (not token limit skips)
    if errors > 0 and errors == len(tickets) and skipped == 0:
        logger.error("All tickets failed to process. Exiting with error.")
        sys.exit(1)
    elif skipped > 0:
        logger.warning("Run completed partially: %d tickets skipped due to Claude API limit.", skipped)


if __name__ == "__main__":
    main()
