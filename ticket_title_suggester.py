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

import os
import re
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from functools import wraps

import requests
import anthropic
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Rate limiting: seconds to wait between API calls
ZENDESK_RATE_LIMIT_DELAY = float(os.environ.get("ZENDESK_RATE_LIMIT_DELAY", "0.5"))
CLAUDE_RATE_LIMIT_DELAY = float(os.environ.get("CLAUDE_RATE_LIMIT_DELAY", "1.0"))

# Retry configuration
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("RETRY_BASE_DELAY", "2.0"))

# Maximum allowed title length for suggestions
MAX_TITLE_LENGTH = 150

# Report output
PST = timezone(timedelta(hours=-8))
_now = datetime.now(PST)
NOW = _now.strftime("%Y-%m-%d_%I%M") + ("am" if _now.hour < 12 else "pm")
REPORT_PATH = os.environ.get("OUTPUT_FILE", f"/tmp/Title_Suggestions_{NOW}.xlsx")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Redaction
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CC_REDACTED]"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), "[TOKEN_REDACTED]"),
]


def redact_pii(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_BASE_DELAY):
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
    pass


def is_claude_limit_error(error: anthropic.APIError) -> bool:
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in CLAUDE_LIMIT_PATTERNS)

# ---------------------------------------------------------------------------
# Zendesk helpers
# ---------------------------------------------------------------------------


def zendesk_auth():
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)


def handle_zendesk_rate_limit(response: requests.Response, attempt: int = 0):
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logger.warning("Zendesk rate limit hit. Waiting %d seconds...", retry_after)
        time.sleep(retry_after)
        return True
    if response.status_code in (500, 502, 503, 504):
        delay = 2 ** attempt
        logger.warning(
            "Zendesk returned %d. Retrying in %ds (attempt %d)...",
            response.status_code, delay, attempt + 1,
        )
        time.sleep(delay)
        return True
    return False


@retry_with_backoff()
def fetch_open_tickets() -> list[dict]:
    tickets = []
    query = "type:ticket status<solved"
    url = f"{ZENDESK_BASE_URL}/search.json"
    params = {"query": query, "sort_by": "created_at", "sort_order": "desc", "per_page": 100}

    while url and len(tickets) < MAX_TICKETS:
        logger.info("Fetching tickets from: %s", url)
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.get(url, auth=zendesk_auth(), params=params, timeout=30)
            if handle_zendesk_rate_limit(resp, attempt):
                continue
            break

        resp.raise_for_status()
        data = resp.json()
        tickets.extend(data.get("results", []))
        url = data.get("next_page")
        params = None
        time.sleep(ZENDESK_RATE_LIMIT_DELAY)

    return tickets[:MAX_TICKETS]


@retry_with_backoff()
def fetch_ticket_comments(ticket_id: int) -> list[dict]:
    url = f"{ZENDESK_BASE_URL}/tickets/{ticket_id}/comments.json"
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, auth=zendesk_auth(), params={"per_page": 5}, timeout=30)
        if handle_zendesk_rate_limit(resp, attempt):
            continue
        break

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

Examples:

Current title: "Help"
Description: "I can't log into my CCAK dashboard. It says my session expired but I just logged in 5 minutes ago."
Suggested title: CCAK dashboard login fails with session expiration error

Current title: "Question about my account"
Description: "I purchased CSA STAR certification last month and need to consolidate it with my other purchases under our company account."
Suggested title: Consolidate CSA STAR certification purchase under company account

Current title: "CSA STAR registry  - update request"
Suggested title: KEEP
"""


def validate_suggestion(suggestion: str, ticket_id: int) -> str | None:
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

    pii_leak_patterns = [
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    ]
    for pattern in pii_leak_patterns:
        if pattern.search(suggestion):
            logger.warning(
                "Ticket #%s: Suggestion appears to contain PII, skipping.", ticket_id,
            )
            return None

    return suggestion


def suggest_title(client: anthropic.Anthropic, ticket: dict, comments: list[dict]) -> dict:
    current_title = ticket.get("subject", ticket.get("raw_subject", ""))
    description = ticket.get("description", "")

    redacted_title = redact_pii(current_title)
    redacted_description = redact_pii(description[:2000])

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
# Spreadsheet builder
# ---------------------------------------------------------------------------

DARK_HEADER   = "1F2D3D"
SUMMARY_BG    = "E8EAF6"
SUGGEST_FILL  = "E8F4E8";  ALT_SUGGEST  = "F0FAF0"
KEEP_FILL     = "F5F5F5";  ALT_KEEP     = "FAFAFA"
ERROR_FILL    = "FFE8E8";  ALT_ERROR    = "FFF0F0"
SKIP_FILL     = "FFF9C4";  ALT_SKIP     = "FFFDE7"
LINK_COLOR    = "1155CC"
SUGGEST_BADGE = "27AE60"
KEEP_BADGE    = "7F8C8D"
ERROR_BADGE   = "C0392B"
SKIP_BADGE    = "F57F17"

HEADERS = [
    "Ticket #", "Status", "Current Title", "Suggested Title",
    "Recommendation", "Reason", "Ticket Status", "Priority",
    "Created", "Last Updated",
]
WIDTHS = [10, 14, 44, 44, 18, 36, 12, 10, 13, 13]


def _border():
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)


def _cell(ws, row, col, value, bold=False, fc="000000",
          bg=None, wrap=False, align="left", size=11):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name="Arial", bold=bold, color=fc, size=size)
    c.alignment = Alignment(horizontal=align, vertical="top", wrap_text=wrap)
    if bg:
        c.fill = PatternFill("solid", start_color=bg)
    c.border = _border()
    return c


def _status_colors(status):
    if status == "Suggestion":
        return (SUGGEST_FILL, ALT_SUGGEST, SUGGEST_BADGE)
    elif status == "Error":
        return (ERROR_FILL, ALT_ERROR, ERROR_BADGE)
    elif status == "Skipped":
        return (SKIP_FILL, ALT_SKIP, SKIP_BADGE)
    return (KEEP_FILL, ALT_KEEP, KEEP_BADGE)


def format_date(iso_string: str | None) -> str:
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return str(iso_string)[:10]


def write_xlsx_report(rows: list[dict], output_path: str, run_meta: dict):
    wb = Workbook()
    ws = wb.active
    ws.title = "Title Suggestions"

    # ── Summary rows ──────────────────────────────────────────────────────
    summary_items = [
        ("Tickets Scanned:",  str(run_meta["tickets_scanned"]),  "1F2D3D"),
        ("Suggestions Made:", str(run_meta["suggestions_made"]), SUGGEST_BADGE),
        ("Titles Kept:",      str(run_meta["titles_kept"]),      KEEP_BADGE),
        ("Errors:",           str(run_meta["errors"]),           ERROR_BADGE),
        ("Skipped (Limit):",  str(run_meta.get("skipped", 0)),  SKIP_BADGE),
    ]
    for sr, (label, val, color) in enumerate(summary_items, 1):
        c = ws.cell(row=sr, column=1, value=label)
        c.font = Font(name="Arial", bold=True, size=12, color=color)
        c.fill = PatternFill("solid", start_color=SUMMARY_BG)
        c.alignment = Alignment(horizontal="right", vertical="center")

        c2 = ws.cell(row=sr, column=2, value=int(val))
        c2.font = Font(name="Arial", bold=True, size=14, color=color)
        c2.fill = PatternFill("solid", start_color=SUMMARY_BG)
        c2.alignment = Alignment(horizontal="left", vertical="center")

        for col in range(3, len(HEADERS) + 1):
            sc = ws.cell(row=sr, column=col)
            sc.fill = PatternFill("solid", start_color=SUMMARY_BG)

    ws.row_dimensions[len(summary_items) + 1].height = 6  # spacer

    # ── Header row ────────────────────────────────────────────────────────
    HEADER_ROW = len(summary_items) + 2
    for col, (h, w) in enumerate(zip(HEADERS, WIDTHS), 1):
        c = ws.cell(row=HEADER_ROW, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
        c.fill = PatternFill("solid", start_color=DARK_HEADER)
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = _border()
        ws.column_dimensions[get_column_letter(col)].width = w
    ws.row_dimensions[HEADER_ROW].height = 24

    # ── Data rows ─────────────────────────────────────────────────────────
    cur_row = HEADER_ROW + 1
    for r in rows:
        status = r.get("Status", "Keep Current")
        fill_main, fill_alt, badge_color = _status_colors(status)
        even = cur_row % 2 == 0
        bg = fill_main if not even else fill_alt

        # Col 1: Ticket # (hyperlinked)
        ticket_id = r.get("Ticket #", "")
        url = r.get("Ticket URL", "")
        lnk = ws.cell(row=cur_row, column=1, value=ticket_id)
        lnk.font = Font(name="Arial", bold=True, color=LINK_COLOR, underline="single", size=11)
        lnk.alignment = Alignment(horizontal="center", vertical="top")
        if url:
            lnk.hyperlink = url
        lnk.fill = PatternFill("solid", start_color=bg)
        lnk.border = _border()

        # Col 2: Status (colour-coded badge)
        _cell(ws, cur_row, 2, status, bold=True, fc=badge_color, bg=bg, align="center")

        # Col 3: Current Title
        _cell(ws, cur_row, 3, r.get("Current Title", ""), bg=bg, wrap=True)

        # Col 4: Suggested Title (bold green if suggestion)
        suggested = r.get("Suggested Title", "")
        if status == "Suggestion" and suggested:
            _cell(ws, cur_row, 4, suggested, bold=True, fc=SUGGEST_BADGE, bg=bg, wrap=True)
        else:
            _cell(ws, cur_row, 4, suggested, bg=bg, wrap=True)

        # Col 5: Recommendation
        _cell(ws, cur_row, 5, r.get("Recommendation", ""), bold=True, fc=badge_color, bg=bg, align="center")

        # Col 6: Reason
        _cell(ws, cur_row, 6, r.get("Reason", ""), bg=bg, wrap=True, size=10)

        # Col 7: Ticket Status
        _cell(ws, cur_row, 7, r.get("Ticket Status", ""), bg=bg, align="center")

        # Col 8: Priority
        _cell(ws, cur_row, 8, r.get("Priority", ""), bg=bg, align="center")

        # Col 9: Created
        _cell(ws, cur_row, 9, r.get("Created", ""), bg=bg, align="center")

        # Col 10: Last Updated
        _cell(ws, cur_row, 10, r.get("Last Updated", ""), bg=bg, align="center")

        ws.row_dimensions[cur_row].height = 48
        cur_row += 1

    ws.freeze_panes = f"A{HEADER_ROW + 1}"

    # ── Executive Summary sheet ───────────────────────────────────────────
    es = wb.create_sheet("Summary", 0)
    today_str = _now.strftime("%Y-%m-%d")

    es.merge_cells("A1:F1")
    title_cell = es.cell(row=1, column=1, value=f"Title Suggestion Report \u2014 {today_str}")
    title_cell.font = Font(name="Arial", bold=True, size=16, color="1F2D3D")
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    es.row_dimensions[1].height = 32

    stats = [
        ("Run Date",         run_meta["run_date"],          "1F2D3D"),
        ("Tickets Scanned",  run_meta["tickets_scanned"],   "1F2D3D"),
        ("Suggestions Made", run_meta["suggestions_made"],  SUGGEST_BADGE),
        ("Titles Kept",      run_meta["titles_kept"],       KEEP_BADGE),
        ("Errors",           run_meta["errors"],            ERROR_BADGE),
        ("Skipped (Limit)",  run_meta.get("skipped", 0),   SKIP_BADGE),
        ("PII Redaction",    "Enabled",                     "1F2D3D"),
        ("Mode",             "Log Only",                    "1F2D3D"),
    ]
    row_num = 3
    for label, val, color in stats:
        es.cell(row=row_num, column=1, value=label).font = Font(
            name="Arial", bold=True, size=11, color="333333")
        v = es.cell(row=row_num, column=2, value=val)
        v.font = Font(name="Arial", bold=True, size=13, color=color)
        v.alignment = Alignment(horizontal="left")
        row_num += 1

    # Top suggestions table
    suggestions_only = [r for r in rows if r.get("Status") == "Suggestion"]
    if suggestions_only:
        row_num += 1
        es.merge_cells(start_row=row_num, start_column=1, end_row=row_num, end_column=4)
        sec = es.cell(row=row_num, column=1, value=f"Suggested Title Changes ({len(suggestions_only)})")
        sec.font = Font(name="Arial", bold=True, size=13, color="FFFFFF")
        sec.fill = PatternFill("solid", start_color=DARK_HEADER)
        sec.alignment = Alignment(horizontal="left", vertical="center")
        es.row_dimensions[row_num].height = 24
        row_num += 1

        top_headers = ["#", "Current Title", "Suggested Title", "Reason"]
        top_widths = [10, 44, 44, 36]
        for ci, (h, w) in enumerate(zip(top_headers, top_widths), 1):
            c = es.cell(row=row_num, column=ci, value=h)
            c.font = Font(name="Arial", bold=True, size=10, color="666666")
            c.border = _border()
            es.column_dimensions[get_column_letter(ci)].width = w
        row_num += 1

        for r in suggestions_only[:10]:
            tid_cell = es.cell(row=row_num, column=1, value=r.get("Ticket #", ""))
            tid_url = r.get("Ticket URL", "")
            tid_cell.font = Font(name="Arial", color=LINK_COLOR, underline="single", size=11)
            if tid_url:
                tid_cell.hyperlink = tid_url
            tid_cell.border = _border()

            es.cell(row=row_num, column=2, value=r.get("Current Title", "")[:60]).border = _border()

            sug_cell = es.cell(row=row_num, column=3, value=r.get("Suggested Title", "")[:60])
            sug_cell.font = Font(name="Arial", bold=True, color=SUGGEST_BADGE, size=11)
            sug_cell.border = _border()

            es.cell(row=row_num, column=4, value=r.get("Reason", "")).border = _border()

            es.row_dimensions[row_num].height = 28
            row_num += 1

    wb.save(output_path)
    logger.info("Spreadsheet saved \u2192 %s (%d data rows)", output_path, len(rows))


# -- Google Drive upload -----------------------------------------------------

def upload_to_gdrive(file_path):
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

        if creds_info.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        else:
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
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            resumable=True,
        )

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
            logger.error("CLAUDE API LIMIT REACHED \u2014 stopping early.")
            logger.error("Reason: %s", e)
            logger.error("Processed %d/%d tickets before limit was hit.", i - 1, len(tickets))
            logger.error("Add credits at https://console.anthropic.com/settings/billing")
            logger.error("=" * 60)
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

    # Print summary
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

    run_meta = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickets_scanned": len(tickets),
        "suggestions_made": suggestion_count,
        "titles_kept": keep_count,
        "errors": errors,
        "skipped": skipped,
    }

    write_xlsx_report(report_rows, REPORT_PATH, run_meta)

    upload_to_gdrive(REPORT_PATH)

    if suggestion_count == 0:
        logger.info("All ticket titles look good \u2014 nothing to suggest!")

    if errors > 0 and errors == len(tickets) and skipped == 0:
        logger.error("All tickets failed to process. Exiting with error.")
        sys.exit(1)
    elif skipped > 0:
        logger.warning("Run completed partially: %d tickets skipped due to Claude API limit.", skipped)


if __name__ == "__main__":
    main()
