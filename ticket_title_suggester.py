"""
Zendesk Ticket Title Suggester (Rule-Based)

Queries Zendesk for open tickets, analyzes their titles using heuristic rules,
and suggests more meaningful titles based on ticket context.

No external AI API required — uses keyword extraction and pattern matching.

Guardrails:
- Rate limiting for Zendesk API calls
- PII redaction in suggestions
- Retry logic with exponential backoff
- Configurable max ticket cap
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
from collections import Counter

import requests
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

# Google Drive upload (optional)
GDRIVE_SA_JSON   = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")

ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2"

# How many tickets to process per run
MAX_TICKETS = int(os.environ.get("MAX_TICKETS", "50"))

# Rate limiting: seconds to wait between Zendesk API calls
ZENDESK_RATE_LIMIT_DELAY = float(os.environ.get("ZENDESK_RATE_LIMIT_DELAY", "0.5"))

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
                except requests.RequestException as e:
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
# Rule-based title analysis
# ---------------------------------------------------------------------------

# Titles that are clearly too vague (case-insensitive exact or near-exact match)
VAGUE_TITLES = {
    "help", "help!", "help me", "please help", "need help",
    "question", "a question", "quick question",
    "issue", "problem", "error", "bug",
    "request", "new request", "support request",
    "urgent", "urgent!", "asap",
    "hi", "hello", "hey", "good morning", "good afternoon",
    "follow up", "follow-up", "following up",
    "update", "status update", "checking in",
    "re:", "fw:", "fwd:",
    "test", "testing", "test ticket",
    "ticket", "new ticket", "support ticket",
    "inquiry", "general inquiry",
    "info", "information", "information needed",
    "assistance", "need assistance", "assistance needed",
    "account", "my account", "account issue", "account problem",
    "access", "access issue", "can't access", "cannot access",
    "login", "log in", "login issue", "can't login", "cannot login",
    "password", "password reset", "forgot password",
    "billing", "invoice", "payment",
    "certificate", "certification",
    "registration", "register",
    "question about my account",
    "not working", "doesn't work", "it's not working",
    "broken", "something is broken",
    "none", "n/a", "na", "no subject", "(no subject)",
    "...", ".", "-", "--", "___",
}

# Words that don't help identify what the ticket is about
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "they", "them", "their",
    "this", "that", "these", "those", "am", "of", "in", "to", "for", "with",
    "on", "at", "from", "by", "about", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "also", "get", "got", "getting", "need", "needed",
    "want", "wanted", "please", "thanks", "thank", "hi", "hello", "hey",
    "dear", "sir", "madam", "team", "support", "help", "issue", "problem",
    "able", "unable", "trying", "try", "tried", "like", "know", "think",
    "still", "seem", "seems", "new", "using", "use", "used",
    "email_redacted", "phone_redacted", "token_redacted",
    "ip_redacted", "ssn_redacted", "cc_redacted",
}

# Known products/features to boost in title suggestions
KNOWN_PRODUCTS = [
    "CSA STAR", "STAR Registry", "STAR Level", "STAR Attestation",
    "CCSK", "CCAK", "CCZT", "Certificate of Cloud Security Knowledge",
    "Certificate of Cloud Auditing Knowledge",
    "Cloud Controls Matrix", "CCM", "CAIQ",
    "AI Safety", "IoT", "Zero Trust",
    "STARWatch", "STAR Watch",
    "GRC Stack", "GRC", "Trusted Cloud Provider",
    "SOC 2", "ISO 27001", "ISO 27017", "ISO 27018",
    "Shared Drive", "Google Drive", "SSO", "MFA", "API",
    "Zendesk", "Dashboard", "Portal",
]

# Compile product patterns for matching (case-insensitive)
PRODUCT_PATTERNS = [(re.compile(re.escape(p), re.IGNORECASE), p) for p in KNOWN_PRODUCTS]


def is_vague_title(title: str) -> bool:
    """Check if a title is too vague/generic to be useful."""
    cleaned = title.strip().lower()
    cleaned = re.sub(r"^(re|fw|fwd)\s*:\s*", "", cleaned).strip()

    # Exact match against vague titles
    if cleaned in VAGUE_TITLES:
        return True

    # Too short (under 3 words or under 10 chars)
    words = cleaned.split()
    if len(words) < 3 or len(cleaned) < 10:
        return True

    # Only contains stop words
    meaningful_words = [w for w in words if w.lower() not in STOP_WORDS]
    if len(meaningful_words) == 0:
        return True

    # Is just a person's name pattern (1-3 capitalized words, no other content)
    name_pattern = re.compile(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,2}$")
    if name_pattern.match(title.strip()):
        return True

    return False


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """Extract the most relevant keywords from text."""
    if not text:
        return []

    text = redact_pii(text)

    # First, check for known product mentions
    found_products = []
    for pattern, product_name in PRODUCT_PATTERNS:
        if pattern.search(text):
            found_products.append(product_name)

    # Tokenize and count meaningful words
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    word_counts = Counter(meaningful)

    # Get top keywords (excluding product names already found)
    product_words = set()
    for p in found_products:
        product_words.update(w.lower() for w in p.split())

    top_words = [
        word for word, _ in word_counts.most_common(max_keywords + 10)
        if word not in product_words
    ][:max_keywords]

    return found_products + top_words


def build_suggested_title(title: str, description: str, comments: list[dict]) -> str:
    """Build a suggested title from the ticket description and comments."""
    # Combine text sources
    all_text = description or ""
    for c in comments[:3]:
        body = c.get("plain_body") or c.get("body", "")
        if body:
            all_text += " " + body

    all_text = all_text[:3000]  # Cap text length

    # Extract keywords
    keywords = extract_keywords(all_text)

    if not keywords:
        return ""

    # Check for common action patterns in the description
    desc_lower = (description or "").lower()[:1500]
    action = ""

    action_patterns = [
        (r"(can't|cannot|unable to|couldn't|could not)\s+(access|log\s*in|sign\s*in|connect|view|open|download|upload|use|find|see|load|reach)",
         "Cannot {match}"),
        (r"(need|want|would like|requesting|request)\s+(?:to\s+)?(add|remove|change|update|reset|create|delete|modify|enable|disable|upgrade|renew|transfer|consolidate)",
         "Request to {match}"),
        (r"(how|where)\s+(?:do|can|to)\s+(.*?)[\?\.]",
         "Question: How to {match}"),
        (r"(error|failed|failure|crash|broken|bug|not working|doesn'?t work|isn'?t working)",
         "Error"),
        (r"(expired?|expir(?:ing|ation))",
         "Expiration issue"),
        (r"(invoice|billing|charge|payment|refund|credit)",
         "Billing inquiry"),
        (r"(certificate|certification|exam|badge|credential)",
         "Certification"),
        (r"(registry|listing|profile|entry)",
         "Registry/listing"),
    ]

    for pattern, template in action_patterns:
        match = re.search(pattern, desc_lower)
        if match:
            if "{match}" in template:
                # Get the matched action words
                groups = match.groups()
                relevant = groups[-1] if len(groups) > 1 else groups[0]
                action = template.replace("{match}", relevant.strip())
            else:
                action = template
            break

    # Build the title
    products = [k for k in keywords if any(p == k for _, p in PRODUCT_PATTERNS)]
    other_keywords = [k for k in keywords if k not in products][:4]

    if products and action:
        title = f"{products[0]}: {action}"
    elif products:
        context = " ".join(other_keywords[:3]).capitalize() if other_keywords else "inquiry"
        title = f"{products[0]} — {context}"
    elif action:
        context = " ".join(other_keywords[:3]) if other_keywords else ""
        title = f"{action}" + (f" — {context}" if context else "")
    else:
        # Fallback: just use top keywords
        title = " ".join(other_keywords[:5]).capitalize()

    # Clean up the title
    title = title.strip(" —-:")
    title = re.sub(r"\s+", " ", title)

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]

    # Enforce max length
    if len(title) > 100:
        title = title[:97] + "..."

    return title if len(title) >= 10 else ""


def suggest_title(ticket: dict, comments: list[dict]) -> dict:
    """Analyze a ticket title using heuristic rules and suggest improvements."""
    current_title = ticket.get("subject", ticket.get("raw_subject", ""))
    description = ticket.get("description", "")

    if not is_vague_title(current_title):
        # Title seems descriptive enough — keep it
        return {
            "suggested_title": "",
            "status": "Keep Current",
            "reason": "Title is already descriptive",
        }

    # Title is vague — try to build a better one
    suggested = build_suggested_title(current_title, description, comments)

    if suggested and suggested.lower() != current_title.lower():
        # Validate the suggestion
        validated = validate_suggestion(suggested, ticket["id"])
        if validated:
            reason = classify_vagueness(current_title)
            return {
                "suggested_title": validated,
                "status": "Suggestion",
                "reason": reason,
            }

    return {
        "suggested_title": "",
        "status": "Keep Current",
        "reason": "Could not generate a better title from ticket content",
    }


def classify_vagueness(title: str) -> str:
    """Return a human-readable reason why the title was flagged."""
    cleaned = title.strip().lower()
    cleaned = re.sub(r"^(re|fw|fwd)\s*:\s*", "", cleaned).strip()

    if cleaned in {"", "none", "n/a", "na", "no subject", "(no subject)", "...", ".", "-", "--", "___"}:
        return "Title is empty or placeholder"

    words = cleaned.split()
    if len(words) == 1:
        return f"Single-word title \"{title.strip()}\" — too vague for triage"

    if len(cleaned) < 10:
        return f"Title too short ({len(cleaned)} chars) to identify the issue"

    meaningful = [w for w in words if w not in STOP_WORDS]
    if len(meaningful) == 0:
        return "Title contains only generic/filler words"

    name_pattern = re.compile(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,2}$")
    if name_pattern.match(title.strip()):
        return "Title appears to be a person's name, not a description"

    if cleaned in VAGUE_TITLES:
        return f"Generic title \"{title.strip()}\" — doesn't describe the specific issue"

    return "Title lacks specificity for effective triage"


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
        ("PII Redaction",    "Enabled",                     "1F2D3D"),
        ("Mode",             "Rule-Based (no AI API)",      "1F2D3D"),
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
    for var in ("ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Zendesk Ticket Title Suggester (Rule-Based)")
    logger.info("=" * 60)
    logger.info("Mode: LOG ONLY (no tickets will be modified)")
    logger.info("Engine: Heuristic rules (no AI API required)")
    logger.info("Max tickets: %d", MAX_TICKETS)
    logger.info("Rate limits: Zendesk=%.1fs", ZENDESK_RATE_LIMIT_DELAY)
    logger.info("Retry config: max_retries=%d, base_delay=%.1fs", MAX_RETRIES, RETRY_BASE_DELAY)
    logger.info("PII redaction: ENABLED")
    logger.info("=" * 60)

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

        result = suggest_title(ticket, comments)

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
    print(f"Tickets scanned: {len(tickets)}")
    print(f"Suggestions made: {suggestion_count}")
    print(f"Titles kept: {keep_count}")
    print(f"Errors encountered: {errors}")
    print(f"Engine: Rule-based heuristics (no AI API)")
    print(f"PII redaction: enabled")
    print("=" * 80)

    for row in report_rows:
        if row["Status"] == "Suggestion":
            print(f"\nTicket #{row['Ticket #']}  {row['Ticket URL']}")
            print(f"  Current:   {row['Current Title']}")
            print(f"  Suggested: {row['Suggested Title']}")
            print(f"  Reason:    {row['Reason']}")

    print("\n" + "=" * 80)

    run_meta = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickets_scanned": len(tickets),
        "suggestions_made": suggestion_count,
        "titles_kept": keep_count,
        "errors": errors,
    }

    write_xlsx_report(report_rows, REPORT_PATH, run_meta)

    upload_to_gdrive(REPORT_PATH)

    if suggestion_count == 0:
        logger.info("All ticket titles look good \u2014 nothing to suggest!")

    if errors > 0 and errors == len(tickets):
        logger.error("All tickets failed to process. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
