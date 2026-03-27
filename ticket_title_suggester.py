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
from datetime import datetime
from functools import wraps

import requests
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZENDESK_SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")          # e.g. "mycompany"
ZENDESK_EMAIL = os.environ.get("ZENDESK_EMAIL")                  # e.g. "agent@company.com"
ZENDESK_API_TOKEN = os.environ.get("ZENDESK_API_TOKEN")          # Zendesk API token
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")           # Claude API key

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


def suggest_title(client: anthropic.Anthropic, ticket: dict, comments: list[dict]) -> str | None:
    """Use Claude to suggest a better title for the given ticket."""
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
            return None
        return validate_suggestion(suggestion, ticket["id"])
    except anthropic.APIError as e:
        logger.error("Claude API error for ticket #%s: %s", ticket["id"], e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    suggestions = []
    errors = 0

    for i, ticket in enumerate(tickets, 1):
        ticket_id = ticket["id"]
        current_title = ticket.get("subject", ticket.get("raw_subject", ""))
        logger.info("[%d/%d] Analyzing ticket #%s: %s", i, len(tickets), ticket_id, current_title)

        try:
            comments = fetch_ticket_comments(ticket_id)
        except requests.RequestException as e:
            logger.error("  Failed to fetch comments for ticket #%s: %s", ticket_id, e)
            errors += 1
            continue

        new_title = suggest_title(client, ticket, comments)

        if new_title:
            suggestions.append({
                "ticket_id": ticket_id,
                "ticket_url": f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}",
                "current_title": current_title,
                "suggested_title": new_title,
            })
            logger.info("  Suggested: %s", new_title)
        else:
            logger.info("  Title is fine, no change suggested.")

    # Print summary
    print("\n" + "=" * 80)
    print(f"TITLE SUGGESTION REPORT -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Tickets scanned: {len(tickets)}")
    print(f"Suggestions made: {len(suggestions)}")
    print(f"Errors encountered: {errors}")
    print(f"PII redaction: enabled")
    print("=" * 80)

    for s in suggestions:
        print(f"\nTicket #{s['ticket_id']}  {s['ticket_url']}")
        print(f"  Current:   {s['current_title']}")
        print(f"  Suggested: {s['suggested_title']}")

    print("\n" + "=" * 80)

    # Also write JSON output for potential downstream use
    output_path = os.environ.get("OUTPUT_FILE", "suggestions.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "run_date": datetime.now().isoformat(),
                "tickets_scanned": len(tickets),
                "suggestions_made": len(suggestions),
                "errors": errors,
                "pii_redaction": True,
                "mode": "log_only",
                "suggestions": suggestions,
            },
            f,
            indent=2,
        )
    logger.info("JSON report written to %s", output_path)

    if not suggestions:
        logger.info("All ticket titles look good -- nothing to suggest!")

    # Exit with error code if too many failures
    if errors > 0 and errors == len(tickets):
        logger.error("All tickets failed to process. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()

