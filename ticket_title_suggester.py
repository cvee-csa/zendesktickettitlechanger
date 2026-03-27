"""
Zendesk Ticket Title Suggester

Queries Zendesk for open tickets, analyzes their content using the Claude API,
and suggests more meaningful titles based on ticket context.
"""

import os
import sys
import json
import logging
from datetime import datetime

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zendesk helpers
# ---------------------------------------------------------------------------


def zendesk_auth():
    """Return the (email/token, api_token) tuple for Zendesk basic auth."""
    return (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)


def fetch_open_tickets() -> list[dict]:
    """Fetch open (and new) tickets from Zendesk using the Search API."""
    tickets = []
    query = "type:ticket status<solved"
    url = f"{ZENDESK_BASE_URL}/search.json"
    params = {"query": query, "sort_by": "created_at", "sort_order": "desc", "per_page": 100}

    while url and len(tickets) < MAX_TICKETS:
        logger.info("Fetching tickets from: %s", url)
        resp = requests.get(url, auth=zendesk_auth(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        tickets.extend(data.get("results", []))
        url = data.get("next_page")
        params = None  # next_page URL already contains query params

    return tickets[:MAX_TICKETS]


def fetch_ticket_comments(ticket_id: int) -> list[dict]:
    """Fetch the first few comments on a ticket to provide context."""
    url = f"{ZENDESK_BASE_URL}/tickets/{ticket_id}/comments.json"
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
- If the current title is already clear and descriptive, respond with "KEEP" and nothing else.
- Respond with ONLY the suggested title (or "KEEP"). No explanation, no quotes.
"""


def suggest_title(client: anthropic.Anthropic, ticket: dict, comments: list[dict]) -> str | None:
    """Use Claude to suggest a better title for the given ticket."""
    current_title = ticket.get("subject", ticket.get("raw_subject", ""))
    description = ticket.get("description", "")

    # Build context from first few comments
    comment_texts = []
    for c in comments[:3]:
        body = c.get("plain_body") or c.get("body", "")
        if body:
            comment_texts.append(body[:1000])  # trim long comments

    user_message = f"""Current title: {current_title}

Ticket description:
{description[:2000]}

Additional comments:
{chr(10).join(comment_texts) if comment_texts else "(none)"}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        suggestion = response.content[0].text.strip()
        if suggestion.upper() == "KEEP":
            return None
        return suggestion
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

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    logger.info("Fetching open tickets from Zendesk (%s)...", ZENDESK_SUBDOMAIN)
    tickets = fetch_open_tickets()
    logger.info("Found %d open tickets to analyze.", len(tickets))

    suggestions = []

    for ticket in tickets:
        ticket_id = ticket["id"]
        current_title = ticket.get("subject", ticket.get("raw_subject", ""))
        logger.info("Analyzing ticket #%s: %s", ticket_id, current_title)

        comments = fetch_ticket_comments(ticket_id)
        new_title = suggest_title(client, ticket, comments)

        if new_title:
            suggestions.append({
                "ticket_id": ticket_id,
                "ticket_url": f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}",
                "current_title": current_title,
                "suggested_title": new_title,
            })
            logger.info(
                "  → Suggested: %s",
                new_title,
            )
        else:
            logger.info("  → Title is fine, no change suggested.")

    # Print summary
    print("\n" + "=" * 80)
    print(f"TITLE SUGGESTION REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Tickets scanned: {len(tickets)}")
    print(f"Suggestions made: {len(suggestions)}")
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
                "suggestions": suggestions,
            },
            f,
            indent=2,
        )
    logger.info("JSON report written to %s", output_path)

    if not suggestions:
        logger.info("All ticket titles look good — nothing to suggest!")


if __name__ == "__main__":
    main()
