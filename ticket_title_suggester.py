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
- Automation/notification email ticket detection
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
REPORT_PATH = os.environ.get("OUTPUT_FILE", f"/tmp/IT_Ops_Title_Suggestions_{NOW}.xlsx")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Redaction
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-.]+)?"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(\+?1?[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b\d{4}[-\\s]?\d{4}[-\\s]?\d{4}[-\\s]?\d{4}\b"), "[CC_REDACTED]"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
]


def redact_pii(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def title_contains_pii(title: str) -> bool:
    """Check if title contains PII patterns."""
    if not title:
        return False
    for pattern, _ in PII_PATTERNS:
        if pattern.search(title):
            return True
    return False


def strip_html(text: str) -> str:
    """Remove HTML tags and common HTML entities from text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;?", " and ", text, flags=re.IGNORECASE)
    text = re.sub(r"&lt;?", "<", text, flags=re.IGNORECASE)
    text = re.sub(r"&gt;?", ">", text, flags=re.IGNORECASE)
    text = re.sub(r"&quot;?", '"', text, flags=re.IGNORECASE)
    text = re.sub(r"&#?\w+;", " ", text)  # any remaining HTML entities
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_subject_line(title: str) -> str:
    """Strip Re:/Fwd: chains, bracketed prefixes, and org suffixes from subject."""
    if not title:
        return title
    cleaned = re.sub(
        r"^(?:(?:re|fw|fwd)\s*:\s*(?:\[.*?\]\s*)?)+", "", title, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(r"\[CloudSecurityAlliance\]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*[-–—]\s*Cloud Security Alliance\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Spam / marketing detection
# ---------------------------------------------------------------------------

SPAM_PATTERNS = [
    re.compile(r"(seo|guest\s*post|backlink|link\s*building|content\s*collaboration)", re.IGNORECASE),
    re.compile(r"(visitor\s*list|attendee\s*list)\s*(for|of|revealed|uncovered)", re.IGNORECASE),
    re.compile(r"(real\s*estate|dormitory|accommodation)\s*(available|opportunity|promotion|sites)", re.IGNORECASE),
    re.compile(r"(high\s*authority|da\s*\d+|domain\s*authority)", re.IGNORECASE),
    re.compile(r"plan\s*your\s*visit\s*to", re.IGNORECASE),
    re.compile(r"(amazing|incredible|exclusive)\s*(content|collaboration|opportunity)", re.IGNORECASE),
    re.compile(r"elevate\s+your\s+.{0,20}\s*seo", re.IGNORECASE),
    re.compile(r"(guest\s*post|sponsored\s*post|paid\s*post)\s*(opportunity|inquiry|proposal)", re.IGNORECASE),
]


def is_spam_ticket(title: str, description: str) -> bool:
    """Check if a ticket is spam/marketing based on title and body patterns."""
    text = f"{title} {(description or '')[:500]}"
    for pattern in SPAM_PATTERNS:
        if pattern.search(text):
            return True
    return False


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
    group = os.environ.get("ZENDESK_GROUP", "IT-Operations")
    query = f'type:ticket status<solved group:"{group}"'
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
    "inquiry", "general inquiry", "membership inquiry",
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
    # Common single-product vague titles
    "csa training", "training", "csa", "star",
    "receipt", "purchase", "order", "transaction",
    "membership", "renewal", "subscription",
}

# Patterns that indicate automated/notification email ticket titles
AUTOMATION_PATTERNS = [
    # Purchase/transaction notifications
    re.compile(r"^purchase\s+notification", re.IGNORECASE),
    re.compile(r"^order\s+(confirmation|receipt|notification)", re.IGNORECASE),
    re.compile(r"^payment\s+(confirmation|receipt|notification|received)", re.IGNORECASE),
    re.compile(r"^transaction\s+(confirmation|receipt|notification)", re.IGNORECASE),
    re.compile(r"^receipt\s+for\s+(your\s+)?purchase", re.IGNORECASE),
    re.compile(r"^your\s+(order|purchase|payment|receipt)", re.IGNORECASE),
    re.compile(r"^invoice\s+(#|number|for)", re.IGNORECASE),
    # Auto-responders and system notifications
    re.compile(r"^(auto[- ]?reply|automatic\s+reply|out\s+of\s+office)", re.IGNORECASE),
    re.compile(r"^(undeliverable|delivery\s+(failure|status)|mail\s+delivery\s+failed)", re.IGNORECASE),
    re.compile(r"^(do\s+not\s+reply|noreply|no-reply)", re.IGNORECASE),
    re.compile(r"^(automated?\s+(message|notification|alert|email|response))", re.IGNORECASE),
    re.compile(r"^(system\s+(notification|alert|message|update))", re.IGNORECASE),
    re.compile(r"^(alert|notification)\s*:", re.IGNORECASE),
    # Subscription/account notifications
    re.compile(r"^(welcome\s+to|thank\s+you\s+for\s+(your\s+)?(order|purchase|registration|signing\s+up))", re.IGNORECASE),
    re.compile(r"^(account\s+(created|activated|confirmation|verification))", re.IGNORECASE),
    re.compile(r"^(password\s+(reset|changed|updated)\s+(request|confirmation|notification))", re.IGNORECASE),
    re.compile(r"^(email\s+(verification|confirmation))", re.IGNORECASE),
    re.compile(r"^(subscription|renewal)\s+(confirmation|notification|reminder)", re.IGNORECASE),
    # Calendar/scheduling
    re.compile(r"^(invitation|invite|accepted|declined|tentative)\s*:", re.IGNORECASE),
    re.compile(r"^(meeting|calendar)\s+(invitation|update|cancellation)", re.IGNORECASE),
    re.compile(r"^(reminder|scheduled)\s*:", re.IGNORECASE),
    # Shipping/delivery
    re.compile(r"^(shipping|delivery|tracking)\s+(confirmation|notification|update)", re.IGNORECASE),
    re.compile(r"^your\s+(package|shipment|order)\s+(has\s+been\s+)?(shipped|delivered)", re.IGNORECASE),
    # Marketing/newsletters
    re.compile(r"^(newsletter|digest|weekly\s+update|monthly\s+update)", re.IGNORECASE),
    re.compile(r"^(special\s+offer|promotion|discount|sale|deal)", re.IGNORECASE),
    # Monitoring/CI/CD
    re.compile(r"^\[?(build|deploy|ci|cd|jenkins|github|gitlab|jira|confluence)\]?\s*", re.IGNORECASE),
    re.compile(r"^(uptime|downtime|monitoring|health\s+check)\s+(alert|notification)", re.IGNORECASE),
    # Contact form submissions
    re.compile(r"^a\s+\w+\s+contact\s+form", re.IGNORECASE),
    re.compile(r"contact\s+form\s+has\s+been\s+(filed|submitted|received)", re.IGNORECASE),
]

# Words/phrases in description that indicate automation-originated tickets
AUTOMATION_BODY_SIGNALS = [
    "this is an automated message",
    "this is an auto-generated",
    "this email was sent automatically",
    "do not reply to this email",
    "do not reply directly",
    "this is a system generated",
    "noreply@", "no-reply@", "donotreply@",
    "automated notification",
    "this message was generated",
    "you are receiving this because",
    "this is a confirmation of your",
    "unsubscribe from these",
    "manage your notification",
    "notification preferences",
    "purchase confirmation",
    "order confirmation",
    "payment receipt",
    "transaction receipt",
    "your receipt from",
    "form has been received from",
    "contact form submission",
]

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
    # PII redaction placeholders
    "email_redacted", "phone_redacted", "token_redacted",
    "ip_redacted", "ssn_redacted", "cc_redacted",
    "redacted",
    # HTML artifacts
    "nbsp", "amp", "quot", "lt", "gt", "http", "https", "www", "com",
    "org", "net", "html", "div", "span", "href", "img", "src", "alt",
    "class", "style", "width", "height", "border", "padding", "margin",
    "font", "color", "size", "table", "tbody", "thead", "col",
    # Email boilerplate words
    "sent", "received", "forwarded", "replied", "subject", "date",
    "wrote", "message", "email", "mail",
    # Generic filler
    "really", "actually", "basically", "something", "anything", "everything",
    "nothing", "someone", "anyone", "everyone", "thing", "things",
    "way", "ways", "lot", "lots", "bit", "much", "many",
    "going", "come", "came", "make", "made", "take", "took", "give", "gave",
    "see", "saw", "say", "said", "tell", "told", "ask", "asked",
    "work", "working", "look", "looking", "put", "keep", "let",
    "begin", "began", "seem", "show", "showed",
}

# Words that look like personal names (short, capitalized) — filter from suggestions
# We detect these dynamically rather than maintaining a list

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
    "TAISE", "Trusted AI Safety Expert",
]

# Compile product patterns for matching (case-insensitive)
PRODUCT_PATTERNS = [(re.compile(re.escape(p), re.IGNORECASE), p) for p in KNOWN_PRODUCTS]


def is_likely_name(word: str) -> bool:
    """Check if a word looks like a personal name (not a product/acronym)."""
    if not word or len(word) < 2:
        return False
    # All-caps words are acronyms, not names
    if word.isupper() and len(word) <= 6:
        return False
    # Check if it's a known product term
    word_lower = word.lower()
    for _, product in PRODUCT_PATTERNS:
        if word_lower in product.lower().split():
            return False
    # Looks like a name: Capitalized, 3-12 chars, all alpha
    if re.match(r"^[A-Z][a-z]{2,11}$", word):
        return True
    return False


def is_automation_ticket(title: str, description: str) -> bool:
    """Check if a ticket was created by an automation or notification email."""
    # Check title against automation patterns
    for pattern in AUTOMATION_PATTERNS:
        if pattern.search(title.strip()):
            return True

    # Check description for automation signals
    desc_lower = (description or "").lower()[:2000]
    signal_count = sum(1 for signal in AUTOMATION_BODY_SIGNALS if signal in desc_lower)
    # If 2+ signals found in the body, it's very likely automated
    if signal_count >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Category detection for suggested titles
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = [
    # Order matters — first match wins, so put more specific patterns first
    # Product/program-specific categories go first to avoid generic matches
    ("STAR/Registry", [
        re.compile(r"\b(star\s+registry|star\s+level|star\s+attestation|starwatch|caiq|ccm|grc\s+stack|trusted\s+cloud)\b", re.IGNORECASE),
        re.compile(r"\bstar\s+(report|review|submission|listing|entry|profile)\b", re.IGNORECASE),
        re.compile(r"(registry|listing)\s+.{0,20}(review|update|entry|profile|submission)", re.IGNORECASE),
    ]),
    ("Certification", [
        re.compile(r"\b(ccsk|ccak|cczt|certificate of cloud|certification|exam\b|badge|credential|voucher|token)\b", re.IGNORECASE),
        re.compile(r"(training|course|learning|study)\s+(material|access|platform|portal)", re.IGNORECASE),
    ]),
    # Billing before Tooling — renewal/invoice/payment takes priority over tool name
    ("Billing", [
        re.compile(r"(invoice|receipt|refund|credit|charge|payment|billing|renewal|quote|pricing|cost)\b", re.IGNORECASE),
        re.compile(r"(purchase|bought|paid|order)\s+.{0,30}(but|however|issue|problem|wrong|error)", re.IGNORECASE),
    ]),
    # Tooling before Access Request — tool-specific mentions take priority
    ("Tooling", [
        re.compile(r"\b(github|gitlab|jira|confluence|slack|zoom|teams|zendesk|airtable|zapier|salesforce|pardot|hubspot|mailgun|surveymonkey)\b", re.IGNORECASE),
        re.compile(r"\b(chatgpt|claude|copilot|ai\s+license|ai\s+vendor)\b", re.IGNORECASE),
        re.compile(r"(consolidat|migrat|decommission|integrat)\w*\s+.{0,30}(tool|platform|service|account|license|subscription)", re.IGNORECASE),
    ]),
    ("Access Request", [
        re.compile(r"(add|remove|grant|revoke|give|need)\s+.{0,30}(access|permission|role|admin|editor|viewer|member)", re.IGNORECASE),
        re.compile(r"(access|permission|role)\s+(to|for|on|request)", re.IGNORECASE),
        re.compile(r"(add|invite|remove)\s+.{0,20}(user|member|team|group)", re.IGNORECASE),
        re.compile(r"(added? to|removed? from|join)\s+.{0,20}(team|group|channel|org|alias)", re.IGNORECASE),
        re.compile(r"\b(add\w*|be\s+added)\b.{0,40}\balias\b", re.IGNORECASE),
        re.compile(r"(sso|oauth|login|sign.?in|password|mfa|2fa)\b", re.IGNORECASE),
    ]),
    ("Infrastructure", [
        re.compile(r"(server|dns|domain|ssl|tls|firewall|vpn|router|firmware|cloudflare|digital.?ocean|aws|azure|gcp)", re.IGNORECASE),
        re.compile(r"(dmarc|dkim|spf|mx\s+record|email\s+(?:config|setting|routing|deliverability))", re.IGNORECASE),
        re.compile(r"(deploy|hosting|uptime|outage|downtime|monitoring|backup)", re.IGNORECASE),
    ]),
    ("Security", [
        re.compile(r"(security|vulnerability|incident|breach|phishing|malware|ransomware|threat|audit)\b", re.IGNORECASE),
        re.compile(r"(security\.txt|pen\s*test|penetration|compliance|pii|data\s+(?:leak|exposure|privacy|protection|flow|governance))", re.IGNORECASE),
        re.compile(r"(stale\s+token|rotate\s+key|secret|credential\s+rotation)", re.IGNORECASE),
    ]),
    ("Configuration", [
        re.compile(r"(config|setting|setup|enable|disable|toggle|update|modify|change)\s+.{0,30}(setting|config|option|feature|flag|policy|rule)", re.IGNORECASE),
        re.compile(r"(redirect|rewrite|iframe|embed|seo|sitemap|robots\.txt|index)", re.IGNORECASE),
        re.compile(r"(workflow|automation|trigger|schedule|cron|sla\s+polic)", re.IGNORECASE),
    ]),
    ("Documentation", [
        re.compile(r"(document|documentation|wiki|guide|readme|runbook|playbook|knowledge\s+base|kb)\b", re.IGNORECASE),
        re.compile(r"(publish|update|review|create)\s+.{0,20}(doc|page|article|paper|policy|procedure|process)", re.IGNORECASE),
        re.compile(r"(cms|content\s+management|working\s+group\s+page)", re.IGNORECASE),
    ]),
    ("Account Issue", [
        re.compile(r"(account|profile|user)\s+.{0,20}(expired?|locked|blocked|suspended|disabled|inactive|missing|wrong|invalid)", re.IGNORECASE),
        re.compile(r"(expired?|expir(?:ing|ation))\s+.{0,20}(account|license|subscription|membership|certificate)", re.IGNORECASE),
        re.compile(r"(cannot|can'?t|unable)\s+.{0,20}(log\s*in|sign\s*in|access|authenticate)", re.IGNORECASE),
    ]),
    ("Data/Reporting", [
        re.compile(r"(dashboard|analytics|metric|data\s+(?:export|import|extract|query|migration))", re.IGNORECASE),
        re.compile(r"(run\s+a\s+report|generate\s+report|pull\s+report)", re.IGNORECASE),
        re.compile(r"(csv|excel|spreadsheet|tableau|google\s+sheet)", re.IGNORECASE),
    ]),
]


def detect_category(title: str, description: str) -> str:
    """Detect the operational category of a ticket based on title and description."""
    text = f"{title} {(description or '')[:1500]}"
    for category, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if pat.search(text):
                return category
    return "General Inquiry"


def is_url_title(title: str) -> bool:
    """Check if a title is primarily a URL or URL fragment."""
    cleaned = title.strip()
    # Title starts with a URL (i.e. the URL *is* the title)
    if re.match(r"^https?://", cleaned, re.IGNORECASE):
        return True
    if re.match(r"^www\.", cleaned, re.IGNORECASE):
        return True
    # Title contains a URL — only flag if the non-URL text is too short to be meaningful
    url_match = re.search(r"https?://\S+", cleaned)
    if url_match:
        non_url_text = cleaned[:url_match.start()] + cleaned[url_match.end():]
        non_url_words = [w for w in non_url_text.split() if len(w) > 2]
        if len(non_url_words) < 3:
            return True
    return False


def is_vague_title(title: str) -> bool:
    """Check if a title is too vague/generic to be useful."""
    cleaned = title.strip().lower()
    cleaned = re.sub(r"^(re|fw|fwd)\s*:\s*", "", cleaned).strip()

    # URL-based titles are always vague
    if is_url_title(title):
        return True

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

    # Strip HTML and URLs before processing
    text = strip_html(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = redact_pii(text)

    # First, check for known product mentions
    found_products = []
    for pattern, product_name in PRODUCT_PATTERNS:
        if pattern.search(text):
            found_products.append(product_name)

    # Tokenize and count meaningful words
    words = re.findall(r"[a-zA-Z]{3,}", text)
    # Filter: stop words, redaction artifacts, likely names, HTML remnants
    meaningful = []
    for w in words:
        w_lower = w.lower()
        if w_lower in STOP_WORDS:
            continue
        if len(w_lower) <= 2:
            continue
        if "redacted" in w_lower:
            continue
        if is_likely_name(w):
            continue
        meaningful.append(w_lower)

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
    # Strip URLs from title so they don't pollute keyword extraction
    title_for_keywords = re.sub(r"https?://\S+", " ", title)
    title_for_keywords = re.sub(r"www\.\S+", " ", title_for_keywords).strip()

    # Combine text sources
    all_text = description or ""
    for c in comments[:3]:
        body = c.get("plain_body") or c.get("body", "")
        if body:
            all_text += " " + body

    all_text = strip_html(all_text[:3000])

    # Extract keywords
    keywords = extract_keywords(all_text)

    if not keywords:
        return ""

    # Check for common action patterns in the description
    desc_clean = strip_html((description or "")[:1500]).lower()
    action = ""

    action_patterns = [
        # Token/voucher issues (very common for CSA)
        (r"(?:have not|haven'?t|did not|didn'?t|never)\s+(?:received?|got|get)\s+(?:(?:my|the|an?)\s+)?(token|voucher|exam|certificate|badge|receipt|confirmation|access|login)",
         "Not received: {match}"),
        # Where/how to find things
        (r"(?:where|how)\s+(?:can|do)\s+i\s+(?:find|get|download|access|view|see|locate|retrieve)\s+(?:(?:my|the|an?)\s+)?(token|voucher|exam|certificate|receipt|course|training|badge)",
         "Question: How to find {match}"),
        # Purchase followed by issue
        (r"i\s+(?:purchased|bought|paid\s+for|ordered)\s+(?:(?:the|a|an)\s+)?(.+?)(?:\s+(?:but|and|however|yet)\s+)",
         "Purchased — {match}"),
        # Can't find specific things
        (r"(?:can'?t|cannot|unable\s+to|couldn'?t)\s+(?:find|locate|see|view)\s+(?:(?:my|the|an?)\s+)?(token|voucher|exam|certificate|receipt|course|account|login)",
         "Cannot find {match}"),
        # Credential/account status issues
        (r"(?:my|the)\s+(certificate|exam|token|voucher|badge|account|login|password|access|membership)\s+(?:is|has|was)\s+(expired?|missing|invalid|wrong|incorrect|locked|blocked|suspended)",
         "{match} issue"),
        # Receipt/refund requests
        (r"(?:need|want|would\s+like|requesting?)\s+(?:a\s+)?(receipt|invoice|refund|credit|reimbursement|proof\s+of\s+purchase)",
         "Request: {match}"),
        # Original patterns
        (r"(can't|cannot|unable to|couldn't|could not)\s+(access|log\s*in|sign\s*in|connect|view|open|download|upload|use|find|see|load|reach)",
         "Cannot {match}"),
        (r"(need|want|would like|requesting|request)\s+(?:to\s+)?(add|remove|change|update|reset|create|delete|modify|enable|disable|upgrade|renew|transfer|consolidate)",
         "Request to {match}"),
        (r"(how|where)\s+(?:do|can|to)\s+(\w+(?:\s+\w+){0,3})",
         "Question: How to {match}"),
        (r"(error|failed|failure|crash|broken|bug|not working|doesn'?t work|isn'?t working)",
         "Error"),
        (r"(expired?|expir(?:ing|ation))",
         "Expiration issue"),
        (r"(invoice|billing|charge|payment|refund|credit)",
         "Billing inquiry"),
        (r"(certificate|certification|exam|badge|credential)",
         "Certification inquiry"),
        (r"(registry|listing|profile|entry)",
         "Registry/listing inquiry"),
    ]

    _articles = {"a", "an", "the", "my", "your", "our", "their"}

    for pattern, template in action_patterns:
        match = re.search(pattern, desc_clean)
        if match:
            if "{match}" in template:
                groups = [g for g in match.groups() if g and g.strip() not in _articles]
                relevant = groups[-1] if groups else ""
                relevant = relevant.strip()
                clean_words = [w for w in relevant.split()
                               if w.lower() not in STOP_WORDS
                               and "redacted" not in w.lower()
                               and not is_likely_name(w.capitalize())]
                if clean_words:
                    action = template.replace("{match}", " ".join(clean_words))
                else:
                    action = template.split("{")[0].strip()
            else:
                action = template
            break

    # Build the title
    products = [k for k in keywords if any(p == k for _, p in PRODUCT_PATTERNS)]
    other_keywords = [k for k in keywords if k not in products
                      and "redacted" not in k.lower()
                      and not is_likely_name(k.capitalize())][:4]

    if products and action:
        suggested = f"{products[0]}: {action}"
    elif products:
        context = " ".join(other_keywords[:3]).capitalize() if other_keywords else "inquiry"
        suggested = f"{products[0]} — {context}"
    elif action:
        context = " ".join(other_keywords[:3]) if other_keywords else ""
        if context:
            suggested = f"{action} — {context}"
        else:
            suggested = action
    else:
        # Fallback: just use top keywords
        suggested = " ".join(other_keywords[:5]).capitalize()

    # Final cleanup
    suggested = suggested.strip(" —-:")
    suggested = re.sub(r"\s+", " ", suggested)
    # Remove any remaining redaction markers
    suggested = re.sub(r"\[?\w*_?REDACTED\]?", "", suggested, flags=re.IGNORECASE).strip()
    suggested = re.sub(r"\s+", " ", suggested).strip(" —-:")

    # Capitalize first letter
    if suggested:
        suggested = suggested[0].upper() + suggested[1:]

    # Check for incomplete suggestions (Issues 3 & 4)
    # Reject if too short, too generic, or ends with incomplete phrase
    if len(suggested) < 15:
        return ""
    
    incomplete_endings = ("to", "how to", "how", "what", "where", "for", "about", "with", "cannot")
    if any(suggested.lower().endswith(ending) for ending in incomplete_endings):
        return ""
    
    # Check for broken/generic patterns like "Cannot" alone or "Question: How to" without object
    if suggested.lower() in ("cannot", "error", "question", "error: ") \
            or (suggested.lower().startswith(("question: how to", "cannot ")) and len(suggested) < 25):
        return ""

    # Check for multiple emails in body indicating list/table (Issue 5)
    if description:
        email_count = len(re.findall(r"\w+@\w+", description))
        title_word_count = len(title.split())
        # If many emails but vague/short title, skip suggestion
        if email_count >= 3 and title_word_count <= 3:
            return ""

    # Prefix with category
    category = detect_category(title, description)
    suggested = f"[{category}] {suggested}"

    # Enforce max length
    if len(suggested) > 100:
        suggested = suggested[:97] + "..."

    return suggested if len(suggested) >= 10 else ""


def suggest_title(ticket: dict, comments: list[dict]) -> dict:
    """Analyze a ticket title using heuristic rules and suggest improvements."""
    current_title = ticket.get("subject", ticket.get("raw_subject", ""))
    description = ticket.get("description", "")

    # Clean subject for analysis (preserve original for report)
    cleaned_title = clean_subject_line(current_title)

    # Check for PII in title FIRST
    if title_contains_pii(current_title):
        return {
            "suggested_title": redact_pii(current_title),
            "status": "PII in Title",
            "reason": "Current title contains personal information (email/phone) that should be removed",
        }

    # Check for spam/marketing tickets
    if is_spam_ticket(cleaned_title, description):
        return {
            "suggested_title": "",
            "status": "Likely Spam",
            "reason": "Title matches spam/marketing patterns — likely not a real support ticket",
        }

    # Check for automation/notification tickets
    if is_automation_ticket(cleaned_title, description):
        # Try to extract product name from multiple automation title formats
        _automation_extractors = [
            (re.compile(r"purchase notification for (.+)", re.IGNORECASE), "Purchase Notification"),
            (re.compile(r"registration notification for (.+)", re.IGNORECASE), "Registration"),
            (re.compile(r"(?:fwd?:\s*)?your purchase of (.+)", re.IGNORECASE), "Purchase Notification"),
            (re.compile(r"receipt for (?:your )?(?:purchase of )?(.+)", re.IGNORECASE), "Receipt"),
            (re.compile(r"order confirmation[:\s]+(.+)", re.IGNORECASE), "Order Confirmation"),
            (re.compile(r"payment of \$[\d.,]+ from .+ for (.+)", re.IGNORECASE), "Payment Received"),
        ]
        for pat, notif_type in _automation_extractors:
            m = pat.search(current_title)
            if m:
                product_name = m.group(1).strip()
                product_name = re.sub(r"\s*[-–—]\s*Cloud Security Alliance\s*$", "", product_name).rstrip(".")
                if len(product_name) > 80:
                    product_name = product_name[:77] + "..."
                category = detect_category(current_title, description)
                suggested = f"[{category}] {product_name} — {notif_type}"
                return {
                    "suggested_title": suggested,
                    "status": "Suggestion",
                    "reason": "Title is from an automated/notification email — needs a human-readable subject",
                }

        # Try to build a better title from the content
        suggested = build_suggested_title(cleaned_title, description, comments)
        reason = classify_vagueness_or_automation(cleaned_title, description)
        if suggested and suggested.lower() != cleaned_title.lower():
            validated = validate_suggestion(suggested, ticket["id"])
            if validated:
                return {
                    "suggested_title": validated,
                    "status": "Suggestion",
                    "reason": reason,
                }
        return {
            "suggested_title": "",
            "status": "Suggestion",
            "reason": reason,
        }

    if not is_vague_title(cleaned_title):
        # Title seems descriptive enough — keep it
        return {
            "suggested_title": "",
            "status": "Keep Current",
            "reason": "Title is already descriptive",
        }

    # Title is vague — try to build a better one
    suggested = build_suggested_title(cleaned_title, description, comments)

    if suggested and suggested.lower() != cleaned_title.lower():
        # Validate the suggestion
        validated = validate_suggestion(suggested, ticket["id"])
        if validated:
            reason = classify_vagueness_or_automation(cleaned_title, description)
            return {
                "suggested_title": validated,
                "status": "Suggestion",
                "reason": reason,
            }

    if is_url_title(current_title):
        reason = "Title is a URL — needs a human-readable subject describing the actual request"
    else:
        reason = "Title is vague but no automatic suggestion could be generated — manual review recommended"
    return {
        "suggested_title": "",
        "status": "Needs Manual Review",
        "reason": reason,
    }


def classify_vagueness_or_automation(title: str, description: str = "") -> str:
    """Return a human-readable reason why the title was flagged."""
    # Check automation first
    if is_automation_ticket(title, description):
        for pattern in AUTOMATION_PATTERNS:
            if pattern.search(title.strip()):
                return "Title is from an automated/notification email — needs a human-readable subject"
        return "Ticket body indicates this was auto-generated — title may not describe the actual issue"

    if is_url_title(title):
        return "Title is a URL — needs a human-readable subject describing the actual request"

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

    # Also reject if redaction markers leaked through
    if re.search(r"redacted", suggestion, re.IGNORECASE):
        logger.warning("Ticket #%s: Suggestion contains redaction markers, skipping.", ticket_id)
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
PII_FILL      = "FFF3E0";  ALT_PII      = "FFF8E8"
SPAM_FILL     = "EEEEEE";  ALT_SPAM     = "F5F5F5"
LINK_COLOR    = "1155CC"
SUGGEST_BADGE = "27AE60"
KEEP_BADGE    = "7F8C8D"
ERROR_BADGE   = "C0392B"
SKIP_BADGE    = "F57F17"
PII_BADGE     = "E65100"
SPAM_BADGE    = "9E9E9E"

HEADERS = [
    "Ticket #", "Status", "Current Title", "Suggested Title",
    "Recommendation", "Reason", "Ticket Status", "Requester",
    "Created", "Last Updated",
]
WIDTHS = [10, 14, 44, 44, 18, 36, 12, 20, 13, 13]


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
    elif status == "PII in Title":
        return (PII_FILL, ALT_PII, PII_BADGE)
    elif status == "Needs Manual Review":
        return (SKIP_FILL, ALT_SKIP, SKIP_BADGE)
    elif status == "Likely Spam":
        return (SPAM_FILL, ALT_SPAM, SPAM_BADGE)
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

        # Col 8: Requester
        _cell(ws, cur_row, 8, r.get("Requester", ""), bg=bg, align="center")

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
    title_cell = es.cell(row=1, column=1, value=f"Title Suggestion Report — {today_str}")
    title_cell.font = Font(name="Arial", bold=True, size=16, color="1F2D3D")
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    es.row_dimensions[1].height = 32

    stats = [
        ("Run Date",         run_meta["run_date"],          "1F2D3D"),
        ("Tickets Scanned",  run_meta["tickets_scanned"],   "1F2D3D"),
        ("Suggestions Made", run_meta["suggestions_made"],  SUGGEST_BADGE),
        ("Titles Kept",      run_meta["titles_kept"],       KEEP_BADGE),
        ("Manual Reviews",   run_meta.get("manual_reviews", 0), SKIP_BADGE),
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

    # Breakdown stats
    row_num += 1
    es.merge_cells(start_row=row_num, start_column=1, end_row=row_num, end_column=2)
    sec = es.cell(row=row_num, column=1, value="Issue Breakdown")
    sec.font = Font(name="Arial", bold=True, size=12, color="1F2D3D")
    sec.alignment = Alignment(horizontal="left", vertical="center")
    row_num += 1

    # Count automation tickets
    automation_count = sum(1 for r in rows if "automated" in r.get("Reason", "").lower() or "auto-generated" in r.get("Reason", "").lower())
    
    # Count vague titles (Suggestion status but not automation, not PII)
    vague_count = sum(1 for r in rows if r.get("Status") == "Suggestion" 
                      and "automated" not in r.get("Reason", "").lower() 
                      and "auto-generated" not in r.get("Reason", "").lower()
                      and "personal information" not in r.get("Reason", "").lower())
    
    # Count PII in Title
    pii_count = sum(1 for r in rows if r.get("Status") == "PII in Title")

    breakdown_items = [
        ("Automation Tickets", automation_count),
        ("Vague Titles", vague_count),
        ("PII in Title", pii_count),
        ("Needs Manual Review", run_meta.get("manual_reviews", 0)),
        ("Spam/Marketing", run_meta.get("spam_flagged", 0)),
    ]
    for label, val in breakdown_items:
        es.cell(row=row_num, column=1, value=label).font = Font(
            name="Arial", bold=True, size=10, color="333333")
        v = es.cell(row=row_num, column=2, value=int(val))
        v.font = Font(name="Arial", bold=True, size=11, color="666666")
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

        for r in suggestions_only:
            tid_cell = es.cell(row=row_num, column=1, value=r.get("Ticket #", ""))
            tid_url = r.get("Ticket URL", "")
            tid_cell.font = Font(name="Arial", color=LINK_COLOR, underline="single", size=11)
            if tid_url:
                tid_cell.hyperlink = tid_url
            tid_cell.border = _border()

            es.cell(row=row_num, column=2, value=r.get("Current Title", "")).border = _border()

            sug_cell = es.cell(row=row_num, column=3, value=r.get("Suggested Title", ""))
            sug_cell.font = Font(name="Arial", bold=True, color=SUGGEST_BADGE, size=11)
            sug_cell.border = _border()

            es.cell(row=row_num, column=4, value=r.get("Reason", "")).border = _border()

            es.row_dimensions[row_num].height = 28
            row_num += 1

    wb.save(output_path)
    logger.info("Spreadsheet saved → %s (%d data rows)", output_path, len(rows))


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
    manual_review_count = 0
    spam_count = 0
    errors = 0

    for i, ticket in enumerate(tickets, 1):
        ticket_id = ticket["id"]
        current_title = ticket.get("subject", ticket.get("raw_subject", ""))
        ticket_status = ticket.get("status", "")
        created_at = ticket.get("created_at", "")
        updated_at = ticket.get("updated_at", "")
        ticket_url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{ticket_id}"

        # Extract requester name
        requester_name = ticket.get("via", {}).get("source", {}).get("from", {}).get("name", "")
        if not requester_name:
            requester_name = str(ticket.get("requester_id", ""))

        logger.info("[%d/%d] Analyzing ticket #%s: %s", i, len(tickets), ticket_id, current_title)

        try:
            comments = fetch_ticket_comments(ticket_id)
        except requests.RequestException as e:
            logger.error("  → Failed to fetch comments for ticket #%s: %s", ticket_id, e)
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
                "Requester": requester_name,
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
            logger.info("  → Suggested: %s", suggested_title if suggested_title else "(flag only — no auto-suggestion)")
        elif status == "Needs Manual Review":
            manual_review_count += 1
            recommendation = "Review Manually"
            logger.info("  → Needs manual review: vague title with no auto-suggestion")
        elif status == "Likely Spam":
            spam_count += 1
            recommendation = "Likely Spam"
            logger.info("  → Likely spam/marketing ticket")
        elif status == "Error":
            errors += 1
            recommendation = "Review Manually"
            logger.info("  → Error analyzing title.")
        else:
            keep_count += 1
            recommendation = "No Action Needed"
            logger.info("  → Title is fine, no change suggested.")

        report_rows.append({
            "Ticket #": ticket_id,
            "Status": status,
            "Current Title": current_title,
            "Suggested Title": suggested_title,
            "Recommendation": recommendation,
            "Reason": reason,
            "Ticket URL": ticket_url,
            "Ticket Status": ticket_status.capitalize(),
            "Requester": requester_name,
            "Created": format_date(created_at),
            "Last Updated": format_date(updated_at),
        })

    # Print summary
    print("\n" + "=" * 80)
    print(f"TITLE SUGGESTION REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Tickets scanned: {len(tickets)}")
    print(f"Suggestions made: {suggestion_count}")
    print(f"Titles kept: {keep_count}")
    print(f"Manual reviews needed: {manual_review_count}")
    print(f"Spam/marketing flagged: {spam_count}")
    print(f"Errors encountered: {errors}")
    print(f"Engine: Rule-based heuristics (no AI API)")
    print(f"PII redaction: enabled")
    print("=" * 80)

    for row in report_rows:
        if row["Status"] == "Suggestion":
            print(f"\nTicket #{row['Ticket #']}  {row['Ticket URL']}")
            print(f"  Current:   {row['Current Title']}")
            print(f"  Suggested: {row['Suggested Title'] or '(review needed — no auto-suggestion)'}")
            print(f"  Reason:    {row['Reason']}")

    print("\n" + "=" * 80)

    run_meta = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickets_scanned": len(tickets),
        "suggestions_made": suggestion_count,
        "titles_kept": keep_count,
        "manual_reviews": manual_review_count,
        "spam_flagged": spam_count,
        "errors": errors,
    }

    write_xlsx_report(report_rows, REPORT_PATH, run_meta)

    upload_to_gdrive(REPORT_PATH)

    if suggestion_count == 0 and manual_review_count == 0:
        logger.info("All ticket titles look good — nothing to suggest!")

    if errors > 0 and errors == len(tickets):
        logger.error("All tickets failed to process. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
