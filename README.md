# Zendesk Ticket Title Changer

Scans open Zendesk tickets and uses the Claude API to suggest clearer, more descriptive titles.

## Setup

### 1. Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|---|---|
| `ZENDESK_SUBDOMAIN` | Your Zendesk subdomain (e.g. `mycompany` from `mycompany.zendesk.com`) |
| `ZENDESK_EMAIL` | Email of the Zendesk user/agent with API access |
| `ZENDESK_API_TOKEN` | Zendesk API token (Admin → Channels → API) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

### 2. Run the Workflow

1. Go to the **Actions** tab in your repo
2. Select **"Suggest Zendesk Ticket Titles"**
3. Click **"Run workflow"**
4. Optionally set the max number of tickets to scan (default: 50)

### 3. View Results

- The suggestions are printed in the workflow logs
- A `suggestions.json` artifact is uploaded to each workflow run with structured output

## Local Usage

```bash
export ZENDESK_SUBDOMAIN="yourcompany"
export ZENDESK_EMAIL="you@company.com"
export ZENDESK_API_TOKEN="your-token"
export ANTHROPIC_API_KEY="your-key"

pip install -r requirements.txt
python ticket_title_suggester.py
```

## Changelog

### 2026-10-30
- **Security**: `Vulnerabilities` (plural) now triggers Security category; token/credential expiry pattern added (e.g. "Authorization Token expiring in 60 days")
- **Automation**: "Active Campaign" (two words) now matches alongside `activecampaign`; recurring tasks/scripts pattern added; `[Zoom Events / Active Campaign Integration]` correctly routed
- **Access Request**: `Removing` (not just `remove`) now matches; `seat` added to access-type words (e.g. "Grant full Zendesk seat"); tool-name + "access" pattern added for Sharepoint, GitHub, Drive, etc.
- **Documentation**: "Offboarding, Add X" pattern added so `[Documentation] Offboarding, Add Check for...` tickets are not wrongly recategorized; `handbook` added as keyword
- **Tooling**: Zoho, Sharepoint, Firecrawl added to context patterns
- **CATEGORY_ALIASES**: `[Permissions]` → Access Request; `[Action Required]` → Automated Notification; `[Offboarding]` / `[Onboarding]` → Onboarding/Offboarding

### 2026-10-29
- **Pattern ordering refactor**: Added early high-confidence mini-blocks (before Access Request) for Security (stale token/credential rotation), Account Issue (account locked), and Networking (Unifi/Ubiquiti) to prevent Access Request's broad oauth/login/alias patterns from stealing those tickets
- **Documentation**: Two-stage block — specific SOP/KB patterns placed before Onboarding so `[Documentation] Renew Zendesk Article` stays Documentation; broader patterns placed before Data/Reporting and Configuration
- **Security**: Broadened stale token pattern to catch "Stale OAuth Token" (multi-word); added CVE, bot traffic, 2SV patterns
- **Onboarding/Offboarding**: Added "remove former/ex employees from staff page" with plural fix; added `delete Microsoft365 accounts` pattern
- **Automation**: Added ActiveCampaign, MCP connector, BrightTalk+AC integration, investigation+integration patterns
- **Tooling**: Added Adobe/Acrobat to context patterns
- **Infrastructure**: Added CertCentral/DigiCert, SSL/cert renewal patterns
- **Configuration**: Added email address issue patterns; email group/alias patterns
- **Data/Reporting**: Added SLA report/graph pattern
- **CATEGORY_ALIASES**: Added `[Reporting]` → Data/Reporting and other legacy prefixes so they trigger recategorization instead of silent skip

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MAX_TICKETS` | `50` | Max tickets to process per run |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `OUTPUT_FILE` | `suggestions.json` | Path for the JSON report |
