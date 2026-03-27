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

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MAX_TICKETS` | `50` | Max tickets to process per run |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `OUTPUT_FILE` | `suggestions.json` | Path for the JSON report |
