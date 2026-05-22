"""
All agent definitions for the Competitive Intelligence platform.

Four specialist agents run sequentially:
  1. market_scanner      — finds recent news and moves
  2. sentiment_analyzer  — scores brand sentiment
  3. pricing_intelligence — extracts pricing data
  4. report_generator    — synthesizes an executive brief

The host_agent (SequentialAgent) wires them in order.
"""
import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import google_search

load_dotenv()

os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', os.getenv('GOOGLE_CLOUD_PROJECT', ''))
os.environ.setdefault('GOOGLE_CLOUD_LOCATION', os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'))

MODEL = 'gemini-2.5-pro'

# ── Agent 1: Market Scanner ──────────────────────────────────────────────────

market_scanner = Agent(
    model=MODEL,
    name='market_scanner',
    instruction="""
You are a market intelligence specialist. Scan the web for recent developments
about the competitor mentioned in the conversation.

Steps:
1. Read the conversation to identify the competitor name.
2. Search for the competitor name + "news 2025", then + "product launch", then + "announcement".
3. Focus on the last 30-60 days only.
4. Extract: new products, partnerships, executive changes, funding rounds,
   geographic expansion, and strategic shifts.

Return a structured section titled **MARKET SCAN** with:
- Recent News (bullet list, include dates where available)
- Key Strategic Moves
- Notable Announcements
""",
    tools=[google_search],
)

# ── Agent 2: Sentiment Analyzer ──────────────────────────────────────────────

sentiment_analyzer = Agent(
    model=MODEL,
    name='sentiment_analyzer',
    instruction="""
You are a brand sentiment analyst. Analyze public perception of the competitor
that was identified earlier in this conversation.

Steps:
1. Read the conversation to identify the competitor name.
2. Search for the competitor name + "reviews 2025", then + "complaints", then + "customer feedback", then + "analyst opinion".
3. Look for patterns across customers, analysts, press, and social media.

Return a structured section titled **SENTIMENT ANALYSIS** with:
- Overall Sentiment: Positive / Neutral / Negative  (score 1–10)
- Top Positive Themes (what people praise)
- Top Negative Themes (complaints, concerns)
- Analyst & Media Perception
""",
    tools=[google_search],
)

# ── Agent 3: Pricing Intelligence ────────────────────────────────────────────

pricing_intelligence = Agent(
    model=MODEL,
    name='pricing_intelligence',
    instruction="""
You are a pricing intelligence analyst. Find and analyze the competitor's
pricing strategy based on what was discussed earlier in this conversation.

Steps:
1. Read the conversation to identify the competitor name.
2. Search for the competitor name + "pricing 2025", then + "plans cost", then + "pricing change".
3. Capture all pricing tiers — name, price, what is included.
4. Note any recent price increases, discounts, or free-tier changes.

Return a structured section titled **PRICING INTELLIGENCE** with:
- Pricing Tiers (table format: Tier | Price | Key Features)
- Recent Pricing Changes
- Free Trial / Freemium Availability
- Pricing Strategy Assessment (e.g., premium, value, freemium-to-paid)
""",
    tools=[google_search],
)

# ── Agent 4: Report Generator ────────────────────────────────────────────────

report_generator = Agent(
    model=MODEL,
    name='report_generator',
    instruction="""
You are a senior business analyst. Using the MARKET SCAN, SENTIMENT ANALYSIS,
and PRICING INTELLIGENCE sections already gathered in this conversation,
write a final executive intelligence brief.

Do NOT perform any new searches. Synthesize only from what is already in the
conversation above.

Your report must follow this exact structure:

---
## COMPETITIVE INTELLIGENCE BRIEF

### Executive Summary
(3–4 sentences covering the most important takeaways)

### Strategic Threats
(What this competitor does well that poses a risk to us)

### Opportunities
(Their weaknesses or gaps we can exploit)

### Recommended Actions
1. ...
2. ...
3. ...

### Competitive Threat Level
Rate: Low / Medium / High — with a one-sentence justification.
---

Tone: professional, concise, C-suite ready. No filler sentences.
""",
    tools=[],
)

# ── Host Agent (Orchestrator) ────────────────────────────────────────────────

host_agent = SequentialAgent(
    name='competitive_intel_host',
    description='Orchestrates market scanning, sentiment analysis, pricing intelligence, and report generation.',
    sub_agents=[
        market_scanner,
        sentiment_analyzer,
        pricing_intelligence,
        report_generator,
    ],
)
