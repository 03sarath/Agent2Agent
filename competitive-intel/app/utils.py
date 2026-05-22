"""
Pure utility functions — no AI dependencies, fully testable.
"""


def sanitize_competitor_name(name: str) -> str:
    """Strip whitespace and validate competitor name."""
    name = name.strip()
    if not name:
        raise ValueError('Competitor name cannot be empty')
    if len(name) > 200:
        raise ValueError('Competitor name is too long (max 200 characters)')
    return name


def build_query(competitor: str, context: str) -> str:
    """Build the prompt sent to the host agent."""
    competitor = sanitize_competitor_name(competitor)
    query = (
        f'Perform a comprehensive competitive intelligence analysis for: {competitor}. '
        f'Cover recent market moves, brand sentiment, and pricing strategy.'
    )
    if context and context.strip():
        query += f' Additional context: {context.strip()}'
    return query


def make_session_id(competitor: str) -> str:
    """Create a stable session ID from a competitor name."""
    return 'session_' + competitor.strip().lower().replace(' ', '_')
