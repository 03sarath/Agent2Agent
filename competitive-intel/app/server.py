"""
Flask web server for the Competitive Intelligence platform.
Exposes three routes:
  GET  /          — serves the UI
  POST /analyze   — runs the agent pipeline, returns JSON report
  GET  /health    — health check

Session management strategy (controlled by AGENT_ENGINE_RESOURCE_NAME in .env):
  - Not set → local mode: InMemorySessionService (dev/testing, ephemeral)
  - Set     → production mode: Vertex AI Agent Engine (persistent sessions, managed)
"""
import asyncio
import os

import vertexai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents import host_agent
from app.utils import build_query, make_session_id, sanitize_competitor_name

load_dotenv()

app = Flask(__name__)

USER_ID = 'flask_user'

# ── Backend selection ─────────────────────────────────────────────────────────
# Set AGENT_ENGINE_RESOURCE_NAME in .env after running deploy.py.
# Example: projects/my-project/locations/us-central1/reasoningEngines/1234567890

RESOURCE_NAME = os.getenv('AGENT_ENGINE_RESOURCE_NAME', '')

if RESOURCE_NAME:
    # Production: delegate everything to Vertex AI Agent Engine.
    # vertexai.Client correctly binds stream_query and all other methods.
    _client = vertexai.Client(
        project=os.environ['GOOGLE_CLOUD_PROJECT'],
        location=os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'),
    )
    remote_agent = _client.agent_engines.get(name=RESOURCE_NAME)
    runner = None
    print(f'[server] Mode: Vertex AI Agent Engine → {RESOURCE_NAME}')
else:
    # Local dev: run the agent in-process with ephemeral in-memory services.
    runner = Runner(
        app_name='competitive_intel',
        agent=host_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    remote_agent = None
    print('[server] Mode: local (InMemorySessionService)')


# ── Core agent execution ─────────────────────────────────────────────────────

async def _run_local(query: str, session_id: str) -> str:
    """Run against the local in-process ADK runner (dev mode).

    ADK 1.9.0 requires an explicit session before run_async.
    """
    existing = await runner.session_service.get_session(
        app_name='competitive_intel',
        user_id=USER_ID,
        session_id=session_id,
    )
    if not existing:
        await runner.session_service.create_session(
            app_name='competitive_intel',
            user_id=USER_ID,
            session_id=session_id,
        )

    message = types.Content(role='user', parts=[types.Part(text=query)])
    final_response = ''
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
    return final_response


def _run_remote(query: str, session_id: str) -> str:
    """Run against Vertex AI Agent Engine (production mode).

    Agent Engine manages session persistence — the same session_id across
    requests means the agent remembers prior context for that competitor.
    """
    # Ensure a session exists for this session_id on Vertex
    try:
        remote_agent.get_session(user_id=USER_ID, session_id=session_id)
    except Exception:
        remote_agent.create_session(user_id=USER_ID, session_id=session_id)

    final_response = ''
    for chunk in remote_agent.stream_query(
        user_id=USER_ID,
        session_id=session_id,
        message=query,
    ):
        if 'content' in chunk:
            for part in chunk['content'].get('parts', []):
                if 'text' in part:
                    final_response += part['text']
    return final_response


def run_agent(query: str, session_id: str) -> str:
    """Unified entry point — routes to local or remote backend automatically."""
    if remote_agent:
        return _run_remote(query, session_id)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_local(query, session_id))
    finally:
        loop.close()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body is required'}), 400

    competitor_raw = body.get('competitor', '')
    context = body.get('context', '')

    try:
        competitor = sanitize_competitor_name(competitor_raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    query = build_query(competitor, context)
    session_id = make_session_id(competitor)

    try:
        report = run_agent(query, session_id)
        return jsonify({'success': True, 'report': report, 'competitor': competitor})
    except Exception as e:
        return jsonify({'error': f'Agent execution failed: {str(e)}'}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
