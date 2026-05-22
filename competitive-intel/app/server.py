"""
Flask web server for the Competitive Intelligence platform.
Exposes three routes:
  GET  /          — serves the UI
  POST /analyze   — calls the deployed Vertex AI Agent Engine, returns JSON report
  GET  /health    — health check

Authentication: run `gcloud auth application-default login` before starting.

Required .env variables:
  GOOGLE_CLOUD_PROJECT          — your GCP project ID
  GOOGLE_CLOUD_LOCATION         — e.g. us-central1
  AGENT_ENGINE_RESOURCE_NAME    — projects/.../reasoningEngines/...
"""
import os

import vertexai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from app.utils import build_query, make_session_id, sanitize_competitor_name

load_dotenv()

app = Flask(__name__)

USER_ID = 'flask_user'

# ── Connect to the deployed agent ─────────────────────────────────────────────
PROJECT       = os.environ['GOOGLE_CLOUD_PROJECT']
LOCATION      = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
RESOURCE_NAME = os.environ['AGENT_ENGINE_RESOURCE_NAME']

client = vertexai.Client(project=PROJECT, location=LOCATION)
agent  = client.agent_engines.get(name=RESOURCE_NAME)

print(f'[server] Connected to: {RESOURCE_NAME}')


# ── Core agent execution ──────────────────────────────────────────────────────

def run_agent(query: str, session_id: str) -> str:
    """Create a session (if needed) and stream the agent response."""
    try:
        agent.get_session(user_id=USER_ID, session_id=session_id)
    except Exception:
        agent.create_session(user_id=USER_ID, session_id=session_id)

    final_response = ''
    for chunk in agent.stream_query(
        user_id=USER_ID,
        session_id=session_id,
        message=query,
    ):
        content = chunk.get('content', {})
        if content:
            for part in content.get('parts', []):
                text = part.get('text', '')
                if text:
                    final_response += text
    return final_response


# ── Routes ────────────────────────────────────────────────────────────────────

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

    query      = build_query(competitor, context)
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
