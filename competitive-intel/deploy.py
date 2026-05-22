"""
Deploy the Competitive Intelligence agent to Vertex AI Agent Engine.

Usage:
    python deploy.py

Prerequisites:
    1. gcloud auth application-default login
    2. gcloud services enable aiplatform.googleapis.com
    3. Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in .env
"""
import os

import vertexai
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines

from app.agents import host_agent

load_dotenv()

PROJECT        = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION       = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
# GCS bucket used to stage agent code before deployment.
# Must exist and be in the same region as LOCATION.
# Create once with: gsutil mb -l us-central1 gs://<your-project-id>-agent-staging
STAGING_BUCKET = os.getenv("STAGING_BUCKET", f"gs://{PROJECT}-agent-staging")


def deploy() -> str:
    """Deploy the host agent to Vertex AI Agent Engine and return its resource name."""
    print(f"Project        : {PROJECT}")
    print(f"Location       : {LOCATION}")
    print(f"Staging bucket : {STAGING_BUCKET}")
    print()

    vertexai.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING_BUCKET)

    adk_app = reasoning_engines.AdkApp(
        agent=host_agent,
        enable_tracing=False,
    )

    print("Deploying to Vertex AI Agent Engine (3-5 minutes)...")

    remote_agent = reasoning_engines.ReasoningEngine.create(
        adk_app,
        requirements=[
            "google-adk==1.9.0",
            "google-genai",
            "cloudpickle>=3.0.0",
        ],
        display_name="Competitive Intelligence Agent",
        description=(
            "Multi-agent competitive intelligence pipeline: "
            "market scan → sentiment → pricing → executive report"
        ),
    )

    resource_name = remote_agent.resource_name
    print(f"\nDeployed: {resource_name}")

    # Quick smoke test
    print("\nRunning smoke test...")
    session = remote_agent.create_session(user_id="deploy_test")
    chunks = list(remote_agent.stream_query(
        user_id="deploy_test",
        session_id=session["id"],
        message="Analyze competitor: Anthropic",
    ))
    print(f"Smoke test received {len(chunks)} response chunks.")
    print("\nDeployment complete.")

    return resource_name


if __name__ == "__main__":
    name = deploy()
    print(f"\nResource name (save this):\n{name}")
