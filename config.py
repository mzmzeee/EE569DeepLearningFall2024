"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "llmapps"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="benbarka/llmapps/vector_store:latest",
    chat_prompt_artifact="benbarka/llmapps/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-4o-mini",
    eval_model="gpt-4o-mini",
    eval_artifact="benbarka/llmapps/generated_examples:v0",
)