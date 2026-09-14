"""Generic OpenAI-compatible text generation route for hosted inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from llm.contracts import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable

router = APIRouter(prefix="/v1", tags=["generation"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Generate structured completions conforming to an OpenAI-compatible JSON Schema."""
    runtime = runtime_from(request)
    try:
        model_status = loaded_model_status(runtime, "text_generation")
    except Exception as error:
        raise unavailable("Text generation model is not ready", error) from error

    if model_status.checkpoint and payload.model != model_status.checkpoint:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested model '{payload.model}' does not match "
                f"loaded checkpoint '{model_status.checkpoint}'"
            ),
        )

    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in payload.messages
    ]
    schema = payload.response_format.json_schema.schema_

    try:
        content = runtime.generate_chat(
            messages,
            response_schema=schema,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except Exception as error:
        raise unavailable("Text generation inference failed", error) from error

    return ChatCompletionResponse(
        model=payload.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(
                    role="assistant",
                    content=content,
                ),
                finish_reason="stop",
            )
        ],
    )
