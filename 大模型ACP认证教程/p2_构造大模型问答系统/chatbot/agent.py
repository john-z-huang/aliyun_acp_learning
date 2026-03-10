import os
from typing import Iterable, Literal

from agentscope.agent import ReActAgent
from agentscope.formatter import (
    DashScopeChatFormatter,
    DashScopeMultiAgentFormatter,
)
from agentscope.model import DashScopeChatModel


def create_agent(
    name: str,
    sys_prompt: str,
    model_name: str = "qwen-plus",
    multi_agent: bool = True,
    long_term_memory: object | None = None,
    long_term_memory_mode: Literal["static_control", "agent_control"] | None = None,
) -> ReActAgent:
    """Create a ReActAgent with a DashScope chat model.

    The API key is read from ``DASHSCOPE_API_KEY``; a placeholder is used if absent
    so that the script remains runnable in dry-run/demo environments.
    """

    model = DashScopeChatModel(
        model_name=model_name,
        api_key=os.environ.get("DASHSCOPE_API_KEY", "your-api-key"),
        stream=False,
    )

    formatter = (
        DashScopeMultiAgentFormatter() if multi_agent else DashScopeChatFormatter()
    )

    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        long_term_memory=long_term_memory,
        long_term_memory_mode=long_term_memory_mode,
    )


def disable_console_output(agents: Iterable[ReActAgent]) -> None:
    for agent in agents:
        agent.set_console_output_enabled(False)
