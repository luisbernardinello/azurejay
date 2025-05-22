from typing import Annotated
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

def create_custom_handoff_to_profile(*, agent_name: str = "Profile") -> BaseTool:
    """Create a custom handoff tool to transfer to the profile agent"""
    
    name = "transfer_to_profile"
    description = "Transfer to the profile agent when user shares personal information like name, location, interests, family, job, etc."
    
    @tool(name, description=description)
    def handoff_to_profile(
        reason: Annotated[str, "Reason for transferring to profile agent"],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name} - {reason}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        messages = state["messages"]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
            },
        )
    
    return handoff_to_profile

def create_custom_handoff_to_correction(*, agent_name: str = "Correction") -> BaseTool:
    """Create a custom handoff tool to transfer to the correction agent"""
    
    name = "transfer_to_correction_agent"
    description = "Transfer to the correction agent when user makes grammar, spelling, or punctuation errors that need correction."
    
    @tool(name, description=description)
    def handoff_to_correction(
        reason: Annotated[str, "Reason for transferring to correction agent"],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name} - {reason}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        messages = state["messages"]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
            },
        )
    
    return handoff_to_correction