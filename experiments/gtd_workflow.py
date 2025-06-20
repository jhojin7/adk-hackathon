#!/usr/bin/env python3
"""
GTD Multi-Agent Workflow Implementation
A complete Getting Things Done system using Google ADK agents.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import Runner, InMemoryRunner
from google.genai import types


# GTD Task Management Tools
def capture_task(task_description: str, context: str = "") -> str:
    """Capture a new task or idea into the inbox."""
    timestamp = datetime.now().isoformat()
    return f"Task captured: '{task_description}' (Context: {context}) at {timestamp}"


def clarify_task(task: str) -> str:
    """Clarify what the task actually means and if it's actionable."""
    return f"Task clarified: '{task}' is actionable. Next action: Process {task}. Context: @computer"


def organize_task(task: str) -> str:
    """Organize the task into appropriate lists/contexts."""
    return f"Task organized into @computer context: {task}"


def review_tasks() -> str:
    """Perform weekly review of all tasks and projects."""
    return "Weekly review completed: All lists reviewed, projects updated, next actions identified"


def engage_with_task(task: str) -> str:
    """Actually do the work - engage with the task."""
    return f"Working on task: {task} - Task completed successfully"


# GTD Agents Definition
capture_agent = LlmAgent(
    name="capture_agent",
    model="gemini-2.0-flash",
    description="Captures tasks and ideas into the GTD inbox",
    instruction="""  
    You are the Capture agent in a GTD system. Your job is to:  
    1. Listen for any tasks, ideas, or commitments mentioned by the user  
    2. Use the capture_task tool to record them immediately  
    3. Don't judge or organize - just capture everything  
    4. Be thorough and don't let anything slip through  
    """,
    tools=[capture_task],
)

clarify_agent = LlmAgent(
    name="clarify_agent",
    model="gemini-2.0-flash",
    description="Clarifies what captured items actually mean",
    instruction="""  
    You are the Clarify agent in a GTD system. Your job is to:  
    1. Take captured items and determine what they really mean  
    2. Ask: Is this actionable? What's the desired outcome?  
    3. Use the clarify_task tool to process each item  
    4. Be specific about next actions and contexts  
    """,
    tools=[clarify_task],
)

organize_agent_inbox = LlmAgent(
    name="organize_agent_inbox",
    model="gemini-2.0-flash",
    description="Organizes clarified tasks into appropriate lists",
    instruction="""  
    You are the Organize agent in a GTD system. Your job is to:  
    1. Take clarified tasks and put them in the right place  
    2. Sort by context (@calls, @computer, @errands, etc.)  
    3. Use the organize_task tool to categorize everything  
    4. Keep the system clean and organized  
    """,
    tools=[organize_task],
)

organize_agent_context = LlmAgent(
    name="organize_agent_context",
    model="gemini-2.0-flash",
    description="Organizes clarified tasks into appropriate lists",
    instruction="""  
    You are the Organize agent in a GTD system. Your job is to:  
    1. Take clarified tasks and put them in the right place  
    2. Sort by context (@calls, @computer, @errands, etc.)  
    3. Use the organize_task tool to categorize everything  
    4. Keep the system clean and organized  
    """,
    tools=[organize_task],
)

review_agent = LlmAgent(
    name="review_agent",
    model="gemini-2.0-flash",
    description="Performs regular reviews of the GTD system",
    instruction="""  
    You are the Review agent in a GTD system. Your job is to:  
    1. Conduct weekly reviews of all lists and projects  
    2. Update project statuses and next actions  
    3. Use the review_tasks tool to perform comprehensive reviews  
    4. Ensure the system stays current and trusted  
    """,
    tools=[review_tasks],
)

engage_agent = LlmAgent(
    name="engage_agent",
    model="gemini-2.0-flash",
    description="Actually executes tasks and gets work done",
    instruction="""  
    You are the Engage agent in a GTD system. Your job is to:  
    1. Look at organized next actions and choose what to work on  
    2. Consider context, time available, and energy level  
    3. Use the engage_with_task tool to actually do the work  
    4. Focus on execution and completion  
    """,
    tools=[engage_with_task],
)

# GTD Workflow Composition
# Sequential workflow for processing inbox items
inbox_processor = SequentialAgent(
    name="inbox_processor",
    description="Processes inbox items through capture -> clarify -> organize",
    sub_agents=[capture_agent, clarify_agent, organize_agent_inbox],
)

# Parallel agent for handling multiple contexts simultaneously
context_processor = ParallelAgent(
    name="context_processor",
    description="Processes multiple contexts in parallel for efficiency",
    sub_agents=[organize_agent_context, engage_agent],
)

# Loop agent for weekly reviews
review_loop = LoopAgent(
    name="review_loop",
    description="Performs regular GTD reviews",
    max_iterations=1,  # Set to 1 for demo, normally would be ongoing
    sub_agents=[review_agent],
)

# Main GTD Coordinator
gtd_coordinator = LlmAgent(
    name="gtd_coordinator",
    model="gemini-2.0-flash",
    description="Coordinates the entire GTD workflow system",
    instruction="""  
    You are the GTD Coordinator. You manage the entire Getting Things Done workflow:  
      
    1. Start by processing any inbox items through the inbox_processor  
    2. Use context_processor for handling multiple task contexts  
    3. Regularly trigger review_loop for system maintenance  
    4. Help users understand and follow GTD principles  
    5. Provide guidance on GTD best practices  
      
    Always explain what's happening in the GTD process and why each step matters.  
    """,
    sub_agents=[inbox_processor, context_processor, review_loop],
)


async def main():
    """Main function to run the GTD workflow."""
    print("🚀 Starting GTD Multi-Agent Workflow System")
    print("=" * 50)

    # Initialize the runner with required parameters
    runner = InMemoryRunner(agent=gtd_coordinator, app_name="GTD_Workflow")

    # Create a session for the conversation
    session = await runner.session_service.create_session(
        app_name="GTD_Workflow", user_id="test_user"
    )

    # Example GTD workflow execution
    test_queries = [
        # "I need to call the dentist to schedule a cleaning appointment",
        "Research vacation destinations for summer trip",
        # "Buy groceries for the week",
        "Review quarterly budget numbers",
    ]

    for query in test_queries:
        print(f"\n📝 Processing: {query}")
        print("-" * 30)

        try:
            # Create user content
            content = types.Content(role="user", parts=[types.Part(text=query)])

            # Run the GTD coordinator with the query
            async for event in runner.run_async(
                user_id=session.user_id, session_id=session.id, new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"🤖 {event.author}: {part.text}")

        except Exception as e:
            print(f"❌ Error processing '{query}': {e}")

    print("\n✅ GTD Workflow Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
