from pathlib import Path
import os
from dotenv import load_dotenv
import json
from pprint import pprint
from datetime import datetime
from google.adk.agents import LlmAgent  # Using LlmAgent instead
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio


load_dotenv()
KEEP_EXPORT_ABSOLUTE_PATH = os.getenv("KEEP_EXPORT_ABSOLUTE_PATH")
if not KEEP_EXPORT_ABSOLUTE_PATH:
    raise ValueError("KEEP_EXPORT_ABSOLUTE_PATH environment variable is not set.")

# Configure Google API key for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Set the API key for Google AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Set the model to use for the LLM agent
# MODEL = "gemma-3-4b-it"
MODEL = "gemini-2.0-flash"


# function to print available keys and sub-keys in the JSON file
def print_keys(data, prefix=""):
    """
    output example:
    ```
    color
    isTrashed
    isPinned
    isArchived
    annotations
    annotations[0]
    annotations[0].description
    annotations[0].source
    annotations[0].title
    annotations[0].url
    annotations[1]
    annotations[1].description
    annotations[1].source
    annotations[1].title
    annotations[1].url
    textContent
    title
    userEditedTimestampUsec
    createdTimestampUsec
    textContentHtml
    labels
    labels[0]
    labels[0].name
    ```
    """
    if isinstance(data, dict):
        for key in data:
            full_key = f"{prefix}.{key}" if prefix else key
            print(full_key)
            print_keys(data[key], full_key)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{prefix}[{index}]"
            print(full_key)
            print_keys(item, full_key)


def convert_timestamp(timestamp_usec):
    """Convert microsecond timestamp to datetime object."""
    return datetime.fromtimestamp(timestamp_usec / 1000000) if timestamp_usec else None


# function to read and process one note to for multimodal llm
def process_note(note):
    """
    Process a single note to extract relevant information for multimodal LLM.
    This function can be expanded based on the specific requirements of the LLM.
    """

    # get attachments if any
    attachments = note.get("attachments", [])
    # change str to Path object
    attachments = [
        Path(KEEP_EXPORT_ABSOLUTE_PATH, att["filePath"]) for att in attachments
    ]

    processed_note = {
        "textContent": note.get("textContent", ""),
        "createdTimestampUsec": convert_timestamp(note.get("createdTimestampUsec")),
        "userEditedTimestampUsec": convert_timestamp(
            note.get("userEditedTimestampUsec")
        ),
        "attachments": attachments,
        "isTrashed": note.get("isTrashed", False),
        "isArchived": note.get("isArchived", False),
    }
    return processed_note


def create_summary(processed_note):
    """
    Create a summary of the processed note using an LLM agent.
    """
    # Skip empty or trashed notes
    if not processed_note["textContent"].strip() or processed_note["isTrashed"]:
        return "Empty or trashed note - no summary generated"

    # Prepare the prompt with note information
    text_content = processed_note["textContent"]
    created_date = processed_note["createdTimestampUsec"]
    edited_date = processed_note["userEditedTimestampUsec"]
    attachments = processed_note["attachments"]
    is_archived = processed_note["isArchived"]

    # Build the prompt
    prompt_parts = [f"Text content: {text_content}"]

    if created_date:
        prompt_parts.append(f"Created: {created_date}")

    if edited_date and edited_date != created_date:
        prompt_parts.append(f"Last edited: {edited_date}")

    if attachments:
        prompt_parts.append(
            f"Attachments: {len(attachments)} file(s) - {[att.name for att in attachments]}"
        )

    if is_archived:
        prompt_parts.append("Note is archived")

    full_prompt = (
        "Please summarize this Google Keep note in 1-2 concise sentences. Focus on the main topic and key information. "
        "If there are images attached, analyze them and incorporate relevant visual information into the summary:\n\n"
        + "\n".join(prompt_parts)
    )

    try:
        # Create a summarizer agent
        summarizer_agent = LlmAgent(
            name="keep_note_summarizer",
            model=MODEL,
            description="Agent that creates summaries of Google Keep notes with text and image content.",
            instruction=(
                "You are a helpful agent that summarizes Google Keep notes. "
                "Create concise, informative summaries in 1-2 sentences that capture "
                "the main topics and key information from both text content and any attached images. "
                "When images are present, analyze them and incorporate relevant visual information "
                "into the summary. If attachments are mentioned, briefly note them in the summary."
            ),
        )

        # Use async runner to get response
        async def get_summary():
            runner = InMemoryRunner(agent=summarizer_agent, app_name="KeepSummarizer")
            session = await runner.session_service.create_session(
                app_name="KeepSummarizer", user_id="keep_user"
            )

            # Create content parts - start with text
            content_parts = [types.Part(text=full_prompt)]

            # Add image attachments if they exist and are valid image files
            if attachments:
                for attachment_path in attachments:
                    if attachment_path.exists() and attachment_path.suffix.lower() in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".webp",
                    ]:
                        try:
                            # Read image file and create image part
                            with open(attachment_path, "rb") as img_file:
                                image_data = img_file.read()

                            # Determine MIME type based on file extension
                            mime_type_map = {
                                ".png": "image/png",
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".gif": "image/gif",
                                ".webp": "image/webp",
                            }
                            mime_type = mime_type_map.get(
                                attachment_path.suffix.lower(), "image/png"
                            )

                            # Add image part to content
                            content_parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type=mime_type, data=image_data
                                    )
                                )
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not load image {attachment_path}: {e}"
                            )

            content = types.Content(role="user", parts=content_parts)

            result_text = ""
            async for event in runner.run_async(
                user_id=session.user_id, session_id=session.id, new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            result_text += part.text

            return result_text.strip() if result_text else "No summary generated"

        # Run the async function
        return asyncio.run(get_summary())

    except Exception as e:
        return f"Error generating summary: {str(e)}"


if __name__ == "__main__":

    for json_file in Path(KEEP_EXPORT_ABSOLUTE_PATH).glob("**/*.json"):
        if not json_file.is_file():
            print(f"Skipping {json_file}, not a file.")

        content = json.load(open(json_file, "r", encoding="utf-8"))
        processed_note = process_note(content)

        # if not processed_note["attachments"]:
        #     continue
        print(processed_note)

        # use adk llm agent to create a summary of processed_note content
        summary = create_summary(processed_note)
        print(summary)

        print("\n" + "=" * 40 + "\n")
    print("Done processing all JSON files.")
