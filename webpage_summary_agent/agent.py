from urllib.parse import urlparse
from google.adk.agents import Agent
import google.generativeai as genai


def fetch_webpage_summary(url: str) -> dict:
    """Fetches a webpage and generates a one-paragraph text summary using Gemini's URL access.

    Args:
        url (str): The URL of the webpage to summarize.

    Returns:
        dict: Status and summary or error message.
    """
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {
                "status": "error",
                "error_message": "Invalid URL format. Please provide a valid URL with protocol (http:// or https://).",
            }

        # Use Gemini's URL access to read and summarize the webpage
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
        Please read the webpage at the following URL and provide a concise one-paragraph summary of its content: {url}

        The summary should:
        - Be exactly one paragraph (no line breaks)
        - Capture the main topics and key information
        - Be clear and informative
        - Include the webpage title if available
        - Be approximately 100-200 words

        Format your response as just the summary paragraph, without any additional text or formatting.
        """

        response = model.generate_content(prompt)

        if response and response.text:
            summary = response.text.strip()

            return {
                "status": "success",
                "summary": summary,
                "url": url,
            }
        else:
            return {
                "status": "error",
                "error_message": "Failed to generate summary from Gemini model.",
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"An error occurred while processing the webpage: {str(e)}",
        }


# Create the Google ADK agent
root_agent = Agent(
    name="webpage_summary_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent that fetches webpages and generates concise one-paragraph text summaries."
    ),
    instruction=(
        "You are a helpful agent that can visit any webpage and provide a clear, "
        "concise one-paragraph summary of its content. When a user provides a URL, "
        "you will fetch the webpage, analyze its content, and return a summary that "
        "captures the main topics and key information discussed on the page."
    ),
    tools=[fetch_webpage_summary],
)
