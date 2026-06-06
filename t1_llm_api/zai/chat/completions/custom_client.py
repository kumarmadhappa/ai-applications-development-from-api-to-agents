import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomZaiAIClient(AIClient):
    """
    Custom HTTP client for ZAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
    """
    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the ZAI client with Bearer token authentication.

        Args:
            endpoint (str): The ZAI API endpoint URL.
            model_name (str): The ZAI model identifier (e.g., 'glm-5.1').
            system_prompt (str): The system-level instruction for the model.
            api_key (str): The raw ZAI API key (will be prefixed with 'Bearer ').

        Raises:
            ValueError: If api_key is None, empty, or contains only whitespace.
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            api_key=f"Bearer {api_key}"
        )

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        # https://docs.z.ai/api-reference/introduction
        
        # - Prepare headers with authorization and content type
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en"
        }

        # - Prepare message history with System prompt
        messages_dicts = [
            {"role": "system", "content": self._system_prompt},
            *[message.to_dict() for message in messages]
        ]

        request_data = {
            "model": self._model_name,
            "messages": messages_dicts
        }

        # - Execute post request to AI API (use `requests`)
        response = requests.post(url=self._endpoint, headers=headers, json=request_data)
        
        # - Parse response
        # - Print response to console
        # - Return ASSISTANT message

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
                print(content)
                return Message(Role.ASSISTANT, content)
            raise ValueError("No Choice has been present in the response")
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        
       

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
        """

        # - Prepare headers with authorization and content type

        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en"
        }

        # - Prepare message history with System prompt
        messages_dicts = [
            {"role": "system", "content": self._system_prompt},
            *[message.to_dict() for message in messages]
        ]

        request_data = {
            "model": self._model_name,
            "messages": messages_dicts,
            "stream": True  # Enable streaming
        }

        # - Execute post request to AI API (use `requests`)
        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, headers=headers, json=request_data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text}")

                # - Handle stream with chunks
                # - Parse response
                # - Print chunks to console
                # - Return ASSISTANT message

                content = ""
                async for line in resp.content:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[len("data: "):]
                        if data_str == "[DONE]":
                            break  # End of stream
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                delta_content = choices[0].get("delta", {}).get("content", "")
                                print(delta_content, end="", flush=True)  # Print token immediately
                                content += delta_content
                        except json.JSONDecodeError:
                            continue  # Ignore malformed lines
                print()
                return Message(role=Role.ASSISTANT, content=''.join(content))
