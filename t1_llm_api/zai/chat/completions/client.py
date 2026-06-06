from zai import ZaiClient
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient

class ZaiAIClient(AIClient):
    """
    Client for ZAI Chat Completions API using the official SDK.

    This implementation uses the official ZAI Python library to interact
    with the Chat Completions API, providing both synchronous and streaming
    response capabilities.

    Attributes:
        _client (ZaiClient): ZAI client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the ZAI Chat Completions client with SDK.

        Args:
            endpoint (str): The ZAI API endpoint (for compatibility, not used by SDK).
            model_name (str): The ZAI model to use (e.g., 'glm-5.1').
            system_prompt (str): The system message to guide the model's behavior.
            api_key (str): The ZAI API key for authentication.
        """

        # Call to __init__ of super class (AIClient expects api_key then system_prompt)
        super().__init__(endpoint, model_name, api_key, system_prompt)

        # Set up ZAI SDK client
        self._client = ZaiClient(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from ZAI's Chat Completions API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """

        # - Prepare message history with System prompt
        messages_dicts = [
            {"role": "system", "content": self._system_prompt},
            *[message.to_dict() for message in messages]
        ]
        # - Call client
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages_dicts
        )
        content = response.choices[0].message.content

        # - Print response to console
        print(content)

        # - Return ASSISTANT message
        return Message(role=Role.ASSISTANT, content=content)
    

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from ZAI's Chat Completions API.

        The response is streamed token-by-token, with each chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives for real-time display.
        """

        # - Prepare message history with System prompt
        messages_dicts = [
            {"role": "system", "content": self._system_prompt},
            *[message.to_dict() for message in messages]
        ]

        # - Call client with streaming mode
        response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages_dicts,
                stream=True
        )
        content = []
        for chunk in response:
            delta = None
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                content.append(delta)
                print(delta, end="", flush=True)
        print()

        return Message(role=Role.ASSISTANT, content=content)