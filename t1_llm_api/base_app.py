from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


async def start(stream: bool, client: AIClient) -> None:
    """
    Start an interactive chat session with an AI client.

    This function runs a continuous loop that:
    1. Prompts the user for input
    2. Sends the conversation history to the AI
    3. Displays the AI's response
    4. Maintains conversation context

    The loop continues until the user types 'exit'.

    Args:
        stream (bool): If True, use streaming responses (real-time token display).
                      If False, use synchronous responses (complete response at once).
        client (AIClient): The AI client instance to use for generating responses.
    """

    conversation = Conversation()
    print()
    print()
    print()
    print("The class for AI client is - " + client.__class__.__name__)
    print("Connected to AI Model - \""+ client._model_name + "\" with endpoint - \"" + client._endpoint + "\"" )
    print("AI Client is " + ("streaming responses." if stream else "using synchronous responses.") )
    
    print("You can start chatting now!")
    print("Type your question or 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break

        # Add user message to conversation
        conversation.add_message(Message(role=Role.USER, content=user_input))

        # Get AI response
        if stream:
            ai_response = await client.stream_response(conversation.get_messages())
        else:
            ai_response = client.response(conversation.get_messages())

        # Add AI response to conversation
        conversation.add_message(ai_response)
