
import json
from pyexpat.errors import messages

import requests

from commons.constants import ZAI_ENDPOINT, ZAI_API_KEY
from t2_llms_output_tuning._clients._base_client import AIClient
from commons.models.message import Message
from commons.models.role import Role

class ZaiAIClient(AIClient):

    def __init__(self, model_name: str):

        if not ZAI_API_KEY or ZAI_API_KEY.strip() == "":
            raise ValueError("API key cannot be null or empty")
        
        super().__init__(
            endpoint=ZAI_ENDPOINT,
            model_name=model_name,
            api_key=f"Bearer {ZAI_API_KEY}",
            api_key_header_name="Authorization",
        )


    

    def response(self, messages: list[Message], print_request: bool, print_only_content: bool, **kwargs) -> Message:

        headers = {
                self._api_key_header_name: self._api_key,
                "Content-Type": "application/json"
        }

        messages_dicts = [message.to_dict() for message in messages]
        request_data = {
            "model": self._model_name,
            "messages": messages_dicts,
            **kwargs
        }
        
        if print_request:
            self._print_request(request_data, headers)

        response = requests.post(url=self._endpoint, headers=headers, json=request_data)

        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        response_json = response.json()
        choices = response_json.get("choices")
        if not choices or len(choices) == 0:
            raise ValueError("API response contains no choices")

        content = choices[0]["message"]["content"]
        print("" + "=" * 50 + " RESPONSE " + "=" * 50)
        if print_only_content:
            print(content)
        else:
            print(json.dumps(response_json, indent=2, sort_keys=True))
        print("=" * 109)
        return Message(Role.ASSISTANT, content)
    

    