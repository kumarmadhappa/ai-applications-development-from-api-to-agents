import asyncio

from t1_llm_api.base_app import start
from commons.constants import ZAI_ENDPOINT, ZAI_API_KEY, DEFAULT_SYSTEM_PROMPT
from t1_llm_api.zai.chat.completions.client import ZaiAIClient
from t1_llm_api.zai.chat.completions.custom_client import CustomZaiAIClient

zai_client = ZaiAIClient(
    endpoint=ZAI_ENDPOINT,
    model_name='glm-5.1',
    api_key=ZAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)

zai_custom_client = CustomZaiAIClient(
    endpoint=ZAI_ENDPOINT,
    model_name='glm-5.1',
    api_key=ZAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)

asyncio.run(
    start(True, zai_custom_client)
)
