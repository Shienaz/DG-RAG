import logging
import os
import time

import dotenv
import tiktoken
from openai import OpenAI

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
# siliconflow endpoints are OpenAI-compatible; we keep the client but override base_url.
SILICONFLOW_TOKEN_LIMIT = 128000


def get_token_limit(model: str = DEFAULT_MODEL) -> int:
    """
    Returns the token limitation of provided model.
    DeepSeek Terminus (via siliconflow) follows GPT-4o-class context length.
    """
    if model == DEFAULT_MODEL:
        return SILICONFLOW_TOKEN_LIMIT
    # Fallback to a sensible default if another model is passed.
    return SILICONFLOW_TOKEN_LIMIT


class ChatGPT(BaseLanguageModel):
    """A class that interacts with OpenAI-compatible chat models (SiliconFlow + DeepSeek).

    This class provides functionality to generate text using ChatGPT models while handling
    token limits, retries, and various input formats.

    Args:
        model_name_or_path (str): The name or path of the ChatGPT-compatible model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5
        base_url (str, optional): Base URL of the OpenAI-compatible endpoint.
        api_key_env (str, optional): Environment variable name holding the API key.

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the ChatGPT model being used
        maximun_token (int): Maximum token limit for the specified model
        client (OpenAI): OpenAI client instance for API interactions

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the ChatGPT model

    Raises:
        KeyError: If the specified model is not found when calculating tokens
        Exception: If generation fails after maximum retries
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        retry: int = 5,
        base_url: str | None = None,
        api_key_env: str = "SILICONFLOW_API_KEY",
    ):
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token = get_token_limit(self.model_name)
        self.base_url = base_url or DEFAULT_BASE_URL

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is not set. Please export your SiliconFlow API key."
            )

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        self.client = client

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # DeepSeek is not bundled in tiktoken; fall back to cl100k_base.
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the OpenAI-compatible API.

        This method sends a request to the SiliconFlow endpoint and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=message, timeout=60, temperature=0.0
                )
                result = response.choices[0].message.content.strip()  # type: ignore
                return result
            except Exception as e:
                logger.error("Message: ", llm_input)
                logger.error("Number of token: ", self.token_len(message_string))
                logger.error(e)
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
