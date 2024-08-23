import os
import requests
from happy_hub.types import ChatCompletion, ChatCompletionChunk
from happy_hub.messages import OpenAIMessage
from openai import OpenAI, Stream
from typing import Dict, Any, List, Union
from happy_hub.models.base_model import ModelBackend


class OllamaModel(ModelBackend):
    
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(
        self,
        url: str,
        model_config_dict: Dict
    ) -> None:
        """
        Initialize an instance of OpenAIModel.

        Parameters:
        - model_label (ModelLabel): The label of the model to be used. It should be one of the enum values from ModelLabel.
        - model_config_dict (Dict): A dictionary containing the configuration parameters for the model.

        Note:
        - This method does not set any API keys. The API key should be set in the environment variable 'OPENAI_API_KEY' before calling this method.

        Returns:
        - None: This method does not return any value.
        """
        super().__init__()
        self.model_config_dict = model_config_dict
        self._client = OpenAI(
            base_url=url,
            api_key="123123"
        )                     
    
    def run(
        self,
        messages: List[OpenAIMessage], # type: ignore
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]: # type: ignore
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        response = self._client.chat.completions.create(
            messages=messages,
            **self.model_config_dict,
        )
        return response    

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode,
            which sends partial results each time.
        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)