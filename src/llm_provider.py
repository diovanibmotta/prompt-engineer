"""
Abstração de providers de LLM com padrão Factory.

Fornece uma interface unificada para trabalhar com diferentes providers
de LLM (OpenAI, Google Gemini) com suporte a fallback de modelos.
"""

import os
import logging
from typing import Optional, Union
from langchain_core.language_models import BaseLanguageModel

from src.config import get_config


# Configurar logging
logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Exceção para erros relacionados a providers de LLM."""
    pass


class LLMProvider:
    """Factory para criação de instâncias de LLM."""

    # Mapeamento de fallback de modelos
    MODEL_FALLBACK = {
        'openai': {
            'gpt-4o': 'gpt-4o-mini',
            'gpt-4o-mini': 'gpt-3.5-turbo'
        },
        'google': {
            'gemini-2.5-flash': 'gemini-1.5-flash',
            'gemini-1.5-flash': 'gemini-1.5-pro'
        }
    }

    @staticmethod
    def get_llm(
        model: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseLanguageModel:
        """
        Retorna uma instância de LLM configurada baseada no provider.

        Suporta fallback automático de modelos se o modelo principal
        não estiver disponível.

        Args:
            model: Nome do modelo (opcional, usa LLM_MODEL da config por padrão)
            temperature: Temperatura para geração (padrão: 0.0 para determinístico)

        Returns:
            Instância de ChatOpenAI ou ChatGoogleGenerativeAI

        Raises:
            LLMProviderError: Se provider não for suportado ou API key não configurada
        """
        config = get_config()
        provider = config.llm_provider
        model_name = model or config.llm_model

        logger.info(f"Inicializando LLM: provider={provider}, model={model_name}, temperature={temperature}")

        if provider == 'openai':
            return LLMProvider._get_openai_llm(model_name, temperature)
        elif provider == 'google':
            return LLMProvider._get_google_llm(model_name, temperature)
        else:
            raise LLMProviderError(
                f"Provider '{provider}' não suportado.\n"
                f"Providers disponíveis: openai, google"
            )

    @staticmethod
    def _get_openai_llm(
        model: str,
        temperature: float
    ) -> BaseLanguageModel:
        """
        Cria instância de ChatOpenAI com suporte a fallback.

        Args:
            model: Nome do modelo
            temperature: Temperatura para geração

        Returns:
            Instância de ChatOpenAI

        Raises:
            LLMProviderError: Se OPENAI_API_KEY não estiver configurada
        """
        from langchain_openai import ChatOpenAI

        config = get_config()
        api_key = config.openai_api_key

        if not api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY não configurada no .env\n"
                "Obtenha uma chave em: https://platform.openai.com/api-keys"
            )

        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key
            )
            logger.info(f"✓ ChatOpenAI inicializado: {model}")
            print(f"✓ LLM Provider: OpenAI ({model})")
            return llm

        except Exception as e:
            # Tentar fallback
            fallback_model = LLMProvider.MODEL_FALLBACK['openai'].get(model)
            if fallback_model and fallback_model != model:
                logger.warning(
                    f"Erro ao usar modelo {model}, tentando fallback: {fallback_model}"
                )
                print(f"⚠ Modelo {model} não disponível, usando fallback: {fallback_model}")
                return LLMProvider._get_openai_llm(fallback_model, temperature)
            else:
                raise LLMProviderError(
                    f"Erro ao inicializar ChatOpenAI com modelo {model}: {str(e)}"
                )

    @staticmethod
    def _get_google_llm(
        model: str,
        temperature: float
    ) -> BaseLanguageModel:
        """
        Cria instância de ChatGoogleGenerativeAI com suporte a fallback.

        Args:
            model: Nome do modelo
            temperature: Temperatura para geração

        Returns:
            Instância de ChatGoogleGenerativeAI

        Raises:
            LLMProviderError: Se GOOGLE_API_KEY não estiver configurada
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        config = get_config()
        api_key = config.google_api_key

        if not api_key:
            raise LLMProviderError(
                "GOOGLE_API_KEY não configurada no .env\n"
                "Obtenha uma chave em: https://aistudio.google.com/app/apikey"
            )

        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
            )
            logger.info(f"✓ ChatGoogleGenerativeAI inicializado: {model}")
            print(f"✓ LLM Provider: Google Gemini ({model})")
            return llm

        except Exception as e:
            # Tentar fallback
            fallback_model = LLMProvider.MODEL_FALLBACK['google'].get(model)
            if fallback_model and fallback_model != model:
                logger.warning(
                    f"Erro ao usar modelo {model}, tentando fallback: {fallback_model}"
                )
                print(f"⚠ Modelo {model} não disponível, usando fallback: {fallback_model}")
                return LLMProvider._get_google_llm(fallback_model, temperature)
            else:
                raise LLMProviderError(
                    f"Erro ao inicializar ChatGoogleGenerativeAI com modelo {model}: {str(e)}"
                )

    @staticmethod
    def get_eval_llm(temperature: float = 0.0) -> BaseLanguageModel:
        """
        Retorna LLM configurado especificamente para avaliação.

        Usa EVAL_MODEL da configuração, que pode ser diferente do modelo principal.

        Args:
            temperature: Temperatura para geração

        Returns:
            Instância de LLM configurada para avaliação
        """
        config = get_config()
        eval_model = config.eval_model
        logger.info(f"Inicializando LLM de avaliação: {eval_model}")
        return LLMProvider.get_llm(model=eval_model, temperature=temperature)


# Funções de conveniência (interface pública)
def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> BaseLanguageModel:
    """
    Obtém uma instância de LLM configurada.

    Args:
        model: Nome do modelo (opcional)
        temperature: Temperatura para geração

    Returns:
        Instância de LLM
    """
    return LLMProvider.get_llm(model=model, temperature=temperature)


def get_eval_llm(temperature: float = 0.0) -> BaseLanguageModel:
    """
    Obtém uma instância de LLM para avaliação.

    Args:
        temperature: Temperatura para geração

    Returns:
        Instância de LLM para avaliação
    """
    return LLMProvider.get_eval_llm(temperature=temperature)
