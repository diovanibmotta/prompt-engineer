"""
Gerenciamento centralizado de configuração do projeto.

Carrega e valida variáveis de ambiente necessárias para o funcionamento
do sistema de otimização de prompts com LangSmith.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


class ConfigError(Exception):
    """Exceção para erros de configuração."""
    pass


class Config:
    """Gerenciador centralizado de configuração."""

    # Variáveis obrigatórias
    REQUIRED_VARS = [
        'LANGSMITH_API_KEY',
        'USERNAME_LANGSMITH_HUB',
        'LLM_PROVIDER'
    ]

    # Variáveis específicas por provider
    PROVIDER_SPECIFIC_VARS = {
        'openai': ['OPENAI_API_KEY'],
        'google': ['GOOGLE_API_KEY']
    }

    # Modelos padrão por provider
    DEFAULT_MODELS = {
        'openai': {
            'main': 'gpt-4o-mini',
            'eval': 'gpt-4o'
        },
        'google': {
            'main': 'gemini-2.5-flash',
            'eval': 'gemini-2.5-flash'
        }
    }

    def __init__(self):
        """Inicializa e valida a configuração."""
        self._validate_required_vars()
        self._validate_provider_specific_vars()
        self._load_configuration()

    @staticmethod
    def _validate_required_vars() -> None:
        """
        Valida se todas as variáveis obrigatórias estão configuradas.

        Raises:
            ConfigError: Se alguma variável obrigatória estiver faltando
        """
        missing_vars = []

        for var in Config.REQUIRED_VARS:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = (
                "❌ Variáveis de ambiente obrigatórias não configuradas:\n"
            )
            for var in missing_vars:
                error_msg += f"   - {var}\n"
            error_msg += "\nConfigure-as no arquivo .env antes de continuar.\n"
            error_msg += "Veja .env.example para um template."
            raise ConfigError(error_msg)

    @staticmethod
    def _validate_provider_specific_vars() -> None:
        """
        Valida variáveis específicas do provider selecionado.

        Raises:
            ConfigError: Se variáveis do provider estiverem faltando
        """
        provider = os.getenv('LLM_PROVIDER', '').lower()

        if provider not in Config.PROVIDER_SPECIFIC_VARS:
            raise ConfigError(
                f"❌ Provider '{provider}' não suportado.\n"
                f"Providers disponíveis: openai, google\n"
                f"Configure LLM_PROVIDER no arquivo .env"
            )

        required_for_provider = Config.PROVIDER_SPECIFIC_VARS[provider]
        missing_vars = []

        for var in required_for_provider:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = (
                f"❌ Variáveis obrigatórias para provider '{provider}' "
                f"não configuradas:\n"
            )
            for var in missing_vars:
                error_msg += f"   - {var}\n"

            if provider == 'openai':
                error_msg += (
                    "\nObtenha uma chave em: "
                    "https://platform.openai.com/api-keys"
                )
            elif provider == 'google':
                error_msg += (
                    "\nObtenha uma chave em: "
                    "https://aistudio.google.com/app/apikey"
                )

            raise ConfigError(error_msg)

    def _load_configuration(self) -> None:
        """Carrega todas as configurações do ambiente."""
        self.langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        self.langsmith_endpoint = os.getenv(
            'LANGSMITH_ENDPOINT',
            'https://api.smith.langchain.com'
        )
        self.langsmith_project = os.getenv('LANGSMITH_PROJECT', 'default')
        self.username_langsmith_hub = os.getenv('USERNAME_LANGSMITH_HUB')

        self.llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        self.llm_model = os.getenv(
            'LLM_MODEL',
            self.DEFAULT_MODELS[self.llm_provider]['main']
        )
        self.eval_model = os.getenv(
            'EVAL_MODEL',
            self.DEFAULT_MODELS[self.llm_provider]['eval']
        )

        # Provider-specific keys
        if self.llm_provider == 'openai':
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        elif self.llm_provider == 'google':
            self.google_api_key = os.getenv('GOOGLE_API_KEY')

    def display_active_config(self) -> None:
        """Exibe a configuração ativa na inicialização."""
        print("\n" + "=" * 60)
        print("🔧 CONFIGURAÇÃO ATIVA")
        print("=" * 60)

        print(f"\n📍 LangSmith:")
        print(f"   Projeto: {self.langsmith_project}")
        print(f"   Endpoint: {self.langsmith_endpoint}")
        print(f"   Username Hub: {self.username_langsmith_hub}")

        print(f"\n🤖 LLM Configuration:")
        print(f"   Provider: {self.llm_provider.upper()}")
        print(f"   Modelo Principal: {self.llm_model}")
        print(f"   Modelo Avaliação: {self.eval_model}")

        print("\n" + "=" * 60 + "\n")

    def to_dict(self) -> dict:
        """
        Retorna configuração como dicionário.

        Returns:
            Dicionário com todas as configurações
        """
        return {
            'langsmith_api_key': self.langsmith_api_key,
            'langsmith_endpoint': self.langsmith_endpoint,
            'langsmith_project': self.langsmith_project,
            'username_langsmith_hub': self.username_langsmith_hub,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'eval_model': self.eval_model
        }


def load_config() -> Config:
    """
    Carrega e valida a configuração do projeto.

    Returns:
        Instância de Config com todas as variáveis validadas

    Raises:
        SystemExit: Se houver erro de configuração (código 1)
    """
    try:
        config = Config()
        config.display_active_config()
        return config
    except ConfigError as e:
        print(str(e))
        sys.exit(1)


# Instância global de configuração (lazy loading)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Obtém a instância global de configuração (singleton).

    Returns:
        Instância de Config

    Raises:
        SystemExit: Se houver erro de configuração (código 1)
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
