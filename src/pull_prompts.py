"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

Tratamento de erros:
- Valida LANGSMITH_API_KEY
- Trata prompts não encontrados
- Implementa retry com backoff exponencial para erros de rede
- Exibe mensagens claras de sucesso/erro
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from utils import save_yaml, print_section_header

load_dotenv()


class PromptPullError(Exception):
    """Exceção para erros ao fazer pull de prompts."""
    pass


def pull_prompt_from_hub(
    prompt_name: str,
    max_retries: int = 3,
    initial_backoff: float = 1.0
) -> Dict[str, Any]:
    """
    Pull prompt from LangSmith Hub and return structured data.
    
    Retrieves a prompt from LangSmith Hub using the provided prompt name,
    extracts the system_prompt, user_prompt, and metadata, and returns
    them in a structured dictionary format.
    
    Implements retry logic with exponential backoff for network errors.
    
    Args:
        prompt_name: Full prompt name (e.g., "leonanluppi/bug_to_user_story_v1")
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff time in seconds (default: 1.0)
    
    Returns:
        Dictionary with keys:
        - description: Prompt description
        - system_prompt: System prompt text
        - user_prompt: User prompt template
        - metadata: Additional metadata from the prompt
    
    Raises:
        PromptPullError: If prompt not found, authentication fails, or network errors persist
    """
    backoff = initial_backoff
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"📥 Tentativa {attempt + 1}/{max_retries}: Fazendo pull do prompt '{prompt_name}'...")
            
            # Pull the prompt from Hub
            prompt = hub.pull(prompt_name)
            
            # Extract prompt data based on type
            prompt_data = _extract_prompt_data(prompt)
            
            print(f"✅ Prompt '{prompt_name}' obtido com sucesso")
            
            return prompt_data
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Handle specific error cases (no retry needed)
            if "not found" in error_msg or "404" in error_msg:
                raise PromptPullError(
                    f"❌ Prompt '{prompt_name}' não encontrado no Hub\n\n"
                    f"Para verificar:\n"
                    f"1. Acesse: https://smith.langchain.com/hub\n"
                    f"2. Procure por: {prompt_name}\n"
                    f"3. Verifique se o nome está correto\n"
                    f"4. Verifique se o prompt é público\n"
                    f"5. Verifique se você tem permissão para acessar\n"
                )
            
            elif "authentication" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
                raise PromptPullError(
                    f"❌ Erro de autenticação com LangSmith\n\n"
                    f"Verifique:\n"
                    f"1. LANGSMITH_API_KEY está configurada corretamente no .env\n"
                    f"2. A chave não expirou\n"
                    f"3. Você tem permissão para acessar este prompt\n"
                    f"4. Obtenha uma nova chave em: https://smith.langchain.com/settings/api-keys\n"
                )
            
            # For network errors, retry with exponential backoff
            if attempt < max_retries - 1:
                print(
                    f"⚠️  Erro ao fazer pull (tentativa {attempt + 1}/{max_retries}): "
                    f"{str(e)[:100]}"
                )
                print(f"⏳ Aguardando {backoff:.1f}s antes de tentar novamente...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                # Last attempt failed
                raise PromptPullError(
                    f"❌ Erro ao fazer pull do prompt após {max_retries} tentativas\n\n"
                    f"Último erro: {str(last_error)}\n\n"
                    f"Possíveis causas:\n"
                    f"1. Problema de conexão com LangSmith\n"
                    f"2. Prompt não existe ou foi deletado\n"
                    f"3. Você não tem permissão para acessar\n"
                    f"4. Servidor LangSmith indisponível\n"
                    f"5. Timeout de conexão\n\n"
                    f"Tente novamente em alguns minutos."
                )

    # Should never reach here
    raise PromptPullError("Erro desconhecido ao fazer pull do prompt")


def _extract_prompt_data(prompt: Any) -> Dict[str, Any]:
    """
    Extract structured data from a LangChain prompt template.
    
    Supports both ChatPromptTemplate and PromptTemplate formats.
    Extracts system_prompt, user_prompt, and metadata.
    
    Args:
        prompt: LangChain prompt template (ChatPromptTemplate or PromptTemplate)
    
    Returns:
        Dictionary with extracted prompt data
    """
    prompt_data = {
        'description': 'Prompt pulled from LangSmith Hub',
        'system_prompt': '',
        'user_prompt': '',
        'metadata': {}
    }
    
    # Handle ChatPromptTemplate
    if isinstance(prompt, ChatPromptTemplate):
        # Extract messages from ChatPromptTemplate
        messages = prompt.messages
        
        for message in messages:
            # Check if it's a system message
            if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                template = message.prompt.template
                
                # Determine if it's system or user prompt based on message type
                message_type = type(message).__name__
                
                if 'System' in message_type:
                    prompt_data['system_prompt'] = template
                elif 'Human' in message_type or 'User' in message_type:
                    prompt_data['user_prompt'] = template
            elif hasattr(message, 'content'):
                # Handle string content messages
                content = message.content
                message_type = type(message).__name__
                
                if 'System' in message_type:
                    prompt_data['system_prompt'] = content
                elif 'Human' in message_type or 'User' in message_type:
                    prompt_data['user_prompt'] = content
    
    # Handle PromptTemplate
    elif isinstance(prompt, PromptTemplate):
        # For PromptTemplate, the entire template is the user prompt
        if hasattr(prompt, 'template'):
            prompt_data['user_prompt'] = prompt.template
        else:
            prompt_data['user_prompt'] = str(prompt)
    
    # Fallback: try to extract from string representation
    if not prompt_data['system_prompt'] and not prompt_data['user_prompt']:
        prompt_str = str(prompt)
        prompt_data['user_prompt'] = prompt_str
    
    # Extract metadata if available
    if hasattr(prompt, 'metadata'):
        prompt_data['metadata'] = prompt.metadata
    
    return prompt_data


def serialize_prompt_to_yaml(prompt_data: Dict[str, Any], output_path: str) -> bool:
    """
    Serialize prompt structure to YAML file.
    
    Saves the prompt data to a YAML file with proper formatting and structure.
    Creates parent directories if they don't exist.
    
    Includes all required fields:
    - description: Prompt description
    - system_prompt: System prompt text
    - user_prompt: User prompt template
    - metadata: Additional metadata
    - version: Version string (default: "v1")
    - tags: List of tags (default: empty list)
    
    Args:
        prompt_data: Structured prompt data with description, system_prompt, user_prompt
        output_path: Path to save YAML file (e.g., "prompts/bug_to_user_story_v1.yml")
    
    Returns:
        True if successful, False otherwise
        
    Raises:
        PromptPullError: If there's an error saving the file
    """
    try:
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current date for created_at field
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare YAML structure with all required fields
        yaml_data = {
            'bug_to_user_story_v1': {
                'description': prompt_data.get('description', 'Prompt from LangSmith Hub'),
                'system_prompt': prompt_data.get('system_prompt', ''),
                'user_prompt': prompt_data.get('user_prompt', ''),
                'version': prompt_data.get('version', 'v1'),
                'created_at': prompt_data.get('created_at', current_date),
                'tags': prompt_data.get('tags', ['bug-analysis', 'user-story', 'product-management']),
                'metadata': prompt_data.get('metadata', {})
            }
        }
        
        # Save to YAML
        success = save_yaml(yaml_data, output_path)
        
        if not success:
            raise PromptPullError(
                f"❌ Erro ao salvar arquivo YAML: {output_path}\n"
                f"Verifique permissões de escrita no diretório."
            )
        
        return True
        
    except PromptPullError:
        raise
    except Exception as e:
        raise PromptPullError(
            f"❌ Erro ao serializar prompt para YAML: {str(e)}"
        )


def verify_yaml_file(file_path: str) -> bool:
    """
    Verify that saved YAML file is valid and parseable.
    
    Checks that:
    - File exists and can be read
    - YAML is valid and parseable
    - Required structure is present
    - Required fields are not empty
    
    Args:
        file_path: Path to YAML file to verify
    
    Returns:
        True if file is valid and parseable, False otherwise
    """
    try:
        from utils import load_yaml
        
        print(f"\n🔍 Verifying YAML file validity...")
        
        # Try to load the YAML file
        data = load_yaml(file_path)
        
        if data is None:
            print(f"❌ Failed to parse YAML file: {file_path}")
            return False
        
        # Check for required structure
        if 'bug_to_user_story_v1' not in data:
            print(f"❌ Invalid YAML structure: 'bug_to_user_story_v1' key not found")
            return False
        
        prompt_data = data['bug_to_user_story_v1']
        
        # Check for required fields
        required_fields = ['description', 'system_prompt', 'user_prompt', 'version']
        missing_fields = [f for f in required_fields if f not in prompt_data]
        
        if missing_fields:
            print(f"❌ Missing required fields: {', '.join(missing_fields)}")
            return False
        
        # Check that system_prompt is not empty
        if not prompt_data.get('system_prompt', '').strip():
            print(f"❌ Field 'system_prompt' is empty")
            return False
        
        # Check that user_prompt is not empty
        if not prompt_data.get('user_prompt', '').strip():
            print(f"❌ Field 'user_prompt' is empty")
            return False
        
        print(f"✅ YAML file is valid and parseable")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying YAML file: {e}")
        return False


def main():
    """
    Main entry point for pulling and serializing prompts from LangSmith Hub.
    
    Orchestrates the complete workflow:
    1. Display progress header
    2. Load configuration
    3. Pull prompt from LangSmith Hub
    4. Serialize prompt to YAML file
    5. Verify saved YAML file is valid and parseable
    6. Display final status
    
    Returns:
        0 if successful, 1 if any step fails
    """
    try:
        print_section_header("🚀 PULL PROMPTS FROM LANGSMITH HUB")
        
        # Step 1: Get configuration
        print("\n📍 Step 1: Loading configuration...")
        try:
            config = get_config()
            print("✅ Configuration loaded successfully")
        except SystemExit:
            # get_config() calls sys.exit(1) on error, which we need to catch
            return 1
        
        # Step 2: Pull prompt from Hub
        print("\n📍 Step 2: Pulling prompt from LangSmith Hub...")
        prompt_name = f"{config.username_langsmith_hub}/bug_to_user_story_v1"
        
        try:
            prompt_data = pull_prompt_from_hub(prompt_name)
        except PromptPullError as e:
            print(f"\n{str(e)}\n")
            return 1
        
        # Step 3: Serialize to YAML
        print("\n📍 Step 3: Serializing prompt to YAML...")
        output_path = "prompts/bug_to_user_story_v1.yml"
        
        try:
            success = serialize_prompt_to_yaml(prompt_data, output_path)
            if not success:
                print_section_header("❌ SERIALIZATION FAILED")
                return 1
        except PromptPullError as e:
            print(f"\n{str(e)}\n")
            return 1
        
        # Step 4: Verify YAML file
        print("\n📍 Step 4: Verifying saved YAML file...")
        if not verify_yaml_file(output_path):
            print_section_header("❌ VERIFICATION FAILED")
            return 1
        
        # Success!
        print_section_header("✅ PULL COMPLETED SUCCESSFULLY")
        print(f"\n📊 Summary:")
        print(f"   ✓ Prompt pulled from Hub: {prompt_name}")
        print(f"   ✓ Serialized to YAML: {output_path}")
        print(f"   ✓ YAML file verified and valid")
        print(f"\n🎯 Next step: Optimize the prompt with advanced techniques (v2)")
        print()
        
        return 0
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print_section_header("❌ PULL FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
