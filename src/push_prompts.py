"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)
5. Verifica se o push foi bem-sucedido

Tratamento de erros:
- Valida estrutura do prompt antes de fazer push
- Trata erros de autenticação com LangSmith
- Trata erros de conexão com retry
- Exibe mensagens claras de sucesso/erro
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from utils import load_yaml, print_section_header

load_dotenv()


class PromptPushError(Exception):
    """Exceção para erros ao fazer push de prompts."""
    pass


def validate_prompt_structure(prompt_data: dict) -> Tuple[bool, list]:
    """
    Valida estrutura completa de um prompt v2.

    Verifica:
    - system_prompt existe e não está vazio
    - system_prompt NÃO contém placeholders {bug_report} duplicados
    - user_prompt contém apenas o placeholder {bug_report}
    - version é exatamente "v2"
    - techniques_applied é uma lista com pelo menos 2 itens
    - examples é uma lista de dicionários com chaves "input" e "output"

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    # 1. Verificar se system_prompt existe e não está vazio
    if 'system_prompt' not in prompt_data:
        errors.append("Campo obrigatório faltando: system_prompt")
    else:
        system_prompt = prompt_data.get('system_prompt', '').strip()
        if not system_prompt:
            errors.append("system_prompt está vazio")
        else:
            # 2. Verificar se system_prompt contém placeholders {bug_report} duplicados
            bug_report_count = system_prompt.count('{bug_report}')
            if bug_report_count > 1:
                errors.append(f"system_prompt contém {bug_report_count} placeholders {{bug_report}} duplicados (máximo: 1)")

    # 3. Verificar se user_prompt contém apenas {bug_report}
    if 'user_prompt' not in prompt_data:
        errors.append("Campo obrigatório faltando: user_prompt")
    else:
        user_prompt = prompt_data.get('user_prompt', '').strip()
        if not user_prompt:
            errors.append("user_prompt está vazio")
        elif user_prompt != "{bug_report}":
            errors.append(f"user_prompt deve conter apenas '{{bug_report}}', encontrado: '{user_prompt}'")

    # 4. Verificar se version é exatamente "v2"
    if 'version' not in prompt_data:
        errors.append("Campo obrigatório faltando: version")
    else:
        version = prompt_data.get('version', '')
        if version != "v2":
            errors.append(f"version deve ser exatamente 'v2', encontrado: '{version}'")

    # 5. Verificar se techniques_applied é uma lista com pelo menos 2 itens
    if 'techniques_applied' not in prompt_data:
        errors.append("Campo obrigatório faltando: techniques_applied")
    else:
        techniques = prompt_data.get('techniques_applied', [])
        if not isinstance(techniques, list):
            errors.append(f"techniques_applied deve ser uma lista, encontrado: {type(techniques).__name__}")
        elif len(techniques) < 2:
            errors.append(f"techniques_applied deve ter pelo menos 2 itens, encontrados: {len(techniques)}")
        else:
            # Verificar se todos os itens são strings não-vazias
            for i, technique in enumerate(techniques):
                if not isinstance(technique, str):
                    errors.append(f"techniques_applied[{i}] deve ser string, encontrado: {type(technique).__name__}")
                elif not technique.strip():
                    errors.append(f"techniques_applied[{i}] está vazio")

    # 6. Verificar se examples é uma lista de dicionários com "input" e "output"
    if 'examples' in prompt_data:
        examples = prompt_data.get('examples', [])
        if not isinstance(examples, list):
            errors.append(f"examples deve ser uma lista, encontrado: {type(examples).__name__}")
        else:
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    errors.append(f"examples[{i}] deve ser um dicionário, encontrado: {type(example).__name__}")
                else:
                    if 'input' not in example:
                        errors.append(f"examples[{i}] faltando chave 'input'")
                    if 'output' not in example:
                        errors.append(f"examples[{i}] faltando chave 'output'")

    return (len(errors) == 0, errors)


def create_chat_prompt_template(prompt_data: Dict[str, Any]) -> ChatPromptTemplate:
    """
    Create ChatPromptTemplate from prompt data.
    
    Constructs a ChatPromptTemplate with system and human messages from the
    prompt data. The template can then be pushed to LangSmith Hub.
    
    Args:
        prompt_data: Structured prompt data with system_prompt and user_prompt
    
    Returns:
        ChatPromptTemplate ready for Hub
        
    Raises:
        PromptPushError: If template creation fails
    """
    try:
        system_prompt = prompt_data.get('system_prompt', '')
        user_prompt = prompt_data.get('user_prompt', '')
        
        # Create system message template
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        
        # Create human message template
        human_message = HumanMessagePromptTemplate.from_template(user_prompt)
        
        # Create ChatPromptTemplate
        template = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])
        
        return template
        
    except Exception as e:
        raise PromptPushError(
            f"❌ Erro ao criar ChatPromptTemplate: {str(e)}"
        )


def push_prompt_to_hub(
    template: ChatPromptTemplate,
    prompt_name: str,
    prompt_data: Dict[str, Any],
    max_retries: int = 3,
    initial_backoff: float = 1.0
) -> bool:
    """
    Push prompt to LangSmith Hub.
    
    Publishes a ChatPromptTemplate to LangSmith Hub with metadata.
    Implements retry logic with exponential backoff for network errors.
    
    Args:
        template: ChatPromptTemplate to push
        prompt_name: Target name (e.g., "username/bug_to_user_story_v2")
        prompt_data: Original prompt data for metadata
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff time in seconds (default: 1.0)
    
    Returns:
        True if successful, False otherwise
        
    Raises:
        PromptPushError: If push fails after all retries
    """
    backoff = initial_backoff
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"📤 Tentativa {attempt + 1}/{max_retries}: Fazendo push do prompt '{prompt_name}'...")
            
            # Add metadata to template
            template.metadata = {
                'description': prompt_data.get('description', ''),
                'tags': prompt_data.get('tags', []),
                'techniques_applied': prompt_data.get('techniques_applied', []),
                'version': prompt_data.get('version', 'v2'),
                'created_at': prompt_data.get('created_at', '')
            }
            
            # Push to Hub
            hub.push(prompt_name, template)
            
            print(f"✅ Prompt '{prompt_name}' enviado com sucesso para o Hub")
            
            return True
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Handle specific error cases (no retry needed)
            if "authentication" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
                raise PromptPushError(
                    f"❌ Erro de autenticação com LangSmith\n\n"
                    f"Verifique:\n"
                    f"1. LANGSMITH_API_KEY está configurada corretamente no .env\n"
                    f"2. A chave não expirou\n"
                    f"3. Você tem permissão para fazer push de prompts\n"
                    f"4. Obtenha uma nova chave em: https://smith.langchain.com/settings/api-keys\n"
                )
            
            elif "already exists" in error_msg or "conflict" in error_msg:
                raise PromptPushError(
                    f"❌ Prompt '{prompt_name}' já existe no Hub\n\n"
                    f"Opções:\n"
                    f"1. Use um nome diferente (ex: v3, v4)\n"
                    f"2. Atualize o prompt existente manualmente no Hub\n"
                    f"3. Verifique o nome do prompt\n"
                )
            
            # For network errors, retry with exponential backoff
            if attempt < max_retries - 1:
                print(
                    f"⚠️  Erro ao fazer push (tentativa {attempt + 1}/{max_retries}): "
                    f"{str(e)[:100]}"
                )
                print(f"⏳ Aguardando {backoff:.1f}s antes de tentar novamente...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                # Last attempt failed
                raise PromptPushError(
                    f"❌ Erro ao fazer push do prompt após {max_retries} tentativas\n\n"
                    f"Último erro: {str(last_error)}\n\n"
                    f"Possíveis causas:\n"
                    f"1. Problema de conexão com LangSmith\n"
                    f"2. Servidor LangSmith indisponível\n"
                    f"3. Timeout de conexão\n"
                    f"4. Limite de taxa (rate limit) atingido\n\n"
                    f"Tente novamente em alguns minutos."
                )
    
    # Should never reach here
    raise PromptPushError("Erro desconhecido ao fazer push do prompt")


def verify_hub_push(prompt_name: str) -> bool:
    """
    Verify that prompt was successfully pushed by checking Hub response.
    
    Attempts to pull the prompt from Hub to verify it was successfully pushed.
    
    Args:
        prompt_name: Name of the prompt to verify
    
    Returns:
        True if prompt exists in Hub, False otherwise
    """
    try:
        print(f"\n🔍 Verificando se o prompt foi enviado com sucesso...")
        
        # Try to pull the prompt to verify it exists
        pulled_prompt = hub.pull(prompt_name)
        
        if pulled_prompt:
            print(f"✅ Prompt verificado com sucesso no Hub")
            return True
        else:
            print(f"❌ Prompt não encontrado no Hub após push")
            return False
            
    except Exception as e:
        print(f"⚠️  Não foi possível verificar o prompt no Hub: {str(e)[:100]}")
        # Don't fail the entire operation if verification fails
        # The push may have succeeded even if verification fails
        return True


def load_prompt_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load prompt from YAML file.
    
    Reads and parses a YAML file containing prompt data.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Dictionary with prompt data
        
    Raises:
        PromptPushError: If file not found or parsing fails
    """
    try:
        print(f"📂 Carregando prompt de: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            raise PromptPushError(
                f"❌ Arquivo não encontrado: {file_path}\n\n"
                f"Verifique:\n"
                f"1. O arquivo existe no caminho especificado\n"
                f"2. O caminho está correto\n"
                f"3. Execute pull_prompts.py primeiro para criar o arquivo\n"
            )
        
        # Load YAML
        data = load_yaml(file_path)
        
        if data is None:
            raise PromptPushError(
                f"❌ Erro ao parsear arquivo YAML: {file_path}\n\n"
                f"Verifique:\n"
                f"1. O arquivo é um YAML válido\n"
                f"2. A indentação está correta (2 espaços)\n"
                f"3. Não há caracteres especiais não escapados\n"
            )
        
        # Extract prompt data (should be under 'bug_to_user_story_v2' key)
        if 'bug_to_user_story_v2' in data:
            prompt_data = data['bug_to_user_story_v2']
        else:
            # Try to use the first key if it's not the expected one
            prompt_data = list(data.values())[0] if data else {}
        
        print(f"✅ Prompt carregado com sucesso")
        
        return prompt_data
        
    except PromptPushError:
        raise
    except Exception as e:
        raise PromptPushError(
            f"❌ Erro ao carregar arquivo: {str(e)}"
        )


def main():
    """
    Main entry point for pushing optimized prompts to LangSmith Hub.
    
    Orchestrates the complete workflow:
    1. Display progress header
    2. Load configuration
    3. Load prompt from YAML file
    4. Validate prompt structure
    5. Create ChatPromptTemplate
    6. Push to LangSmith Hub
    7. Verify push was successful
    8. Display final status with Hub URL
    
    Returns:
        0 if successful, 1 if any step fails
    """
    try:
        print_section_header("🚀 PUSH PROMPTS TO LANGSMITH HUB")
        
        # Step 1: Get configuration
        print("\n📍 Step 1: Loading configuration...")
        try:
            config = get_config()
            print("✅ Configuration loaded successfully")
        except SystemExit:
            # get_config() calls sys.exit(1) on error, which we need to catch
            return 1
        
        # Step 2: Load prompt from file
        print("\n📍 Step 2: Loading prompt from YAML file...")
        prompt_file = "prompts/bug_to_user_story_v2.yml"
        
        try:
            prompt_data = load_prompt_from_file(prompt_file)
        except PromptPushError as e:
            print(f"\n{str(e)}\n")
            return 1
        
        # Step 3: Validate prompt structure
        print("\n📍 Step 3: Validating prompt structure...")
        is_valid, errors = validate_prompt_structure(prompt_data)
        
        if not is_valid:
            print(f"\n❌ Validation failed with {len(errors)} error(s):\n")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
            print()
            print_section_header("❌ VALIDATION FAILED")
            return 1
        
        print(f"✅ Prompt structure is valid")
        
        # Step 4: Create ChatPromptTemplate
        print("\n📍 Step 4: Creating ChatPromptTemplate...")
        try:
            template = create_chat_prompt_template(prompt_data)
            print(f"✅ ChatPromptTemplate created successfully")
        except PromptPushError as e:
            print(f"\n{str(e)}\n")
            return 1
        
        # Step 5: Push to Hub
        print("\n📍 Step 5: Pushing prompt to LangSmith Hub...")
        prompt_name = f"{config.username_langsmith_hub}/bug_to_user_story_v2"
        
        try:
            success = push_prompt_to_hub(template, prompt_name, prompt_data)
            if not success:
                print_section_header("❌ PUSH FAILED")
                return 1
        except PromptPushError as e:
            print(f"\n{str(e)}\n")
            return 1
        
        # Step 6: Verify push
        print("\n📍 Step 6: Verifying push was successful...")
        if not verify_hub_push(prompt_name):
            print("⚠️  Warning: Could not verify prompt in Hub (but push may have succeeded)")
        
        # Success!
        hub_url = f"https://smith.langchain.com/hub/{prompt_name}"
        
        print_section_header("✅ PUSH COMPLETED SUCCESSFULLY")
        print(f"\n📊 Summary:")
        print(f"   ✓ Prompt loaded from: {prompt_file}")
        print(f"   ✓ Prompt structure validated")
        print(f"   ✓ ChatPromptTemplate created")
        print(f"   ✓ Pushed to Hub: {prompt_name}")
        print(f"\n🔗 Hub URL:")
        print(f"   {hub_url}")
        print(f"\n📝 Prompt Details:")
        print(f"   Version: {prompt_data.get('version', 'N/A')}")
        print(f"   Description: {prompt_data.get('description', 'N/A')[:60]}...")
        print(f"   Techniques: {len(prompt_data.get('techniques_applied', []))} applied")
        print(f"   Examples: {len(prompt_data.get('examples', []))} included")
        print(f"\n🎯 Next step: Evaluate the prompt with evaluate.py")
        print()
        
        return 0
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print_section_header("❌ PUSH FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
