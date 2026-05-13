"""
Módulo de avaliação de prompts otimizados.

Este módulo implementa a orquestração completa de avaliação:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Executa prompts otimizados contra o dataset
3. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
4. Agrega resultados e determina status de aprovação
5. Formata e exibe resultados

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.

**Validates: Requirements 10.1, 10.2, 10.3, 10.8, 10.9, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10**
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from src.utils import check_env_vars, format_score, print_section_header, load_yaml
from src.llm_provider import get_llm, get_eval_llm
from src.metrics import (
    evaluate_f1_score, 
    evaluate_clarity, 
    evaluate_precision,
    calculate_helpfulness,
    calculate_correctness
)

load_dotenv()


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Carrega dataset de avaliação de arquivo JSONL.
    
    Cada linha do arquivo deve conter um objeto JSON com:
    - inputs: dicionário com "bug_report"
    - outputs: dicionário com "reference" (user story esperada)
    - metadata: dicionário com "domain", "type", "complexity"
    
    Args:
        jsonl_path: Caminho para o arquivo JSONL
    
    Returns:
        Lista de exemplos carregados do arquivo
    
    Raises:
        FileNotFoundError: Se arquivo não existir
        json.JSONDecodeError: Se linha não for JSON válido
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.8, 10.9**
    """
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"❌ Erro ao parsear linha {line_num}: {e}")
                        raise

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        raise
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        raise


def execute_prompt_on_dataset(
    prompt_template: ChatPromptTemplate,
    dataset: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Executa prompt otimizado contra todos os exemplos do dataset.
    
    Para cada exemplo no dataset:
    1. Extrai o bug_report do campo inputs
    2. Executa o prompt com o bug_report
    3. Coleta a saída gerada
    4. Compara com a saída esperada (reference)
    
    Args:
        prompt_template: ChatPromptTemplate carregado do YAML
        dataset: Lista de exemplos do dataset
    
    Returns:
        Lista de resultados com: input, expected_output, actual_output
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
    """
    results = []
    llm = get_llm(temperature=0)
    
    print(f"\n📋 Executando prompt em {len(dataset)} exemplos...")
    
    for i, example in enumerate(dataset, 1):
        try:
            # Extrair inputs e outputs
            inputs = example.get("inputs", {})
            outputs = example.get("outputs", {})
            metadata = example.get("metadata", {})
            
            bug_report = inputs.get("bug_report", "")
            reference_output = outputs.get("reference", "")
            
            if not bug_report:
                print(f"   ⚠️  Exemplo {i}: bug_report vazio, pulando")
                continue
            
            # Executar prompt
            chain = prompt_template | llm
            response = chain.invoke({"bug_report": bug_report})
            actual_output = response.content
            
            # Armazenar resultado
            result = {
                "example_index": i,
                "bug_report": bug_report,
                "expected_output": reference_output,
                "actual_output": actual_output,
                "metadata": metadata
            }
            results.append(result)
            
            print(f"   [{i}/{len(dataset)}] ✓ Executado")
            
        except Exception as e:
            print(f"   [{i}/{len(dataset)}] ❌ Erro: {e}")
            continue
    
    return results


def aggregate_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrega resultados de avaliação calculando todas as 5 métricas.
    
    Para cada resultado:
    1. Calcula F1-Score, Clarity, Precision usando LLM-as-Judge
    2. Calcula Helpfulness = (Clarity + Precision) / 2
    3. Calcula Correctness = (F1-Score + Precision) / 2
    
    Depois agrega:
    1. Calcula média de cada métrica entre todos os exemplos
    2. Determina pass/fail (todos >= 0.9?)
    3. Retorna estrutura com scores e reasoning
    
    Args:
        results: Lista de resultados de execute_prompt_on_dataset
    
    Returns:
        Dicionário com:
        - metrics: dicionário com scores agregados
        - individual_results: lista com scores de cada exemplo
        - passed: boolean indicando se todos >= 0.9
        - reasoning: explicação dos resultados
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10**
    """
    print(f"\n📊 Calculando 5 métricas para {len(results)} exemplos...")
    
    individual_results = []
    f1_scores = []
    clarity_scores = []
    precision_scores = []
    
    for i, result in enumerate(results, 1):
        try:
            bug_report = result["bug_report"]
            actual_output = result["actual_output"]
            expected_output = result["expected_output"]
            
            # Calcular 3 métricas base
            f1_result = evaluate_f1_score(bug_report, actual_output, expected_output)
            clarity_result = evaluate_clarity(bug_report, actual_output, expected_output)
            precision_result = evaluate_precision(bug_report, actual_output, expected_output)
            
            f1_score = f1_result["score"]
            clarity_score = clarity_result["score"]
            precision_score = precision_result["score"]
            
            # Calcular 2 métricas derivadas
            helpfulness = calculate_helpfulness(clarity_score, precision_score)
            correctness = calculate_correctness(f1_score, precision_score)
            
            # Armazenar scores individuais
            f1_scores.append(f1_score)
            clarity_scores.append(clarity_score)
            precision_scores.append(precision_score)
            
            individual_results.append({
                "example_index": i,
                "metrics": {
                    "helpfulness": helpfulness,
                    "correctness": correctness,
                    "f1_score": f1_score,
                    "clarity": clarity_score,
                    "precision": precision_score
                },
                "reasoning": {
                    "f1_score": f1_result.get("reasoning", ""),
                    "clarity": clarity_result.get("reasoning", ""),
                    "precision": precision_result.get("reasoning", "")
                }
            })
            print(f"   [{i}/{len(results)}] F1:{f1_score:.4f} Clarity:{clarity_score:.4f} Precision:{precision_score:.4f}")
            
        except Exception as e:
            print(f"   [{i}/{len(results)}] ❌ Erro ao calcular métricas: {e}")
            continue
    
    # Calcular médias
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    
    avg_helpfulness = calculate_helpfulness(avg_clarity, avg_precision)
    avg_correctness = calculate_correctness(avg_f1, avg_precision)
    
    # Determinar pass/fail
    all_metrics = [avg_helpfulness, avg_correctness, avg_f1, avg_clarity, avg_precision]
    passed = all(score >= 0.9 for score in all_metrics)
    
    return {
        "metrics": {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        },
        "individual_results": individual_results,
        "passed": passed,
        "total_examples": len(results)
    }


def format_evaluation_output(results: Dict[str, Any]) -> str:
    """
    Formata resultados de avaliação para exibição no terminal.
    
    Exibe:
    1. Todos os 5 metric scores com 4 casas decimais
    2. Indicadores pass/fail (✓/✗) para cada métrica
    3. Status geral: "✅ APPROVED" ou "❌ FAILED"
    4. Lista de métricas que falharam (se houver)
    5. Reasoning para cada métrica
    
    Args:
        results: Dicionário retornado por aggregate_evaluation_results
    
    Returns:
        String formatada para exibição
    
    **Validates: Requirements 5.8, 5.9, 5.10**
    """
    metrics = results["metrics"]
    passed = results["passed"]
    
    output = []
    output.append("\n" + "=" * 70)
    output.append("RESULTADOS DA AVALIAÇÃO")
    output.append("=" * 70)
    
    # Exibir scores com 4 casas decimais
    output.append("\n📊 Scores das Métricas:")
    output.append("-" * 70)
    
    for metric_name in ["helpfulness", "correctness", "f1_score", "clarity", "precision"]:
        score = metrics[metric_name]
        indicator = "✓" if score >= 0.9 else "✗"
        output.append(f"  {metric_name:15s}: {score:.4f} {indicator}")
    
    # Status geral
    output.append("\n" + "-" * 70)
    if passed:
        output.append("✅ APROVADO - Todas as métricas >= 0.9")
    else:
        output.append("❌ REPROVADO")
        failing_metrics = [name for name, score in metrics.items() if score < 0.9]
        output.append(f"\n⚠️  Métricas abaixo de 0.9:")
        for metric in failing_metrics:
            output.append(f"   - {metric}: {metrics[metric]:.4f}")
    
    output.append("-" * 70)
    
    return "\n".join(output)


def main():
    """
    Script de entrada para orquestração completa de avaliação.
    
    Fluxo:
    1. Valida configuração (.env)
    2. Carrega dataset de JSONL
    3. Carrega prompt otimizado de YAML
    4. Executa prompt em todos os exemplos
    5. Calcula 5 métricas
    6. Agrega resultados
    7. Formata e exibe resultados
    8. Retorna exit code 0 (sucesso) ou 1 (falha)
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10**
    """
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")
    
    # Validar configuração
    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")
    
    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")
    
    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")
    
    if not check_env_vars(required_vars):
        return 1
    
    # Carregar dataset
    jsonl_path = "datasets/bug_to_user_story.jsonl"
    
    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1
    
    try:
        dataset = load_dataset_from_jsonl(jsonl_path)
        print(f"✓ Dataset carregado: {len(dataset)} exemplos")
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return 1
    
    # Carregar prompt otimizado
    prompt_yaml_path = "prompts/bug_to_user_story_v2.yml"
    
    if not Path(prompt_yaml_path).exists():
        print(f"❌ Arquivo de prompt não encontrado: {prompt_yaml_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1
    
    try:
        prompt_data = load_yaml(prompt_yaml_path)
        if not prompt_data:
            print(f"❌ Erro ao carregar YAML: {prompt_yaml_path}")
            return 1
        
        # Extrair dados do prompt
        prompt_key = list(prompt_data.keys())[0]
        prompt_config = prompt_data[prompt_key]
        
        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt = prompt_config.get("user_prompt", "{bug_report}")
        
        # Criar ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        
        print(f"✓ Prompt carregado: {prompt_key}")
    except Exception as e:
        print(f"❌ Erro ao carregar prompt: {e}")
        return 1
    
    # Executar prompt no dataset
    try:
        results = execute_prompt_on_dataset(prompt_template, dataset)
        if not results:
            print("❌ Nenhum resultado foi gerado")
            return 1
        print(f"✓ Prompt executado em {len(results)} exemplos")
    except Exception as e:
        print(f"❌ Erro ao executar prompt: {e}")
        return 1
    
    # Agregar resultados
    try:
        aggregated = aggregate_evaluation_results(results)
        print(f"✓ Métricas calculadas")
    except Exception as e:
        print(f"❌ Erro ao calcular métricas: {e}")
        return 1
    
    # Formatar e exibir resultados
    output = format_evaluation_output(aggregated)
    print(output)
    
    # Retornar exit code apropriado
    if aggregated["passed"]:
        print("\n✅ Todos os prompts atingiram todas as métricas >= 0.9!")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("\n⚠️  Alguns prompts não atingiram todas as métricas >= 0.9")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1


if __name__ == "__main__":
    sys.exit(main())
