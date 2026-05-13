import pytest
import yaml
from pathlib import Path


@pytest.fixture
def prompt_data():
    """Load prompt from bug_to_user_story_v2.yml in test setup."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    prompt_config = data.get('bug_to_user_story_v2', data)
    return prompt_config


def test_prompt_has_system_prompt(prompt_data):
    """Verify system_prompt field exists and is not empty. Validates: Requirements 9.2"""
    assert 'system_prompt' in prompt_data
    system_prompt = prompt_data['system_prompt']
    assert isinstance(system_prompt, str)
    assert len(system_prompt) > 0


def test_prompt_has_role_definition(prompt_data):
    """Verify prompt defines a specific persona. Validates: Requirements 9.3"""
    system_prompt = prompt_data['system_prompt']
    has_role_marker = "You are" in system_prompt or "You are an" in system_prompt
    assert has_role_marker
    specific_personas = ["Product Manager", "expert", "specialist", "engineer", "architect", "developer", "analyst"]
    has_specific_persona = any(persona.lower() in system_prompt.lower() for persona in specific_personas)
    assert has_specific_persona


def test_prompt_mentions_format(prompt_data):
    """Verify prompt specifies output format. Validates: Requirements 9.4"""
    system_prompt = prompt_data['system_prompt']
    format_keywords = ["format", "structure", "Markdown", "OUTPUT FORMAT"]
    has_format_spec = any(keyword in system_prompt for keyword in format_keywords)
    assert has_format_spec
    user_story_keywords = ["User Story", "As a", "user story"]
    has_user_story_format = any(keyword in system_prompt for keyword in user_story_keywords)
    assert has_user_story_format


def test_prompt_has_few_shot_examples(prompt_data):
    """Verify prompt contains at least 3 examples. Validates: Requirements 9.5"""
    assert 'examples' in prompt_data
    examples = prompt_data['examples']
    assert isinstance(examples, list)
    assert len(examples) >= 3
    for idx, example in enumerate(examples):
        assert isinstance(example, dict)
        assert 'input' in example
        assert 'output' in example
        assert len(example['input']) > 0
        assert len(example['output']) > 0


def test_prompt_no_todos(prompt_data):
    """Verify no [TODO] or TODO markers remain in prompt. Validates: Requirements 9.6"""
    system_prompt = prompt_data['system_prompt']
    assert "[TODO]" not in system_prompt
    assert "TODO" not in system_prompt
    assert "[FIXME]" not in system_prompt


def test_minimum_techniques(prompt_data):
    """Verify at least 2 techniques listed in techniques_applied. Validates: Requirements 9.7"""
    assert 'techniques_applied' in prompt_data
    techniques = prompt_data['techniques_applied']
    assert isinstance(techniques, list)
    assert len(techniques) >= 2
    for idx, technique in enumerate(techniques):
        assert isinstance(technique, str)
        assert len(technique) > 0
