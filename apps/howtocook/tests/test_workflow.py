import pytest
from howtocook.workflow import get_llm, get_model, normalize_recipe_tags


def test_normalize_recipe_tags_filters_unknown_and_duplicate_values() -> None:
    recipe = {
        "tags": {
            "cuisines": ["川菜", "川菜", "不存在"],
            "flavors": ["麻辣", 1],
            "scenes": "晚餐",
        }
    }

    assert normalize_recipe_tags(recipe)["tags"] == {
        "cuisines": ["川菜"],
        "flavors": ["麻辣"],
        "scenes": [],
    }


def test_normalize_recipe_tags_supplies_empty_groups() -> None:
    assert normalize_recipe_tags({})["tags"] == {
        "cuisines": [],
        "flavors": [],
        "scenes": [],
    }


def test_get_llm_requires_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY required"):
        get_llm("moonshotai/kimi-k2.6")


def test_get_llm_uses_openrouter_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")

    llm = get_llm("moonshotai/kimi-k2.6")

    assert llm.model_name == "moonshotai/kimi-k2.6"
    assert str(llm.openai_api_base) == "https://openrouter.example/v1"


def test_bailian_defaults_and_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bailian")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert get_model("process") == "qwen3.6-plus"
    assert get_model("refine") == "qwen3.7-max"
    assert str(get_llm(get_model("process")).openai_api_base) == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_custom_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.setenv("CUSTOM_API_KEY", "test-key")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://custom.example/v1")

    assert get_model("process") == "qwen3.7-max"
    assert get_model("refine") == "qwen3.7-max"
    assert str(get_llm("qwen3.7-max").openai_api_base) == "https://custom.example/v1"
