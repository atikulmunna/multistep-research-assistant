from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_model_routing_end_to_end(monkeypatch):
    called_models = []

    def fake_post(url, headers=None, json=None, timeout=0):
        model = json["model"]
        prompt = json["messages"][0]["content"]
        called_models.append(model)

        if "Classify the research query" in prompt:
            content = '{"query_type":"exploratory","scope":"broad"}'
        elif "Break this query into" in prompt:
            content = (
                '{"sub_questions":['
                '{"question":"Q1 impact","priority":1,"answered":false,"parent_question":null},'
                '{"question":"Q2 risks","priority":2,"answered":false,"parent_question":null},'
                '{"question":"Q3 outlook","priority":3,"answered":false,"parent_question":null}'
                ']}'
            )
        elif "Extract key factual claims" in prompt:
            subq = prompt.split("Sub-question: ", 1)[1].split("\n", 1)[0]
            content = (
                '{"key_information":["'
                + subq
                + ' shows strong evidence"],'
                '"citations":["https://example.com/source"],'
                '"contradictions":[]}'
            )
        elif "Create a short executive summary" in prompt:
            content = "Executive summary."
        else:
            content = "Section body."

        return _FakeResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)

    settings = Settings(
        llm_provider="openrouter",
        llm_model="openai/gpt-oss-120b:free",
        llm_model_planning="openai/gpt-oss-20b:free",
        llm_model_analysis="deepseek/deepseek-r1-0528:free",
        llm_model_writing="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test-key",
        search_provider="mock",
        max_research_iterations=3,
        min_relevance_score=0.8,
    )

    assistant = ResearchAssistant(settings=settings)
    result = assistant.research("Impact of AI on education")

    assert result["final_report"]
    assert "openai/gpt-oss-20b:free" in called_models
    assert "deepseek/deepseek-r1-0528:free" in called_models
    assert "openai/gpt-oss-120b:free" in called_models

