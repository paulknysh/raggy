from langchain_core.documents import Document

from raggy import llm_filter


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.response


def _docs(*contents):
    return [Document(page_content=c) for c in contents]


def test_keeps_only_yes_chunks_in_original_order():
    llm = FakeLLM('{"0": "yes", "1": "no", "2": "yes"}')
    docs = _docs("A", "B", "C")

    kept = llm_filter.filter_docs_by_relevance("q", docs, llm)

    assert [d.page_content for d in kept] == ["A", "C"]
    assert llm.calls == 1


def test_all_no_returns_empty_context():
    llm = FakeLLM('{"0": "no", "1": "no"}')

    kept = llm_filter.filter_docs_by_relevance("q", _docs("A", "B"), llm)

    assert kept == []


def test_unparseable_response_is_fail_open():
    llm = FakeLLM("definitely not json")
    docs = _docs("A", "B", "C")

    kept = llm_filter.filter_docs_by_relevance("q", docs, llm)

    assert kept == docs


def test_loose_index_verdict_format_is_accepted():
    llm = FakeLLM("0: yes\n1: no\n2: yes")

    kept = llm_filter.filter_docs_by_relevance("q", _docs("A", "B", "C"), llm)

    assert [d.page_content for d in kept] == ["A", "C"]


def test_skips_llm_call_for_single_chunk():
    llm = FakeLLM("{broken")

    kept = llm_filter.filter_docs_by_relevance("q", _docs("A"), llm)

    assert [d.page_content for d in kept] == ["A"]
    assert llm.calls == 0
