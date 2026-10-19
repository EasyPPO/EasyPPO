"""The HTTP judge client: an empty program is a compile error, not a failed episode."""

from agentic_critic import fcs_judge_http


def test_empty_program_scores_zero_without_calling_the_server(monkeypatch):
    def no_call(*args, **kwargs):
        raise AssertionError("the server must not be called for an empty program")

    monkeypatch.setattr(fcs_judge_http, "_call", no_call)
    for code in ("", "  \n\t"):
        result = fcs_judge_http.judge("http://127.0.0.1:1", "p", code)
        assert result.score == 0.0 and result.status == "Compile Error"
