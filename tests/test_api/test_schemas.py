import pytest
from pydantic import ValidationError

from src.api.v1.schemas import ExecuteRequest, FetchRequest


def test_execute_request_valid():
    req = ExecuteRequest(model_id="gpt2", input="hello", params={"max_length": 50})
    assert req.model_id == "gpt2"
    assert req.force_reload is False


def test_execute_request_missing_model_id():
    with pytest.raises(ValidationError):
        ExecuteRequest(input="hello")


def test_execute_request_missing_input():
    with pytest.raises(ValidationError):
        ExecuteRequest(model_id="gpt2")


def test_fetch_request_valid():
    req = FetchRequest(model_id="gpt2")
    assert req.model_id == "gpt2"


def test_execute_request_defaults():
    req = ExecuteRequest(model_id="gpt2", input="test")
    assert req.params == {}
    assert req.force_reload is False
