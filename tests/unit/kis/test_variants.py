from types import SimpleNamespace

from hcmai.kis.variants import ControlledQueryExpander


class Provider:
    def suggest(self, request):
        del request
        queries = [
            "Nam sees 2 red cars and does not run",
            "Nam sees 3 red cars and does not run",
            "Nam sees 2 blue cars and does not run",
            "Nam sees 2 red cars and runs",
            "Nam sees 2 red cars and does not run from another angle",
        ]
        return SimpleNamespace(suggestions=[
            SimpleNamespace(query=query, focus="literal") for query in queries
        ])


def test_original_is_retained_and_hard_constraint_changes_are_rejected():
    plan = ControlledQueryExpander(Provider()).expand(
        "Nam sees 2 red cars and does not run"
    )

    assert plan.variants[0].kind == "original"
    assert plan.variants[0].weight == 1.0
    assert [item.query for item in plan.variants[1:]] == [
        "Nam sees 2 red cars and does not run from another angle"
    ]
    assert plan.variants[1].weight < plan.variants[0].weight
    assert plan.warnings == (
        "query expansion rejected 4 unfaithful variant(s)",
    )


def test_provider_failure_falls_back_to_original_query():
    class FailedProvider:
        def suggest(self, request):
            del request
            raise TimeoutError("optional provider timed out")

    plan = ControlledQueryExpander(FailedProvider()).expand("người đi xe đạp")

    assert [item.query for item in plan.variants] == ["người đi xe đạp"]
    assert plan.warnings == ("query expansion fallback (TimeoutError)",)


def test_no_provider_is_deterministic_original_only():
    expander = ControlledQueryExpander()

    assert expander.expand("  xe buýt đỏ  ") == expander.expand("  xe buýt đỏ  ")
