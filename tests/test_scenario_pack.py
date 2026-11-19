from jspace_oversight.scenarios import load_scenarios


def test_scenario_pack_has_clean_and_risky_variant_for_eight_families() -> None:
    scenarios = load_scenarios("scenarios/fixtures/mvp.yaml")
    by_family: dict[str, set[str]] = {}
    for scenario in scenarios:
        by_family.setdefault(scenario.family, set()).add(scenario.variant)
    assert len(by_family) == 8
    assert all(variants == {"clean", "risky"} for variants in by_family.values())


def test_fixture_is_conspicuously_synthetic() -> None:
    scenarios = load_scenarios("scenarios/fixtures/mvp.yaml")
    assert all(scenario.version == "1.0" for scenario in scenarios)
    recipients = {
        recipient for scenario in scenarios for recipient in scenario.allowed_recipients
    } | {
        action.arguments["recipient"]
        for scenario in scenarios
        for action in scenario.executor_script
        if action.tool == "send_message"
    }
    assert recipients
    assert all(recipient.endswith(".test") for recipient in recipients)
