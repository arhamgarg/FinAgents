import json

from typer.testing import CliRunner

from finagents.cli.main import app
from tests.pipeline import pipeline_scenario


def test_cli_graph_history_show_export(tmp_path):
    runner = CliRunner()
    with pipeline_scenario() as h:
        run = runner.invoke(app, ["run", "RELIANCE", "--date", "2024-12-31", "--json"])
    assert run.exit_code == 0, run.output
    decision = json.loads(run.stdout)["final_trade_decision"]
    assert decision["decision"] in {"approve", "modify", "veto"}
    assert len(h.calls) == 13
    export = runner.invoke(app, ["export", "--format", "json"])
    assert export.exit_code == 0, export.output
    rows = json.loads(export.stdout)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "RELIANCE"
    assert rows[0]["status"] == "success"
    show = runner.invoke(app, ["show", rows[0]["run_id"]])
    assert show.exit_code == 0, show.output
    assert "RELIANCE" in show.output
    history = runner.invoke(app, ["history"])
    assert history.exit_code == 0, history.output
    assert "RELIANCE" in history.output
