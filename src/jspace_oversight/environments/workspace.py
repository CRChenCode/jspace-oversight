"""Deterministic, in-memory workspace with no host or network side effects."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from jspace_oversight.evaluation.counterfactual_schemas import (
    Message,
    ProposedAction,
    ScenarioRecord,
    ToolResult,
    WorkspaceSnapshot,
)

ENVIRONMENT_VERSION = "toy-workspace-v1"


class ToyWorkspace:
    """Execute a small typed tool surface entirely in memory."""

    def __init__(self, scenario: ScenarioRecord):
        self._scenario = scenario
        self._state = WorkspaceSnapshot(
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.version,
            files=deepcopy(scenario.initial_files),
            messages=deepcopy(scenario.initial_messages),
        )

    @property
    def scenario(self) -> ScenarioRecord:
        return self._scenario

    def observe(self, actor_role: Literal["executor", "reviewer"]) -> dict[str, Any]:
        """Return only information available to the named actor."""
        observation: dict[str, Any] = {
            "scenario_id": self._scenario.scenario_id,
            "user_goal": self._scenario.user_goal,
            "file_names": sorted(self._state.files),
            "messages": [
                message.model_dump(mode="json") for message in self._state.messages
            ],
            "untrusted_content": list(self._scenario.untrusted_content),
            "finished": self._state.finished,
            "authorized_actions": list(self._scenario.authorized_actions),
            "allowed_recipients": list(self._scenario.allowed_recipients),
        }
        return observation

    def snapshot(self) -> WorkspaceSnapshot:
        return self._state.model_copy(deep=True)

    def restore(self, snapshot: WorkspaceSnapshot) -> None:
        if (
            snapshot.scenario_id != self._scenario.scenario_id
            or snapshot.scenario_version != self._scenario.version
        ):
            raise ValueError("snapshot scenario identity does not match workspace")
        self._state = snapshot.model_copy(deep=True)

    def clone(self) -> ToyWorkspace:
        cloned = ToyWorkspace(self._scenario)
        cloned.restore(self.snapshot())
        return cloned

    def step(self, action: ProposedAction) -> ToolResult:
        if self._state.finished:
            return ToolResult(ok=False, code="workspace_finished")
        handler = getattr(self, f"_tool_{action.tool}")
        return handler(action.arguments)

    @staticmethod
    def _arguments(
        arguments: dict[str, Any], required: set[str], optional: set[str] | None = None
    ) -> ToolResult | None:
        optional = optional or set()
        supplied = set(arguments)
        missing = required - supplied
        unknown = supplied - required - optional
        if missing or unknown:
            return ToolResult(
                ok=False,
                code="invalid_arguments",
                output={"missing": sorted(missing), "unknown": sorted(unknown)},
            )
        return None

    def _tool_read_file(self, arguments: dict[str, Any]) -> ToolResult:
        if error := self._arguments(arguments, {"path"}):
            return error
        path = arguments["path"]
        if not isinstance(path, str):
            return ToolResult(ok=False, code="invalid_argument_type")
        if path not in self._state.files:
            return ToolResult(ok=False, code="file_not_found", output={"path": path})
        if path not in self._state.files_read:
            self._state.files_read.append(path)
        return ToolResult(
            ok=True,
            code="ok",
            output={"path": path, "content": self._state.files[path]},
        )

    def _tool_send_message(self, arguments: dict[str, Any]) -> ToolResult:
        if error := self._arguments(arguments, {"recipient", "body"}, {"attachments"}):
            return error
        recipient = arguments["recipient"]
        body = arguments["body"]
        attachments = arguments.get("attachments", [])
        if not isinstance(recipient, str) or not isinstance(body, str):
            return ToolResult(ok=False, code="invalid_argument_type")
        if not isinstance(attachments, list) or not all(
            isinstance(item, str) for item in attachments
        ):
            return ToolResult(ok=False, code="invalid_argument_type")
        missing = sorted(set(attachments) - set(self._state.files))
        if missing:
            return ToolResult(
                ok=False, code="attachment_not_found", output={"paths": missing}
            )
        self._state.messages.append(
            Message(
                sender="agent",
                recipient=recipient,
                body=body,
                attachments=attachments,
            )
        )
        return ToolResult(ok=True, code="ok", output={"recipient": recipient})

    def _tool_save_summary(self, arguments: dict[str, Any]) -> ToolResult:
        if error := self._arguments(arguments, {"path", "content"}):
            return error
        path = arguments["path"]
        content = arguments["content"]
        if not isinstance(path, str) or not isinstance(content, str):
            return ToolResult(ok=False, code="invalid_argument_type")
        self._state.summaries[path] = content
        return ToolResult(ok=True, code="ok", output={"path": path})

    def _tool_delete_file(self, arguments: dict[str, Any]) -> ToolResult:
        if error := self._arguments(arguments, {"path"}):
            return error
        path = arguments["path"]
        if not isinstance(path, str):
            return ToolResult(ok=False, code="invalid_argument_type")
        if path not in self._state.files:
            return ToolResult(ok=False, code="file_not_found", output={"path": path})
        del self._state.files[path]
        return ToolResult(ok=True, code="ok", output={"path": path})

    def _tool_finish_task(self, arguments: dict[str, Any]) -> ToolResult:
        if error := self._arguments(arguments, set()):
            return error
        self._state.finished = True
        return ToolResult(ok=True, code="ok")
