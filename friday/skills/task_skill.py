"""
F.R.I.D.A.Y. Task Skill
=======================
The bridge between the user's voice/commands and the GoalPlanner/GoalEngine.
Allows F.R.I.D.A.Y. to plan and execute multi-step autonomous tasks.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from friday.actions import Action, ActionResult, ActionStatus
from friday.goal_engine import get_goal_engine
from friday.goal_planner import get_goal_planner

logger = logging.getLogger(__name__)


class TaskSkill:
    """
    Skill for managing autonomous task planning and execution.
    Bridges FridayEngine's goal components to the ActionRouter.
    """

    def __init__(self):
        self._planner = get_goal_planner()
        self._engine = get_goal_engine()
        self._logger = logging.getLogger("TaskSkill")

    def register(self, router: Any) -> None:
        """Register task-related actions with the router."""
        router.register(
            Action(
                name="plan_task",
                patterns=[r"(?:plan|design|outline)\s+(?:task|objective|goal|mission)\s+(?P<objective>.+)"],
                handler=self._handle_plan_task,
                description="Decompose a high-level objective into a structured plan."
            )
        )
        router.register(
            Action(
                name="execute_task",
                patterns=[r"(?:execute|start|run|begin|perform)\s+(?:task|objective|goal|mission)\s+(?P<objective>.+)"],
                handler=self._handle_execute_task,
                description="Plan and start execution of a task immediately."
            )
        )
        router.register(
            Action(
                name="task_status",
                patterns=[
                    r"(?:check\s+)?(?:task|goal|job)\s+status(?:\s+(?P<goal_id>\d+))?",
                    r"status\s+of\s+(?:task|goal)\s+(?P<goal_id>\d+)",
                    r"what\s+tasks?\s+are\s+(?:running|active|in\s+progress)"
                ],
                handler=self._handle_task_status,
                description="Check the status of active tasks."
            )
        )
        router.register(
            Action(
                name="cancel_task",
                patterns=[r"(?:cancel|stop|terminate|kill)\s+(?:task|goal)\s+(?P<goal_id>\d+)"],
                handler=self._handle_cancel_task,
                description="Cancel a running task."
            )
        )

    def _handle_plan_task(self, match: re.Match) -> ActionResult:
        objective = match.group("objective")
        return self._action_plan_task(objective)

    def _handle_execute_task(self, match: re.Match) -> ActionResult:
        objective = match.group("objective")
        return self._action_execute_task(objective)

    def _handle_task_status(self, match: re.Match) -> ActionResult:
        gd = match.groupdict()
        goal_id = gd.get("goal_id")
        return self._action_task_status(goal_id)

    def _handle_cancel_task(self, match: re.Match) -> ActionResult:
        goal_id = match.group("goal_id")
        return self._action_cancel_task(goal_id)

    def _action_plan_task(self, objective: str) -> ActionResult:
        """
        Decompose a high-level objective into a structured plan.
        """
        if not self._planner.is_ready:
            self._planner.setup()

        plan = self._planner.plan(objective)
        if not plan or not plan.success:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I'm sorry, Sir, I couldn't formulate a plan for that objective.",
            )

        explanation = self._planner.explain_plan(plan)
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=explanation,
            data={
                "plan": plan.to_goal_steps(),
                "title": plan.title,
                "priority": plan.priority.value,
            },
        )

    def _action_execute_task(self, objective: str) -> ActionResult:
        """
        Plan AND start execution of a task immediately.
        """
        if not self._planner.is_ready:
            self._planner.setup()

        plan = self._planner.plan(objective)
        if not plan or not plan.success:
            return ActionResult(status=ActionStatus.EXECUTION_ERROR, message="Failed to plan the task, Sir.")

        goal_id = self._engine.create_goal(
            title=plan.title,
            description=plan.description,
            steps=plan.to_goal_steps(),
            priority=plan.priority,
        )
        self._engine.start_goal(goal_id)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Acknowledged. I've started the plan for '{plan.title}' in the background.",
            data={"goal_id": goal_id},
        )

    def _action_task_status(self, goal_id: str | None = None) -> ActionResult:
        """Check the status of active tasks."""
        if goal_id:
            try:
                gid = int(goal_id)
                status = self._engine.get_goal_status(gid)
                if not status:
                    return ActionResult(status=ActionStatus.EXECUTION_ERROR, message=f"Task {gid} not found.")
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Task {gid} is {status}.",
                    data={"status": status},
                )
            except ValueError:
                return ActionResult(status=ActionStatus.EXECUTION_ERROR, message="Invalid Task ID.")

        active = self._engine.get_active_goals()
        if not active:
            return ActionResult(
                status=ActionStatus.SUCCESS, message="No active tasks at the moment, Sir."
            )

        summary = "\n".join([f"- [{g['id']}] {g['title']} ({g['status']})" for g in active])
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"I have {len(active)} tasks in progress:\n{summary}",
            data={"tasks": active},
        )

    def _action_cancel_task(self, goal_id: str) -> ActionResult:
        """Cancel a running task."""
        try:
            gid = int(goal_id)
            self._engine.cancel_goal(gid)
            return ActionResult(
                status=ActionStatus.SUCCESS, message=f"Task {gid} has been cancelled."
            )
        except ValueError:
            return ActionResult(status=ActionStatus.EXECUTION_ERROR, message="Invalid Task ID.")


def register(router: Any) -> None:
    """Register the TaskSkill with the router."""
    skill = TaskSkill()
    skill.register(router)
