#!/usr/bin/env python3
"""
User Simulator for interactive instruction clarification.

Simulates a knowledgeable user who can answer agent questions about tasks
and provide failure feedback. Has access to ground truth (evaluator code,
expected outputs, task config) but responds naturally.

Context provided matches clarify_osworld_instructions.py:
  - Full task JSON config
  - Evaluator function source code
  - Actual file contents (images as base64, text in full, xlsx as CSV)
  - Agent trajectory with screenshots

Uses the same LMMAgent / engine infrastructure as the main agent.

Used by run scripts when --method is clarify_upon_retry.
"""

import ast
import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from gui_agents.s3.core.mllm import LMMAgent

logger = logging.getLogger("desktopenv.user_simulator")

_REPO_ROOT = Path(__file__).parent
CACHE_DIR = str(_REPO_ROOT / "cache")
EVALUATORS_DIR = str(_REPO_ROOT / "desktop_env" / "evaluators")


FAILURE_FEEDBACK_SYSTEM_PROMPT = """\
You are the user who issued the following task to a computer-use agent:

"{instruction}"

The agent attempted the task but FAILED. You need to give the agent concise, \
actionable feedback about what went wrong so it can try again.

Below is your internal knowledge about the task:

{context}

{file_diff_section}

RULES FOR FEEDBACK:
1. Be specific about what is wrong — mention exact values, file names, or formats \
that are incorrect.
2. Do NOT give step-by-step instructions. Just point out the problem.
3. Do NOT reveal evaluator function names or code.
4. Keep it to 2-4 sentences.
5. If you can identify the specific mismatch between what the agent produced and \
what you expected, state it clearly."""


# ------------------------------------------------------------------
# File content helpers (matching clarify_osworld_instructions.py)
# ------------------------------------------------------------------

def get_file_content_for_llm(file_path: str) -> Optional[Dict[str, Any]]:
    """Get file content in LLM-friendly format.

    Matches clarify_osworld_instructions.py:
    - Images: base64-encoded as image_url content blocks
    - Text files: full content, no truncation
    - Spreadsheets: converted to CSV
    - Files >10MB: metadata only
    - docx/other binary: metadata only
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)

        if file_size > 10 * 1024 * 1024:
            return {"type": "metadata", "content": f"File too large ({file_size} bytes)", "format": ext}

        if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
            with open(file_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext[1:]};base64,{img_data}"},
            }

        if ext in (".txt", ".json", ".xml", ".csv", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"type": "text", "text": f"File content ({ext}):\n{content}"}

        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
            csv_content = df.to_csv(index=False)
            return {"type": "text", "text": f"Spreadsheet data ({ext}):\n{csv_content}"}

        return {"type": "metadata", "content": f"File format {ext} not displayable ({file_size} bytes)"}

    except Exception as e:
        return {"type": "metadata", "content": f"Error reading file: {e}"}


def extract_cache_files(example_config: Dict[str, Any], example_id: str) -> List[Dict[str, Any]]:
    """Extract cache files referenced in the example config.

    Matches clarify_osworld_instructions.py — handles both input files
    and expected/ground truth files, including list-format expected configs.
    """
    cache_files = []
    cache_path = os.path.join(CACHE_DIR, example_id)

    # Input files from config downloads
    for config_item in example_config.get("config", []):
        if config_item.get("type") == "download":
            for file_info in config_item.get("parameters", {}).get("files", []):
                vm_path = file_info.get("path", "")
                cache_filename = os.path.basename(vm_path)
                cache_file_path = os.path.join(cache_path, cache_filename)
                if os.path.exists(cache_file_path):
                    file_content = get_file_content_for_llm(cache_file_path)
                    cache_files.append({
                        "type": "input_file",
                        "vm_path": vm_path,
                        "description": f"Input file downloaded to {vm_path}",
                        "content": file_content,
                    })

    # Expected/ground truth files from evaluator
    evaluator = example_config.get("evaluator", {})
    expected = evaluator.get("expected", {})
    if isinstance(expected, dict):
        expected_list = [expected]
    elif isinstance(expected, list):
        expected_list = [item for item in expected if isinstance(item, dict)]
    else:
        expected_list = []

    for exp in expected_list:
        if exp.get("type") == "cloud_file":
            dest = exp.get("dest", "")
            if isinstance(dest, list):
                dest = dest[0] if dest else ""
            if dest:
                cache_file_path = os.path.join(cache_path, dest)
                if os.path.exists(cache_file_path):
                    file_content = get_file_content_for_llm(cache_file_path)
                    cache_files.append({
                        "type": "ground_truth_file",
                        "description": f"Expected output/ground truth file: {dest}",
                        "content": file_content,
                    })

    return cache_files


class UserSimulator:
    """Simulates a user who can answer clarification questions and provide
    failure feedback for a specific task."""

    def __init__(
        self,
        example_config: Dict[str, Any],
        env,
        engine_params: Dict[str, Any],
    ):
        """
        Args:
            example_config: The task's JSON config
            env: The DesktopEnv instance (for pulling files from VM)
            engine_params: Engine params dict, e.g. {"engine_type": "azure", "model": "gpt-5", "temperature": 1}
        """
        self.example_config = example_config
        self.example_id = example_config.get("id", "unknown")
        self.instruction = example_config.get("instruction", "")
        self.env = env
        self.engine_params = engine_params
        self.interaction_log: List[Dict[str, Any]] = []

        # Build static context once (text + file content blocks)
        self._context_text, self._context_content_blocks = self._build_context()

        # Persistent LMMAgent so token usage accumulates across calls
        self._agent = LMMAgent(engine_params=self.engine_params, system_prompt="")

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _find_function_in_files(self, func_name: str) -> Optional[str]:
        """Find evaluator function source code by AST walking."""
        for root, _dirs, files in os.walk(EVALUATORS_DIR):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == func_name:
                            lines = content.split("\n")
                            end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
                            return "\n".join(lines[node.lineno - 1 : end])
                except Exception:
                    continue
        return None

    def _get_evaluator_source(self) -> str:
        evaluator = self.example_config.get("evaluator", {})
        func_spec = evaluator.get("func")
        if not func_spec:
            return "No evaluator function specified."

        func_names = [func_spec] if isinstance(func_spec, str) else func_spec
        parts = []
        for fn in func_names:
            if fn == "infeasible":
                parts.append("# Function: infeasible (task marked as impossible)")
                continue
            impl = self._find_function_in_files(fn)
            parts.append(f"# Function: {fn}\n{impl}" if impl else f"# Function: {fn} (not found)")
        return "\n\n".join(parts)

    def _build_context(self):
        """Build static context: returns (text_str, content_blocks_list).

        text_str: task config + evaluator source (for the system prompt)
        content_blocks: file contents as OpenAI content blocks (text + images)
        """
        text = (
            f"## Task Configuration\n```json\n{json.dumps(self.example_config, indent=2)}\n```\n\n"
            f"## Evaluator Function\n```python\n{self._get_evaluator_source()}\n```"
        )

        # File content blocks (matching clarify_osworld_instructions.py)
        cache_files = extract_cache_files(self.example_config, self.example_id)
        content_blocks = []
        for cf in cache_files:
            if cf["content"] is None:
                continue
            if cf["content"]["type"] in ("text", "image_url"):
                content_blocks.append({"type": "text", "text": f"\n**{cf['type']}**: {cf['description']}"})
                content_blocks.append(cf["content"])
            else:
                content_blocks.append({
                    "type": "text",
                    "text": f"\n**{cf['type']}**: {cf['description']}\nFile info: {cf['content']['content']}",
                })

        return text, content_blocks

    # ------------------------------------------------------------------
    # Trajectory formatting with screenshots
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trajectory_content_blocks(
        trajectory: List[Dict],
        trajectory_screenshots: List[bytes],
    ) -> List[Dict[str, Any]]:
        """Build content blocks for agent trajectory including screenshots."""
        blocks = []

        if not trajectory:
            blocks.append({"type": "text", "text": "No actions taken yet."})
            return blocks

        blocks.append({"type": "text", "text": "\n## Agent's trajectory:"})

        for i, step in enumerate(trajectory):
            step_num = step.get("step_num", "?")
            action = step.get("action", "unknown")
            blocks.append({"type": "text", "text": f"Step {step_num}: {action}"})

            # Add screenshot if available for this step
            if i < len(trajectory_screenshots) and trajectory_screenshots[i] is not None:
                img_data = base64.b64encode(trajectory_screenshots[i]).decode("utf-8")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                })

        return blocks

    # ------------------------------------------------------------------
    # Pull agent output files from VM and diff against ground truth
    # ------------------------------------------------------------------

    def _get_file_diff_blocks(self) -> List[Dict[str, Any]]:
        """Pull agent's output files from VM and diff against ground truth.

        Returns content blocks (text + images) for the diff.
        Handles evaluator result/expected as lists.
        """
        evaluator = self.example_config.get("evaluator", {})
        result_cfgs = evaluator.get("result", {})
        expected_cfgs = evaluator.get("expected", {})

        # Normalize to lists
        if isinstance(result_cfgs, dict):
            result_cfgs = [result_cfgs]
        elif not isinstance(result_cfgs, list):
            return []

        if isinstance(expected_cfgs, dict):
            expected_cfgs = [expected_cfgs]
        elif not isinstance(expected_cfgs, list):
            return []

        from desktop_env.evaluators.getters.file import get_vm_file, get_cloud_file

        blocks = []
        for result_cfg, expected_cfg in zip(result_cfgs, expected_cfgs):
            result_type = result_cfg.get("type", "") if isinstance(result_cfg, dict) else ""
            if result_type != "vm_file":
                continue

            # Pull result from VM
            try:
                result_path = get_vm_file(self.env, result_cfg)
            except Exception as e:
                logger.warning(f"Could not pull result file from VM: {e}")
                blocks.append({"type": "text", "text": f"Could not retrieve agent's output: {e}"})
                continue

            if result_path is None:
                blocks.append({"type": "text", "text": "Agent's output file not found on VM."})
                continue

            # Pull expected
            try:
                if isinstance(expected_cfg, dict) and expected_cfg.get("type") == "cloud_file":
                    expected_path = get_cloud_file(self.env, expected_cfg)
                else:
                    continue
            except Exception as e:
                logger.warning(f"Could not get expected file: {e}")
                continue

            if expected_path is None:
                continue

            # Get content blocks for both files (handle list or string paths)
            result_paths = result_path if isinstance(result_path, list) else [result_path]
            expected_paths = expected_path if isinstance(expected_path, list) else [expected_path]

            for rp in result_paths:
                if rp is None:
                    continue
                result_content = get_file_content_for_llm(rp)
                if result_content:
                    blocks.append({"type": "text", "text": f"\n## Agent's output ({os.path.basename(rp)}):"})
                    if result_content["type"] in ("text", "image_url"):
                        blocks.append(result_content)
                    else:
                        blocks.append({"type": "text", "text": result_content.get("content", "")})

            for ep in expected_paths:
                if ep is None:
                    continue
                expected_content = get_file_content_for_llm(ep)
                if expected_content:
                    blocks.append({"type": "text", "text": f"\n## Expected output ({os.path.basename(ep)}):"})
                    if expected_content["type"] in ("text", "image_url"):
                        blocks.append(expected_content)
                    else:
                        blocks.append({"type": "text", "text": expected_content.get("content", "")})

        return blocks

    # ------------------------------------------------------------------
    # LLM call via LMMAgent (multimodal)
    # ------------------------------------------------------------------

    def _call_llm_multimodal(self, system_prompt: str, content_blocks: List[Dict]) -> str:
        """Make a multimodal LLM call with text + image content blocks."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content_blocks},
        ]

        response = self._agent.get_response(messages=messages)

        if response is None or response.strip() == "":
            logger.warning("UserSimulator LLM returned empty response")
            return "Sorry, I'm having trouble responding right now. Please proceed as best you can."
        return response.strip()

    def provide_failure_feedback(self, trajectory: List[Dict], trajectory_screenshots: List[bytes] = None) -> str:
        """Auto-generate feedback after a failed evaluation (Clarify upon Retry).

        Args:
            trajectory: List of step dicts from the failed attempt.
            trajectory_screenshots: List of screenshot bytes corresponding to trajectory steps.

        Returns:
            Feedback string to inject into agent context.
        """
        file_diff_blocks = self._get_file_diff_blocks()
        file_diff_section = (
            "Here is a comparison of what the agent produced vs. what you expected:"
            if file_diff_blocks
            else "No file comparison available for this task."
        )

        system = FAILURE_FEEDBACK_SYSTEM_PROMPT.format(
            instruction=self.instruction,
            context=self._context_text,
            file_diff_section=file_diff_section,
        )

        # Build content blocks: static file context + file diffs + trajectory with screenshots
        content = list(self._context_content_blocks)
        content.extend(file_diff_blocks)
        content.extend(self._build_trajectory_content_blocks(trajectory, trajectory_screenshots or []))
        content.append({"type": "text", "text": "\nThe agent failed. What should it do differently?"})

        feedback = self._call_llm_multimodal(system, content)

        self.interaction_log.append({
            "type": "failure_feedback",
            "feedback": feedback,
            "had_file_diff": bool(file_diff_blocks),
            "step_count_at_time": len(trajectory),
        })
        logger.info(f"[UserSim] Failure feedback: {feedback}")
        return feedback

    def get_token_usage(self) -> Dict[str, int]:
        """Return exact token usage from the user simulator's engine."""
        try:
            engine = self._agent.engine
            return {
                "prompt_tokens": getattr(engine, 'total_prompt_tokens', 0),
                "completion_tokens": getattr(engine, 'total_completion_tokens', 0),
                "total_tokens": getattr(engine, 'total_tokens', 0),
                "num_calls": getattr(engine, 'num_calls', 0),
            }
        except Exception:
            return {}

    def get_log(self) -> Dict[str, Any]:
        """Return full interaction log for saving."""
        return {
            "example_id": self.example_id,
            "instruction": self.instruction,
            "interactions": self.interaction_log,
            "token_usage": self.get_token_usage(),
        }
