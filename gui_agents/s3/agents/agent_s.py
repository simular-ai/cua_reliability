import json
import logging
import os
import platform
from typing import Dict, List, Tuple

from gui_agents.s3.agents.grounding import ACI
from gui_agents.s3.agents.worker import Worker

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
    ):
        """Initialize UIAgent

        Args:
            worker_engine_params: Configuration parameters for the worker LLM agent
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
        """
        self.worker_engine_params = worker_engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass


class AgentS3(UIAgent):
    """Agent that uses no hierarchy for less inference time"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        feedback_results_file_path: str = None,
        feedback_key: str = None,
        enable_plan: bool = False,
        plan_cache_dir: str = None,
        enable_plan_feedback: bool = False,
        plan_feedback_file: str = None,
    ):
        """Initialize a minimalist AgentS2 without hierarchy

        Args:
            worker_engine_params: Configuration parameters for the worker agent.
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
            feedback_results_file_path: Path to feedback results file
            feedback_key: Key for extracting feedback from results file
            enable_plan: Whether to enable pre-generated plan mode
            plan_cache_dir: Directory containing pre-generated plans
            enable_plan_feedback: Whether to enable plan feedback mode
            plan_feedback_file: File containing plan feedback cache (JSONL format)
        """

        super().__init__(worker_engine_params, grounding_agent, platform)
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.enable_plan = enable_plan
        self.plan_cache_dir = plan_cache_dir
        self.enable_plan_feedback = enable_plan_feedback
        self.plan_feedback_file = plan_feedback_file
        # Optional carryover feedback injection
        external_feedback_text = None
        try:
            if feedback_results_file_path and feedback_key:
                if os.path.isfile(feedback_results_file_path):
                    with open(feedback_results_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Expect a dict mapping {domain}/{example_id} -> feedback text
                    external_feedback_text = data.get(feedback_key)
        except Exception as e:
            logger.error(f"Failed to load feedback from {feedback_results_file_path} with key {feedback_key}: {e}")

        # Load plan feedback cache if enabled
        self.plan_feedback_cache = {}
        if self.enable_plan_feedback and self.plan_feedback_file:
            try:
                if os.path.exists(self.plan_feedback_file):
                    with open(self.plan_feedback_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                feedback_entry = json.loads(line)
                                task_id = feedback_entry.get('task_id')
                                if task_id:
                                    self.plan_feedback_cache[task_id] = feedback_entry
                    logger.info(f"Loaded {len(self.plan_feedback_cache)} plan feedback entries")
                else:
                    logger.warning(f"Plan feedback cache file not found: {self.plan_feedback_file}")
            except Exception as e:
                logger.error(f"Failed to load plan feedback cache: {e}")

        self.reset(external_feedback_text=external_feedback_text)

    def reset(self, external_feedback_text: str = None) -> None:
        """Reset agent state and initialize components"""
        self.executor = Worker(
            worker_engine_params=self.worker_engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
            external_feedback_text=external_feedback_text,
            enable_plan=self.enable_plan,
            plan_cache_dir=self.plan_cache_dir,
            enable_plan_feedback=self.enable_plan_feedback,
            plan_feedback_cache=getattr(self, 'plan_feedback_cache', {}),
        )

    def inject_message(self, text: str) -> None:
        """Inject a message into the agent's conversation without resetting."""
        if hasattr(self.executor, 'inject_message'):
            self.executor.inject_message(text)

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # Initialize the three info dictionaries
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )

        # concatenate the three info dictionaries
        info = {**{k: v for d in [executor_info or {}] for k, v in d.items()}}

        return info, actions
    
    def inject_plan_for_task(self, instruction: str, domain: str, example_id: str) -> None:
        """Inject a pre-generated plan for a specific task.
        
        This should be called after reset() but before the first predict() call.
        
        Args:
            instruction: The task instruction
            domain: The domain of the task (e.g., 'os', 'office') 
            example_id: The example ID (e.g., 'ubuntu_1234')
        """
        if hasattr(self.executor, 'inject_plan_for_task'):
            self.executor.inject_plan_for_task(instruction, domain, example_id)
    
    def inject_plan_feedback_for_task(self, instruction: str, domain: str, example_id: str) -> None:
        """Inject plan feedback for a specific task.
        
        This should be called after reset() but before the first predict() call.
        
        Args:
            instruction: The task instruction
            domain: The domain of the task (e.g., 'os', 'office')
            example_id: The example ID (e.g., 'ubuntu_1234') 
        """
        if hasattr(self.executor, 'inject_plan_feedback_for_task'):
            self.executor.inject_plan_feedback_for_task(instruction, domain, example_id)
        else:
            logger.warning("Executor does not support plan injection")
