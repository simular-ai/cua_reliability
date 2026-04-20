from functools import partial
import json
import logging
import os
import textwrap
from typing import Dict, List, Tuple, Optional

from gui_agents.s3.agents.grounding import ACI
from gui_agents.s3.core.module import BaseModule
from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s3.utils.common_utils import (
    call_llm_safe,
    call_llm_formatted,
    parse_code_from_string,
    split_thinking_response,
    create_pyautogui_code,
)
from gui_agents.s3.utils.formatters import (
    SINGLE_ACTION_FORMATTER,
    CODE_VALID_FORMATTER,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        external_feedback_text: Optional[str] = None,
        enable_plan: bool = False,
        plan_cache_dir: str = None,
        enable_plan_feedback: bool = False,
        plan_feedback_cache: Dict = None,
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            worker_engine_params: Dict
                Parameters for the worker agent
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
            external_feedback_text: str
                External feedback text to include in system prompt
            enable_plan: bool
                Whether to enable pre-generated plan mode
            plan_cache_dir: str
                Path to directory containing pre-generated plans
            enable_plan_feedback: bool
                Whether to enable plan feedback mode
            plan_feedback_cache: Dict
                Dictionary containing plan feedback cache
        """
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        self.use_thinking = worker_engine_params.get("model", "") in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
        ]
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.external_feedback_text = external_feedback_text
        self.enable_plan = enable_plan
        self.plan_cache_dir = plan_cache_dir
        self.enable_plan_feedback = enable_plan_feedback
        self.plan_feedback_cache = plan_feedback_cache or {}
        # Plan-related attributes
        self.current_plan = None
        self.plan_cache = {}

        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            type(self.grounding_agent), skipped_actions=skipped_actions
        ).replace("CURRENT_OS", self.platform)

        # Inject carryover feedback at the start of the system prompt if provided
        if self.external_feedback_text:
            sys_prompt = (
                sys_prompt
                + "\n\n# Carryover Feedback from prior trials\n"
                + self.external_feedback_text.strip()
                + "\n"
            )

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = self._create_agent(
            PROCEDURAL_MEMORY.REFLECTION_ON_TRAJECTORY
        )

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []

    def flush_messages(self):
        """Flush messages based on the model's context limits.

        This method ensures that the agent's message history does not exceed the maximum trajectory length.

        Side Effects:
            - Modifies the messages of generator, reflection, and bon_judge agents to fit within the context limits.
        """
        engine_type = self.engine_params.get("engine_type", "")

        # Flush strategy for long-context models: keep all text, only keep latest images
        if engine_type in ["anthropic", "openai", "gemini"]:
            max_images = self.max_trajectory_length
            for agent in [self.generator_agent, self.reflection_agent]:
                if agent is None:
                    continue
                # keep latest k images
                img_count = 0
                for i in range(len(agent.messages) - 1, -1, -1):
                    for j in range(len(agent.messages[i]["content"])):
                        if "image" in agent.messages[i]["content"][j].get("type", ""):
                            img_count += 1
                            if img_count > max_images:
                                del agent.messages[i]["content"][j]

        # Flush strategy for non-long-context models: drop full turns
        else:
            # generator msgs are alternating [user, assistant], so 2 per round
            if len(self.generator_agent.messages) > 2 * self.max_trajectory_length + 1:
                self.generator_agent.messages.pop(1)
                self.generator_agent.messages.pop(1)
            # reflector msgs are all [(user text, user image)], so 1 per round
            if len(self.reflection_agent.messages) > self.max_trajectory_length + 1:
                self.reflection_agent.messages.pop(1)

    def _generate_reflection(self, instruction: str, obs: Dict) -> Tuple[str, str]:
        """
        Generate a reflection based on the current observation and instruction.

        Args:
            instruction (str): The task instruction.
            obs (Dict): The current observation containing the screenshot.

        Returns:
            Optional[str, str]: The generated reflection text and thoughts, if any (turn_count > 0).

        Side Effects:
            - Updates reflection agent's history
            - Generates reflection response with API call
        """
        reflection = None
        reflection_thoughts = None
        if self.enable_reflection:
            # Load the initial message
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=obs["screenshot"],
                    role="user",
                )
            # Load the latest action
            else:
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1],
                    image_content=obs["screenshot"],
                    role="user",
                )
                full_reflection = call_llm_safe(
                    self.reflection_agent,
                    temperature=self.temperature,
                    use_thinking=self.use_thinking,
                )
                reflection, reflection_thoughts = split_thinking_response(
                    full_reflection
                )
                self.reflections.append(reflection)
                logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
                logger.info("REFLECTION: %s", reflection)
        return reflection, reflection_thoughts
    
    def _load_plan_cache(self) -> Dict:
        """Load the plan cache from the plan cache directory."""
        if not self.plan_cache_dir or not os.path.exists(self.plan_cache_dir):
            logger.warning(f"Plan cache directory not found: {self.plan_cache_dir}")
            return {}
            
        plan_cache_file = os.path.join(self.plan_cache_dir, "plan_cache.json")
        if not os.path.exists(plan_cache_file):
            logger.warning(f"Plan cache file not found: {plan_cache_file}")
            return {}
            
        try:
            with open(plan_cache_file, 'r', encoding='utf-8') as f:
                plan_cache = json.load(f)
            logger.info(f"Loaded plan cache with {len(plan_cache)} entries")
            return plan_cache
        except Exception as e:
            logger.error(f"Failed to load plan cache: {e}")
            return {}
    
    def _get_plan_for_task(self, domain: str, example_id: str) -> Optional[str]:
        """Get the pre-generated plan for a specific task."""
        if not self.enable_plan:
            return None
            
        if not self.plan_cache:
            self.plan_cache = self._load_plan_cache()
            
        plan_key = f"{domain}/{example_id}"
        plan_entry = self.plan_cache.get(plan_key)
        
        if plan_entry:
            plan_text = plan_entry.get("plan_text", "")
            logger.info(f"Found pre-generated plan for {plan_key}")
            return plan_text
        else:
            # Try to load individual plan file
            plan_file = os.path.join(self.plan_cache_dir, domain, example_id, "generated_plan.txt")
            if os.path.exists(plan_file):
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        plan_text = f.read().strip()
                    logger.info(f"Loaded individual plan file for {plan_key}")
                    return plan_text
                except Exception as e:
                    logger.error(f"Failed to load plan file {plan_file}: {e}")
            
            logger.warning(f"No pre-generated plan found for {plan_key}")
            return None
    
    def _inject_plan(self, instruction: str, domain: str = None, example_id: str = None) -> None:
        """Inject pre-generated plan into context on first turn only."""
        if not self.enable_plan or self.turn_count > 0:
            return
            
        plan_text = None
        if domain and example_id:
            plan_text = self._get_plan_for_task(domain, example_id)
        
        if not plan_text:
            logger.warning(f"No plan available to inject for {domain}/{example_id}")
            return
        
        # Format the plan for injection
        plan_injection_text = f"""[PLAN TO FOLLOW] PRE-GENERATED STEP-BY-STEP PLAN: Below is a detailed plan for completing the task: `{instruction}`
You must follow this plan step-by-step to succeed at the task.

{plan_text}

IMPORTANT: Follow this plan closely. Each step should guide your next action. Adapt the plan to the current screen state when necessary, but stay true to the overall plan structure."""
        
        self.generator_agent.add_message(text_content=plan_injection_text, role="user")
        self.current_plan = plan_text
        
        logger.info(f"Injected pre-generated plan for task")
        logger.info(f"PLAN INJECTION TEXT:\n{plan_injection_text}")
    
    def inject_plan_feedback_for_task(self, instruction: str, domain: str, example_id: str) -> None:
        """Inject plan feedback for a specific task.
        
        This should be called after reset() but before generate_next_action().
        Only injects on first turn to avoid repeated injection.
        
        Args:
            instruction: The task instruction
            domain: The domain of the task
            example_id: The example ID
        """
        if not self.enable_plan_feedback or self.turn_count > 0:
            return
            
        task_key = f"{domain}/{example_id}"
        feedback_entry = self.plan_feedback_cache.get(task_key)
        
        if not feedback_entry:
            logger.warning(f"No plan feedback available for task: {task_key}")
            return
        
        feedback_text = feedback_entry.get("feedback", "")
        if not feedback_text:
            logger.warning(f"Empty feedback for task: {task_key}")
            return
        
        # Format the plan feedback for injection
        feedback_injection_text = f"""[PLAN FEEDBACK] LEARNING FROM PREVIOUS ATTEMPTS:

Task: `{instruction}`

Based on analysis of previous successful and failed attempts at this task, here's what you should know:

{feedback_text}

IMPORTANT: Use this feedback to guide your approach. Pay attention to what has worked and what has failed in similar situations. Adapt your strategy based on these insights while staying focused on completing the task successfully."""
        
        self.generator_agent.add_message(text_content=feedback_injection_text, role="user")
        
        logger.info(f"Injected plan feedback for task: {task_key}")
        logger.info(f"FEEDBACK INJECTION TEXT:\n{feedback_injection_text}")

    def inject_plan_for_task(self, instruction: str, domain: str, example_id: str) -> None:
        """Public method to inject a pre-generated plan for a specific task.
        
        This should be called after the agent is reset but before generate_next_action.
        
        Args:
            instruction: The task instruction
            domain: The domain of the task (e.g., 'os', 'office')
            example_id: The example ID (e.g., 'ubuntu_1234')
        """
        if self.turn_count > 0:
            logger.warning("Cannot inject plan after agent has already started generating actions")
            return
            
        self._inject_plan(instruction, domain, example_id)

    def inject_message(self, text: str):
        """Inject a message into the agent's conversation without resetting.

        Used for retry signals and user clarification responses.
        """
        self.generator_agent.add_message(text_content=text, role="user")

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """

        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)

        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )

        # Load the task into the system prompt
        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)
            
            # Inject pre-generated plan if enabled (requires domain and example_id to be passed)
            # This will be called from the run script with proper domain/example_id context

        # Get the per-step reflection
        reflection, reflection_thoughts = self._generate_reflection(instruction, obs)
        if reflection:
            generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"

        # Get the grounding agent's knowledge base buffer
        generator_message += (
            f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.notes)}]\n"
        )

        # Add code agent result from previous step if available (from full task or subtask execution)
        if (
            hasattr(self.grounding_agent, "last_code_agent_result")
            and self.grounding_agent.last_code_agent_result is not None
        ):
            code_result = self.grounding_agent.last_code_agent_result
            generator_message += f"\nCODE AGENT RESULT:\n"
            generator_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            generator_message += f"Steps Completed: {code_result['steps_executed']}\n"
            generator_message += f"Max Steps: {code_result['budget']}\n"
            generator_message += (
                f"Completion Reason: {code_result['completion_reason']}\n"
            )
            generator_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                generator_message += f"Execution History:\n"
                for i, step in enumerate(code_result["execution_history"]):
                    action = step["action"]
                    # Format code snippets with proper backticks
                    if "```python" in action:
                        # Extract Python code and format it
                        code_start = action.find("```python") + 9
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            python_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```python\n{python_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    elif "```bash" in action:
                        # Extract Bash code and format it
                        code_start = action.find("```bash") + 7
                        code_end = action.find("```", code_start)
                        if code_end != -1:
                            bash_code = action[code_start:code_end].strip()
                            generator_message += (
                                f"Step {i+1}: \n```bash\n{bash_code}\n```\n"
                            )
                        else:
                            generator_message += f"Step {i+1}: \n{action}\n"
                    else:
                        generator_message += f"Step {i+1}: \n{action}\n"
            generator_message += "\n"

            # Save code agent result to text file
            try:
                import os
                from datetime import datetime

                # Create logs directory if it doesn't exist
                logs_dir = "logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    f"logs/code_agent_result_step_{self.turn_count + 1}_{timestamp}.txt"
                )

                with open(filename, "w") as f:
                    f.write(f"CODE AGENT RESULT - Step {self.turn_count + 1}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(
                        f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
                    )
                    f.write(f"Steps Completed: {code_result['steps_executed']}\n")
                    f.write(f"Max Steps: {code_result['budget']}\n")
                    f.write(f"Completion Reason: {code_result['completion_reason']}\n")
                    f.write(f"Summary: {code_result['summary']}\n")
                    if code_result["execution_history"]:
                        f.write(f"\nExecution History:\n")
                        for i, step in enumerate(code_result["execution_history"]):
                            f.write(f"\nStep {i+1}:\n")
                            f.write(f"Action: {step['action']}\n")
                            if "thoughts" in step:
                                f.write(f"Thoughts: {step['thoughts']}\n")

                logger.info(f"Code agent result saved to: {filename}")
            except Exception as e:
                logger.error(f"Failed to save code agent result to file: {e}")

            # Log the code agent result section for debugging (truncated execution history)
            log_message = f"\nCODE AGENT RESULT:\n"
            log_message += (
                f"Task/Subtask Instruction: {code_result['task_instruction']}\n"
            )
            log_message += f"Steps Completed: {code_result['steps_executed']}\n"
            log_message += f"Max Steps: {code_result['budget']}\n"
            log_message += f"Completion Reason: {code_result['completion_reason']}\n"
            log_message += f"Summary: {code_result['summary']}\n"
            if code_result["execution_history"]:
                log_message += f"Execution History (truncated):\n"
                # Only log first 3 steps and last 2 steps to keep logs manageable
                total_steps = len(code_result["execution_history"])
                for i, step in enumerate(code_result["execution_history"]):
                    if i < 3 or i >= total_steps - 2:  # First 3 and last 2 steps
                        action = step["action"]
                        if "```python" in action:
                            code_start = action.find("```python") + 9
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                python_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```python\n{python_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        elif "```bash" in action:
                            code_start = action.find("```bash") + 7
                            code_end = action.find("```", code_start)
                            if code_end != -1:
                                bash_code = action[code_start:code_end].strip()
                                log_message += (
                                    f"Step {i+1}: ```bash\n{bash_code}\n```\n"
                                )
                            else:
                                log_message += f"Step {i+1}: {action}\n"
                        else:
                            log_message += f"Step {i+1}: {action}\n"
                    elif i == 3 and total_steps > 5:
                        log_message += f"... (truncated {total_steps - 5} steps) ...\n"

            logger.info(
                f"WORKER_CODE_AGENT_RESULT_SECTION - Step {self.turn_count + 1}: Code agent result added to generator message:\n{log_message}"
            )

            # Reset the code agent result after adding it to context
            self.grounding_agent.last_code_agent_result = None

        # Finalize the generator message
        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )

        # Generate the plan and next action
        format_checkers = [
            SINGLE_ACTION_FORMATTER,
            partial(CODE_VALID_FORMATTER, self.grounding_agent, obs),
        ]
        plan = call_llm_formatted(
            self.generator_agent,
            format_checkers,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        self.worker_history.append(plan)
        self.generator_agent.add_message(plan, role="assistant")
        logger.info("PLAN:\n %s", plan)

        # Extract the next action from the plan
        plan_code = parse_code_from_string(plan)
        try:
            assert plan_code, "Plan code should not be empty"
            exec_code = create_pyautogui_code(self.grounding_agent, plan_code, obs)
        except Exception as e:
            logger.error(
                f"Could not evaluate the following plan code:\n{plan_code}\nError: {e}"
            )
            exec_code = self.grounding_agent.wait(
                1.333
            )  # Skip a turn if the code cannot be evaluated

        executor_info = {
            "plan": plan,
            "plan_code": plan_code,
            "exec_code": exec_code,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
            "code_agent_output": (
                self.grounding_agent.last_code_agent_result
                if hasattr(self.grounding_agent, "last_code_agent_result")
                and self.grounding_agent.last_code_agent_result is not None
                else None
            ),
        }
        self.turn_count += 1
        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()
        return executor_info, [exec_code]
