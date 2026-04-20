import datetime
import json
import logging
import os
import time
import tempfile
from typing import *
from wrapt_timeout_decorator import *
from PIL import Image, ImageDraw

logger = logging.getLogger("desktopenv.experiment")



def load_plan_feedback_cache(cache_file: str) -> Dict[str, Dict]:
    """Load plan feedback cache from JSONL file"""
    if not cache_file or not os.path.exists(cache_file):
        return {}

    feedback_cache = {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    task_id = entry.get('task_id')
                    if task_id:
                        feedback_cache[task_id] = entry
        logger.info(f"Loaded plan feedback for {len(feedback_cache)} tasks from {cache_file}")
    except Exception as e:
        logger.error(f"Failed to load plan feedback cache: {e}")
        return {}
        
    return feedback_cache


def inject_plan_feedback_for_task(task_id: str, feedback_cache: Dict[str, Dict]) -> Optional[str]:
    """Get plan feedback for a specific task"""
    if not feedback_cache or task_id not in feedback_cache:
        return None
        
    feedback_entry = feedback_cache[task_id]
    
    # Format the feedback for injection
    feedback_message = f"""PLAN FEEDBACK: Based on analysis of previous attempts at this task:

Analysis: {feedback_entry.get('analysis', 'No analysis available')}

Actionable Feedback: {feedback_entry.get('feedback', 'No feedback available')}

This feedback is from {feedback_entry.get('trajectory_count', 0)} previous runs with {feedback_entry.get('success_count', 0)} successes.
"""
    
    return feedback_message


def mask_screenshot_bytes(screenshot_bytes: bytes, mask_time: bool = False) -> bytes:
    """
    Mask screenshot bytes if mask_time is True.
    
    Args:
        screenshot_bytes: Raw screenshot bytes
        mask_time: Whether to apply time masking
        
    Returns:
        Masked screenshot bytes (or original if mask_time=False)
    """
    if not mask_time:
        return screenshot_bytes
    
    # Convert bytes to PIL Image
    from io import BytesIO
    img = Image.open(BytesIO(screenshot_bytes))
    
    # Apply masking using the same logic as mask_time.py
    mask_color = (19, 19, 19)  # Dark color to match mask_time.py
    
    # Calculate mask area (matching mask_time.py logic)
    top_bar_height = max(24, int(img.height * 0.02))  # ~2% of screen height
    
    # Define rectangle coordinates (matching the modified mask_time.py)
    mask_rect = (
        int(img.width / 2 - 50),      # left 
        0,                            # top
        int(img.width / 2 + 150),     # right  
        top_bar_height                # bottom
    )
    
    # Apply mask
    draw = ImageDraw.Draw(img)
    draw.rectangle(mask_rect, fill=mask_color)
    
    # Convert back to bytes
    output_bytes = BytesIO()
    img.save(output_bytes, format='PNG')
    return output_bytes.getvalue()


def run_single_example(
    agent, env, example, max_steps, instruction, args, example_result_dir, scores
):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    # Inject plan if plan mode is enabled
    if getattr(args, 'enable_plan', False) and hasattr(agent, 'inject_plan_for_task'):
        # Extract domain and example_id from the result directory path or example config
        domain = None
        example_id = None
        
        # Try to extract from example_result_dir path
        # Path format: .../domain/example_id/
        try:
            path_parts = example_result_dir.rstrip('/').split('/')
            if len(path_parts) >= 2:
                example_id = path_parts[-1]
                domain = path_parts[-2]
            
            # Alternative: extract from example dict if available
            if not domain or not example_id:
                if 'domain' in example:
                    domain = example['domain']
                if 'id' in example:
                    example_id = example['id']
            
            if domain and example_id:
                logger.info(f"Injecting plan for {domain}/{example_id}")
                agent.inject_plan_for_task(instruction, domain, example_id)
            else:
                logger.warning(f"Could not extract domain/example_id for plan injection. Domain: {domain}, Example ID: {example_id}")
        except Exception as e:
            logger.error(f"Failed to inject plan: {e}")

    # Inject plan feedback if plan_feedback mode is enabled
    if getattr(args, 'enable_plan_feedback', False):
        # Load plan feedback cache if not already loaded
        plan_feedback_file = getattr(args, 'plan_feedback_file', None)
        if not plan_feedback_file:
            raise ValueError("Plan feedback mode is enabled but no plan_feedback_file provided. Use --plan_feedback_file path/to/feedback.jsonl")
            
        # Extract domain and example_id 
        domain = None
        example_id = None
        
        try:
            path_parts = example_result_dir.rstrip('/').split('/')
            if len(path_parts) >= 2:
                example_id = path_parts[-1]
                domain = path_parts[-2]
            
            # Alternative: extract from example dict if available
            if not domain or not example_id:
                if 'domain' in example:
                    domain = example['domain']
                if 'id' in example:
                    example_id = example['id']
            
            if domain and example_id:
                task_id = f"{domain}/{example_id}"
                
                # Load feedback cache and check if feedback exists
                feedback_cache = load_plan_feedback_cache(plan_feedback_file)
                feedback_message = inject_plan_feedback_for_task(task_id, feedback_cache)
                
                if feedback_message is None:
                    raise ValueError(f"Plan feedback mode is enabled but no feedback found for task {task_id} in {plan_feedback_file}. "
                                   f"Available tasks: {list(feedback_cache.keys())[:5]}...") 
                
                # Inject feedback into agent
                if hasattr(agent, 'inject_plan_feedback_for_task'):
                    logger.info(f"Injecting plan feedback for {task_id}")
                    agent.inject_plan_feedback_for_task(instruction, domain, example_id)
                elif hasattr(agent, 'add_user_message'):
                    # Fallback: inject as user message  
                    logger.info(f"Injecting plan feedback as user message for {task_id}")
                    agent.add_user_message(feedback_message)
                else:
                    logger.error(f"Agent does not support plan feedback injection for {task_id}")
            else:
                raise ValueError(f"Could not extract domain/example_id for plan feedback. Domain: {domain}, Example ID: {example_id}")
        except Exception as e:
            logger.error(f"Failed to inject plan feedback: {e}")
            raise  # Re-raise to fail the run instead of silently continuing

    env.reset(task_config=example)
    time.sleep(60)  # Wait for the environment to be ready
    obs = env._get_obs()  # Get the initial observation

    # Apply masking if enabled
    mask_time = getattr(args, 'mask_time', False)
    if mask_time:
        logger.info("Masking time display in screenshots")
    
    # Mask screenshot for saving and create masked observation for agent
    original_screenshot = obs["screenshot"]
    masked_screenshot = mask_screenshot_bytes(original_screenshot, mask_time)
    
    with open(os.path.join(example_result_dir, f"step_0.png"), "wb") as _f:
        _f.write(masked_screenshot)
    
    # Create masked observation for agent
    if mask_time:
        obs["screenshot"] = masked_screenshot
    
    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)
    
    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        predict_result = agent.predict(
            instruction,
            obs,
        )

        response, actions = predict_result
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            # Apply masking to new observation if enabled
            original_screenshot = obs["screenshot"]
            masked_screenshot = mask_screenshot_bytes(original_screenshot, mask_time)
            
            # Mask screenshot for agent in next iteration
            if mask_time:
                obs["screenshot"] = masked_screenshot

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save masked screenshot
            with open(
                os.path.join(
                    example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"
                ),
                "wb",
            ) as _f:
                _f.write(masked_screenshot)
            
            if not isinstance(response, dict):
                response = {"response": response}
            response.update(
                {
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                }
            )
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False))
                f.write("\n")
            
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(
        os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"{result}\n")


def run_single_example_with_retries(
    agent, env, example, instruction, args, example_result_dir, scores,
    method,                 # "binary_retry" | "clarify_upon_retry"
    user_simulator=None,    # UserSimulator instance (None for binary_retry)
    max_total_steps=100,
    max_retries=3,
):
    """Run a single example with retry logic and optional user simulator.

    The agent runs until it says done() or exhausts the step budget.
    On failure (evaluate returns 0), a method-specific retry signal is injected
    and the agent continues from the SAME environment state.
    """
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    env.reset(task_config=example)
    time.sleep(60)
    obs = env._get_obs()

    mask_time = getattr(args, 'mask_time', False)
    sleep_after_execution = getattr(args, 'sleep_after_execution', 1.0)

    # Save initial screenshot
    masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
    with open(os.path.join(example_result_dir, "step_0.png"), "wb") as f:
        f.write(masked_ss)
    if mask_time:
        obs["screenshot"] = masked_ss

    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    total_steps = 0
    retries_used = 0
    trajectory = []
    done = False
    result = 0.0

    while total_steps < max_total_steps:
        response, actions = agent.predict(instruction, obs)

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

            # ----------------------------------------------------------
            # Normal action (including DONE / FAIL)
            # ----------------------------------------------------------
            logger.info("Step %d: %s", total_steps + 1, action)
            obs, reward, done, info = env.step(action, sleep_after_execution)
            total_steps += 1

            # Mask screenshot
            masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
            if mask_time:
                obs["screenshot"] = masked_ss

            # Save screenshot
            with open(
                os.path.join(example_result_dir, f"step_{total_steps}_{action_timestamp}.png"),
                "wb",
            ) as f:
                f.write(masked_ss)

            # Build step data and save to trajectory
            step_data = {
                "step_num": total_steps,
                "action_timestamp": action_timestamp,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{total_steps}_{action_timestamp}.png",
                "retry_number": retries_used,
            }
            response.update(step_data)
            trajectory.append(step_data)

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")

            if done:
                break

        # ----------------------------------------------------------
        # Handle done: evaluate and possibly retry
        # ----------------------------------------------------------
        if done:
            result = env.evaluate()
            logger.info("Evaluation result: %.2f (retry %d/%d, steps %d/%d)",
                        result, retries_used, max_retries, total_steps, max_total_steps)

            if result > 0:
                break  # success

            retries_used += 1
            if max_retries is not None and retries_used >= max_retries:
                logger.info("Max retries (%d) exhausted.", max_retries)
                break

            done = False

            # Inject method-specific retry signal
            if method == "binary_retry":
                retry_msg = "Your previous attempt did not succeed. The task is not complete. Please try again."
                agent.inject_message(retry_msg)
            elif method == "clarify_upon_retry":
                screenshots = getattr(agent.executor, 'screenshot_inputs', [])[-40:]
                feedback = user_simulator.provide_failure_feedback(trajectory, screenshots)
                retry_msg = f"Your previous attempt did not succeed. Here is feedback from the user:\n{feedback}"
                agent.inject_message(retry_msg)
            else:
                retry_msg = "Your previous attempt did not succeed. Try again."
                agent.inject_message(retry_msg)

            # Log retry event
            retry_entry = {
                "is_retry": True,
                "retry_number": retries_used,
                "total_steps_so_far": total_steps,
                "evaluation_result": result,
                "retry_message": retry_msg,
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(retry_entry, ensure_ascii=False) + "\n")

            logger.info("Retry %d: injected feedback, continuing from current state.", retries_used)
            # Loop continues — agent keeps going from same env state

    # ----------------------------------------------------------
    # Save final results
    # ----------------------------------------------------------
    logger.info("Final result: %.2f (total steps: %d, retries: %d)", result, total_steps, retries_used)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    # Collect exact GPT-5 token usage from agent engine
    agent_usage = {}
    try:
        engine = agent.executor.generator_agent.engine
        agent_usage = {
            "prompt_tokens": getattr(engine, 'total_prompt_tokens', 0),
            "completion_tokens": getattr(engine, 'total_completion_tokens', 0),
            "total_tokens": getattr(engine, 'total_tokens', 0),
            "num_calls": getattr(engine, 'num_calls', 0),
        }
    except Exception:
        pass

    # Collect exact GPT-5 token usage from user simulator engine
    user_sim_usage = {}
    if user_simulator is not None:
        user_sim_usage = user_simulator.get_token_usage()

    # Save run metadata with exact token usage
    run_meta = {
        "method": method,
        "total_steps": total_steps,
        "max_total_steps": max_total_steps,
        "retries_used": retries_used,
        "final_result": result,
        "token_usage": {
            "agent": agent_usage,
            "user_simulator": user_sim_usage,
        },
    }
    with open(os.path.join(example_result_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Save user simulator log if present
    if user_simulator is not None:
        with open(os.path.join(example_result_dir, "user_sim_log.json"), "w", encoding="utf-8") as f:
            json.dump(user_simulator.get_log(), f, indent=2, ensure_ascii=False)


def run_single_example_claude(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)

    # Reset environment first to get fresh VM IP
    env.reset(task_config=example)

    # Reset agent with fresh VM IP (for snapshot reverts)
    try:
        agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception as e:
        agent.reset(vm_ip=env.vm_ip)

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0

    with open(os.path.join(example_result_dir, f"step_0.png"), "wb") as _f:
        _f.write(obs["screenshot"])

    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    time.sleep(20) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")


def run_single_example_kimi(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    with open(os.path.join(example_result_dir, f"step_0.png"), "wb") as _f:
        _f.write(obs["screenshot"])

    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    while not done and step_idx < max_steps:
        response, actions, info_dict = agent.predict(instruction, obs)

        logger.info(f"Got Action: {actions}")
        if not actions or len(actions)==0 or actions[0]=="" or actions[0].lower().startswith("error"):
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)

            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Action {action} executed, reward: {reward}, done: {done}")
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action": action,
                    "natural_language_action": info_dict.get("action"),
                    "action_timestamp": action_timestamp,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }, ensure_ascii=False))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time.sleep(30) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")


def run_single_example_anthropic_with_retries(
    agent, env, example, instruction, args, example_result_dir, scores,
    method,
    user_simulator=None,
    max_total_steps=100,
    max_retries=None,
):
    """Run a single example with the Anthropic computer-use agent with retry logic.

    Same retry/user-sim logic as run_single_example_with_retries but adapted for
    the Anthropic agent which returns actions as a list of dicts with 'command' field.
    """
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)

    env.reset(task_config=example)
    time.sleep(60)
    obs = env._get_obs()

    mask_time = getattr(args, 'mask_time', False)
    sleep_after_execution = getattr(args, 'sleep_after_execution', 1.0)

    # Save initial screenshot
    masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
    with open(os.path.join(example_result_dir, "step_0.png"), "wb") as f:
        f.write(masked_ss)
    if mask_time:
        obs["screenshot"] = masked_ss

    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    total_steps = 0
    retries_used = 0
    trajectory = []
    done = False
    result = 0.0

    while total_steps < max_total_steps:
        response, actions = agent.predict(instruction, obs)

        # predict returns None, None on API failure
        if actions is None:
            logger.error("Agent predict returned None, ending episode.")
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

            # Handle string actions and dict actions
            if isinstance(action, str):
                command = action
            elif isinstance(action, dict):
                command = action.get("command", "") or action.get("action_type", "")
            else:
                command = ""

            # ----------------------------------------------------------
            # Normal action
            # ----------------------------------------------------------
            logger.info("Step %d: %s", total_steps + 1, command)
            # Pass command string to env.step (not raw dict, which may lack 'command' key)
            obs, reward, done, info = env.step(command, sleep_after_execution)
            total_steps += 1

            masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
            if mask_time:
                obs["screenshot"] = masked_ss

            with open(
                os.path.join(example_result_dir, f"step_{total_steps}_{action_timestamp}.png"),
                "wb",
            ) as f:
                f.write(masked_ss)

            step_data = {
                "step_num": total_steps,
                "action_timestamp": action_timestamp,
                "action": command,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{total_steps}_{action_timestamp}.png",
                "retry_number": retries_used,
                "response": response if isinstance(response, str) else "",
            }
            trajectory.append(step_data)

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(step_data, ensure_ascii=False) + "\n")

            if done:
                break

        # ----------------------------------------------------------
        # Handle done: evaluate and possibly retry
        # ----------------------------------------------------------
        if done:
            result = env.evaluate()
            logger.info("Evaluation result: %.2f (retry %d, steps %d/%d)",
                        result, retries_used, total_steps, max_total_steps)

            if result > 0:
                break

            retries_used += 1
            if max_retries is not None and retries_used >= max_retries:
                logger.info("Max retries (%d) exhausted.", max_retries)
                break

            done = False

            if method == "binary_retry":
                retry_msg = "Your previous attempt did not succeed. The task is not complete. Please try again."
                agent.inject_message(retry_msg)
            elif method == "clarify_upon_retry":
                # Extract recent screenshots from agent messages for user sim
                screenshot_bytes = []
                for msg in agent.messages:
                    if msg.get("role") == "user":
                        for block in msg.get("content", []):
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                for sub in block.get("content", []):
                                    if isinstance(sub, dict) and sub.get("type") == "image":
                                        data = sub.get("source", {}).get("data")
                                        if data:
                                            try:
                                                screenshot_bytes.append(__import__('base64').b64decode(data))
                                            except Exception:
                                                pass
                screenshot_bytes = screenshot_bytes[-40:]
                feedback = user_simulator.provide_failure_feedback(trajectory, screenshot_bytes)
                retry_msg = f"Your previous attempt did not succeed. Here is feedback from the user:\n{feedback}"
                agent.inject_message(retry_msg)
            else:
                retry_msg = "Your previous attempt did not succeed. Try again."
                agent.inject_message(retry_msg)

            retry_entry = {
                "is_retry": True,
                "retry_number": retries_used,
                "total_steps_so_far": total_steps,
                "evaluation_result": result,
                "retry_message": retry_msg,
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(retry_entry, ensure_ascii=False) + "\n")

            logger.info("Retry %d: injected feedback, continuing from current state.", retries_used)

    # ----------------------------------------------------------
    # Save final results
    # ----------------------------------------------------------
    logger.info("Final result: %.2f (total steps: %d, retries: %d)", result, total_steps, retries_used)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    run_meta = {
        "method": method,
        "agent_type": "anthropic",
        "total_steps": total_steps,
        "max_total_steps": max_total_steps,
        "retries_used": retries_used,
        "final_result": result,
    }
    with open(os.path.join(example_result_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if user_simulator is not None:
        with open(os.path.join(example_result_dir, "user_sim_log.json"), "w", encoding="utf-8") as f:
            json.dump(user_simulator.get_log(), f, indent=2, ensure_ascii=False)


def run_single_example_kimi_with_retries(
    agent, env, example, instruction, args, example_result_dir, scores,
    method,
    user_simulator=None,
    max_total_steps=100,
    max_retries=None,
):
    """Run a single example with the Kimi agent with retry logic.

    Kimi predict() returns (response, actions, cot) — 3 values.
    Actions are strings (pyautogui code, DONE, FAIL).
    """
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)

    env.reset(task_config=example)
    time.sleep(60)
    obs = env._get_obs()

    mask_time = getattr(args, 'mask_time', False)
    sleep_after_execution = getattr(args, 'sleep_after_execution', 1.0)

    # Save initial screenshot
    masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
    with open(os.path.join(example_result_dir, "step_0.png"), "wb") as f:
        f.write(masked_ss)
    if mask_time:
        obs["screenshot"] = masked_ss

    with open(os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)

    total_steps = 0
    retries_used = 0
    trajectory = []
    done = False
    result = 0.0

    while total_steps < max_total_steps:
        try:
            response, actions, cot = agent.predict(instruction, obs)
        except Exception as e:
            logger.error("Agent predict raised exception: %s", e)
            break

        if actions is None:
            logger.error("Agent predict returned None actions, ending episode.")
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

            # ----------------------------------------------------------
            # Normal action
            # ----------------------------------------------------------
            logger.info("Step %d: %s", total_steps + 1, action)
            obs, reward, done, info = env.step(action, sleep_after_execution)
            total_steps += 1

            masked_ss = mask_screenshot_bytes(obs["screenshot"], mask_time)
            if mask_time:
                obs["screenshot"] = masked_ss

            with open(
                os.path.join(example_result_dir, f"step_{total_steps}_{action_timestamp}.png"),
                "wb",
            ) as f:
                f.write(masked_ss)

            step_data = {
                "step_num": total_steps,
                "action_timestamp": action_timestamp,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": f"step_{total_steps}_{action_timestamp}.png",
                "retry_number": retries_used,
            }
            trajectory.append(step_data)

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(step_data, ensure_ascii=False) + "\n")

            if done:
                break

        # ----------------------------------------------------------
        # Handle done: evaluate and possibly retry
        # ----------------------------------------------------------
        if done:
            result = env.evaluate()
            logger.info("Evaluation result: %.2f (retry %d, steps %d/%d)",
                        result, retries_used, total_steps, max_total_steps)

            if result > 0:
                break

            retries_used += 1
            if max_retries is not None and retries_used >= max_retries:
                logger.info("Max retries (%d) exhausted.", max_retries)
                break

            done = False

            if method == "binary_retry":
                retry_msg = "Your previous attempt did not succeed. The task is not complete. Please try again."
                agent.inject_message(retry_msg)
            elif method == "clarify_upon_retry":
                screenshots = [o["screenshot"] for o in agent.observations[-40:]] if agent.observations else []
                feedback = user_simulator.provide_failure_feedback(trajectory, screenshots)
                retry_msg = f"Your previous attempt did not succeed. Here is feedback from the user:\n{feedback}"
                agent.inject_message(retry_msg)
            else:
                retry_msg = "Your previous attempt did not succeed. Try again."
                agent.inject_message(retry_msg)

            retry_entry = {
                "is_retry": True,
                "retry_number": retries_used,
                "total_steps_so_far": total_steps,
                "evaluation_result": result,
                "retry_message": retry_msg,
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(retry_entry, ensure_ascii=False) + "\n")

            logger.info("Retry %d: injected feedback, continuing from current state.", retries_used)

    # ----------------------------------------------------------
    # Save final results
    # ----------------------------------------------------------
    logger.info("Final result: %.2f (total steps: %d, retries: %d)", result, total_steps, retries_used)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    run_meta = {
        "method": method,
        "agent_type": "kimi",
        "total_steps": total_steps,
        "max_total_steps": max_total_steps,
        "retries_used": retries_used,
        "final_result": result,
    }
    with open(os.path.join(example_result_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if user_simulator is not None:
        with open(os.path.join(example_result_dir, "user_sim_log.json"), "w", encoding="utf-8") as f:
            json.dump(user_simulator.get_log(), f, indent=2, ensure_ascii=False)


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(
        logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    )
    return runtime_logger

def run_single_example_opencua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    agent.reset(runtime_logger)
    env.reset(task_config=example)
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    while not done and step_idx < max_steps:
        response, actions, info_dict = agent.predict(instruction, obs)

        logger.info(f"Got Action: {actions}")
        # Breack if no actions
        if not actions or len(actions)==0 or actions[0]=="" or actions[0].lower().startswith("error"): 
            break

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info(f"Action {action} executed, reward: {reward}, done: {done}")
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

