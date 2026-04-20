"""Plan Generation Script - Generate step-by-step plans for given tasks.

Example:

python experiments/generate_plans.py \
    --provider_name "aws" \
    --headless \
    --num_envs 5 \
    --region "us-east-1" \
    --test_all_meta_path evaluation_examples/test_nogdrive.json \
    --result_dir "./plan_cache" \
    --model_provider "openai" \
    --model uitars15-7b \
    --model_url "http://127.0.0.1:8000/v1" \
    --model_temperature 0 \
    --action_space "pyautogui" \
    --observation_type "screenshot"
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from multiprocessing import Process, Manager, current_process, Queue

# Add repo root and osworld to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, 'osworld'))

import lib_run_single
from desktop_env.desktop_env import DesktopEnv

from dotenv import load_dotenv
load_dotenv()

# Logger Configs
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)

stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter("desktopenv"))
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")

def distribute_tasks(test_all_meta: dict) -> list:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks

def generate_plan_for_task(task_queue: Queue, args: argparse.Namespace, shared_plans: dict, engine_params):
    active_environments = []
    env = None
    try:
        # Use IMAGE_ID_MAP for AWS provider to get snapshot_name
        snapshot_name = None
        region = getattr(args, 'region', None)
        if args.provider_name == 'aws' and region is not None:
            try:
                from desktop_env.providers.aws.manager import IMAGE_ID_MAP
                screen_size = (args.screen_width, args.screen_height)
                snapshot_name = IMAGE_ID_MAP[region].get(screen_size, IMAGE_ID_MAP[region][(1920, 1080)])
            except Exception as e:
                logger.error(f"Failed to get snapshot_name from IMAGE_ID_MAP: {e}")
                snapshot_name = None

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=region,
            snapshot_name=snapshot_name,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=False,  # Plan generation only needs screenshots
            enable_proxy=True,
            client_password=getattr(args, 'client_password', ''),
        )

        # Create a special plan generation agent (no grounding needed)
        agent = PlanGenerationAgent(
            engine_params,
            platform="linux",
        )
        
        active_environments.append(env)
        logger.info(f"Plan generation process {current_process().name} started.")
        
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            
            domain, example_id = item
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"{args.examples_dir}/{domain}/{example_id}.json"
                )
                
                # Handle clarified examples structure (domain/example_id/example_id.json)
                if not os.path.exists(config_file):
                    config_file = os.path.join(
                        args.test_config_base_dir, f"{args.examples_dir}/{domain}/{example_id}/{example_id}.json"
                    )
                    
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                    
                instruction = example["instruction"]
                
                # Create plan directory
                plan_result_dir = os.path.join(
                    args.result_dir,
                    "plans",
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )
                os.makedirs(plan_result_dir, exist_ok=True)
                
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {instruction}")
                
                try:
                    # Check if we should force regeneration
                    force_regenerate = getattr(args, 'force_regenerate', False)
                    
                    # Generate plan for this task
                    plan = generate_single_plan(
                        agent,
                        env,
                        example,
                        instruction,
                        plan_result_dir,
                        shared_plans,
                        domain,
                        example_id,
                        force_regenerate
                    )
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in plan generation for {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Save error info
                    error_file = os.path.join(plan_result_dir, "plan_generation_error.txt")
                    with open(error_file, "w") as f:
                        f.write(f"Plan generation failed for {domain}/{example_id}: {e}\n")
                        f.write(traceback.format_exc())
                        
            except Exception as e:
                logger.error(f"Task-level error in plan generation {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
    except Exception as e:
        logger.error(f"Process-level error in plan generation {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")

def generate_single_plan(agent, env, example, instruction, plan_result_dir, shared_plans, domain, example_id, force_regenerate=False):
    """Generate a single step-by-step plan for the given task - just initial screenshot + plan generation."""
    
    # Check if plan already exists (caching logic)
    plan_file = os.path.join(plan_result_dir, "generated_plan.txt")
    plan_result_file = os.path.join(plan_result_dir, "plan_result.json")
    
    if not force_regenerate and os.path.exists(plan_file) and os.path.exists(plan_result_file):
        logger.info(f"Plan already exists for {domain}/{example_id}, loading from cache...")
        
        # Load existing plan
        with open(plan_file, "r", encoding="utf-8") as f:
            plan_text = f.read()
        
        with open(plan_result_file, "r", encoding="utf-8") as f:
            plan_result = json.load(f)
        
        # Add to shared plans cache
        plan_key = f"{domain}/{example_id}"
        shared_plans[plan_key] = {
            "plan_text": plan_text,
            "plan_file": plan_file,
            "instruction": instruction,
            "domain": domain,
            "example_id": example_id
        }
        
        logger.info(f"Loaded existing plan for {domain}/{example_id}")
        return plan_text
    
    logger.info(f"Generating new plan for {domain}/{example_id}...")
    
    # Reset environment and get initial screenshot only
    env.reset(task_config=example)
    time.sleep(60)  # Wait for the environment to be ready
    obs = env._get_obs()  # Get the initial observation

    # Save initial screenshot
    screenshot_data = obs["screenshot"]
    screenshot_path = os.path.join(plan_result_dir, "initial_screenshot.png")
    
    if isinstance(screenshot_data, bytes):
        # Screenshot is already in bytes format
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_data)
    else:
        # Screenshot is numpy array or PIL Image, convert to PNG
        import numpy as np
        from PIL import Image
        
        if isinstance(screenshot_data, np.ndarray):
            img = Image.fromarray(screenshot_data)
        elif hasattr(screenshot_data, 'save'):
            img = screenshot_data
        else:
            raise ValueError(f"Unsupported screenshot data type: {type(screenshot_data)}")
            
        img.save(screenshot_path, format='PNG')
    
    # Save instruction
    with open(os.path.join(plan_result_dir, "instruction.txt"), "w", encoding="utf-8") as f:
        f.write(instruction)
    
    # Generate the plan using simple LLM call
    plan_result = agent.generate_plan(instruction, obs)
    
    # Extract plan text from result
    plan_text = plan_result.get("plan", "")
    
    logger.info(f"Plan text extracted: '{plan_text}' (length: {len(plan_text)})")
    
    # Save plan to file
    plan_file = os.path.join(plan_result_dir, "generated_plan.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write(plan_text)
    
    # Save full plan result with metadata
    plan_result_file = os.path.join(plan_result_dir, "plan_result.json")
    with open(plan_result_file, "w", encoding="utf-8") as f:
        json.dump(plan_result, f, indent=2, ensure_ascii=False)
    
    # Add to shared plans cache
    plan_key = f"{domain}/{example_id}"
    shared_plans[plan_key] = {
        "plan_text": plan_text,
        "plan_file": plan_file,
        "instruction": instruction,
        "domain": domain,
        "example_id": example_id
    }
    
    logger.info(f"Generated plan for {domain}/{example_id}")
    logger.info(f"PLAN:\n{plan_text}")
    
    return plan_text

class PlanGenerationAgent:
    """A simple agent that generates step-by-step plans for tasks using the proper agent pattern."""
    
    def __init__(self, engine_params, platform="linux"):
        self.engine_params = engine_params
        self.platform = platform
        
        from gui_agents.s3.utils.common_utils import call_llm_safe, call_llm_formatted
        from gui_agents.s3.core.module import BaseModule
        import re
        
        self.call_llm_safe = call_llm_safe
        self.call_llm_formatted = call_llm_formatted
        self.BaseModule = BaseModule
        
    def generate_plan(self, instruction: str, obs: dict) -> dict:
        """Generate a step-by-step plan for the given task using the proper agent pattern."""
        
        import base64
        import io
        from PIL import Image
        import numpy as np
        import logging
        import re
        
        logger = logging.getLogger("desktopenv.experiment")
        
        # Handle screenshot data - could be bytes or numpy array
        screenshot_data = obs["screenshot"]
        
        if isinstance(screenshot_data, bytes):
            # Screenshot is already in bytes format
            screenshot_bytes = screenshot_data
        elif isinstance(screenshot_data, np.ndarray):
            # Screenshot is numpy array, convert to bytes
            img = Image.fromarray(screenshot_data)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            screenshot_bytes = img_buffer.getvalue()
        else:
            # Try to handle as PIL Image
            if hasattr(screenshot_data, 'save'):
                img_buffer = io.BytesIO()
                screenshot_data.save(img_buffer, format='PNG')
                screenshot_bytes = img_buffer.getvalue()
            else:
                raise ValueError(f"Unsupported screenshot data type: {type(screenshot_data)}")
        
        # Create agent for plan generation
        plan_generation_prompt = f"""You are an expert at creating step-by-step plans for GUI automation tasks.

Task: {instruction}

Please analyze the current screenshot and create a detailed step-by-step plan to accomplish this task.

Your plan should:
1. Be specific about what to click, type, or interact with
2. Include the sequence of actions needed  
3. Be detailed enough that another agent could follow it

Respond with a numbered list of steps:
1. [Action description]
2. [Action description]
...

Only provide the plan - do not execute any actions."""
        
        # Create BaseModule agent
        plan_module = self.BaseModule(self.engine_params, self.platform)
        plan_agent = plan_module._create_agent(plan_generation_prompt)
        
        # Add message with screenshot
        plan_agent.add_message(
            f"Task: {instruction}\n\nPlease create a step-by-step plan based on the current screenshot.", 
            image_content=screenshot_bytes, 
            role="user"
        )
        
        # Create plan formatter to validate numbered list format
        def plan_formatter(response):
            """Validate that response contains a properly formatted numbered list of steps."""
            # Check for numbered list pattern
            matches = re.findall(r'^(\d+)\.', response, re.MULTILINE)
            has_numbered_list = len(matches) > 0 and all(int(num) == i for i, num in enumerate(matches, start=1))
            
            if not has_numbered_list:
                return False, "The response must be a numbered list of steps starting from 1. Example:\n1. First step\n2. Second step\n..."
            
            # Check that steps are not empty
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            step_lines = [line for line in lines if re.match(r'^\d+\.', line)]
            
            for step_line in step_lines:
                # Extract content after number and period
                content = re.sub(r'^\d+\.\s*', '', step_line).strip()
                if not content:
                    return False, "Each step must have non-empty content after the number."
            
            return True, ""
        
        # Generate the plan with format validation
        response = self.call_llm_formatted(
            plan_agent,
            [plan_formatter],
            temperature=self.engine_params.get("temperature", 1.0)
        )
        
        logger.info(f"Generated plan response: {response}")
        return {"plan": response}

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate step-by-step plans for benchmark tasks"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--provider_name", type=str, default="vmware",
        help="Virtualization provider (vmware, docker, aws, azure, gcp, virtualbox)"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")  
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)

    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )
    parser.add_argument(
        "--examples_dir", type=str, default="examples", 
        help="Examples directory name (e.g., 'examples' or 'examples_clarified')"
    )
    parser.add_argument("--result_dir", type=str, default="./results")

    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )

    # lm config
    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument("--model_temperature", type=float, default=None, help="Temperature to fix the generation model at")
    
    parser.add_argument(
        "--force_regenerate", 
        action="store_true",
        help="Force regeneration of plans even if they already exist in cache"
    )

    args = parser.parse_args()
    return args

def test_plan_generation(args: argparse.Namespace, test_all_meta: dict) -> None:
    logger.info("Plan Generation Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks for plan generation: {len(all_tasks)}")

    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": getattr(args, 'model_url', ''),
        "api_key": getattr(args, 'model_api_key', ''),
        "temperature": getattr(args, 'model_temperature', None),
    }
    with Manager() as manager:
        shared_plans = manager.dict()
        
        # Load existing plan cache if it exists
        plan_cache_file = os.path.join(
            args.result_dir,
            "plans",
            args.action_space, 
            args.observation_type,
            args.model,
            "plan_cache.json"
        )
        
        if os.path.exists(plan_cache_file):
            try:
                with open(plan_cache_file, "r", encoding="utf-8") as f:
                    existing_cache = json.load(f)
                
                # Populate shared_plans with existing cache
                for key, value in existing_cache.items():
                    shared_plans[key] = value
                
                logger.info(f"Loaded existing plan cache with {len(existing_cache)} entries from: {plan_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing plan cache: {e}")
        else:
            logger.info(f"No existing plan cache found at: {plan_cache_file}")
        
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
            
        num_envs = args.num_envs
        processes = []
        for i in range(num_envs):
            p = Process(
                target=generate_plan_for_task,
                args=(task_queue, args, shared_plans, engine_params),
                name=f"PlanGenProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started plan generation process {p.name} with PID {p.pid}")
            
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Plan generation interrupted by user.")
            for p in processes:
                if p.is_alive():
                    p.terminate()
        
        # Save updated plan cache (preserving existing entries)
        plan_cache_file = os.path.join(
            args.result_dir,
            "plans",
            args.action_space, 
            args.observation_type,
            args.model,
            "plan_cache.json"
        )
        os.makedirs(os.path.dirname(plan_cache_file), exist_ok=True)
        
        with open(plan_cache_file, "w", encoding="utf-8") as f:
            json.dump(dict(shared_plans), f, indent=2, ensure_ascii=False)
            
        logger.info(f"Updated plan cache with {len(shared_plans)} total tasks")
        logger.info(f"Plan cache saved to: {plan_cache_file}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # save args to json 
    plan_args_path = os.path.join(
        args.result_dir,
        "plans",
        args.action_space,
        args.observation_type,
        args.model,
        "plan_generation_args.json",
    )
    os.makedirs(os.path.dirname(plan_args_path), exist_ok=True)
    with open(plan_args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_plan_generation(args, test_all_meta)