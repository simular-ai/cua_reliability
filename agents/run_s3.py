"""Agent S3 runner for OSWorld evaluation.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import signal
import time
import shutil
from multiprocessing import Process, Manager, current_process, Queue

# Add repo root and osworld to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, 'osworld'))

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mask_time import mask_ubuntu_time

from dotenv import load_dotenv
load_dotenv()




#  Logger Configs {{{ #
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
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

def distribute_tasks(test_all_meta: dict) -> list:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks

def process_signal_handler(signum, frame, env_idx):
    logger.info(f"Process {env_idx + 1} received signal {signum}. Shutting down...")
    local_vars = frame.f_locals
    active_environments = local_vars.get('active_environments', [])
    for env in active_environments:
        if env is not None:
            try:
                logger.info(f"Process {env_idx + 1} closing environment...")
                env.close()
                logger.info(f"Process {env_idx + 1} environment closed successfully")
            except Exception as e:
                logger.error(f"Process {env_idx + 1} error closing environment: {e}")
    logger.info(f"Process {env_idx + 1} shutdown complete. Exiting.")
    sys.exit(0)

def check_plan_exists(args: argparse.Namespace, domain: str, example_id: str) -> tuple[bool, dict]:
    """Check if a plan exists for the given domain/example_id.
    
    Returns:
        tuple: (plan_exists, plan_data)
    """
    if not args.enable_plan or not args.plan_cache_dir:
        return False, {}
    
    plan_cache_file = os.path.join(args.plan_cache_dir, "plan_cache.json")
    
    if not os.path.exists(plan_cache_file):
        return False, {}
    
    try:
        with open(plan_cache_file, 'r', encoding='utf-8') as f:
            plan_cache = json.load(f)
        
        plan_key = f"{domain}/{example_id}"
        if plan_key in plan_cache:
            return True, plan_cache[plan_key]
        else:
            return False, {}
    except Exception as e:
        logger.error(f"Error reading plan cache file: {e}")
        return False, {}

def check_plan_feedback_exists(args: argparse.Namespace, domain: str, example_id: str) -> tuple[bool, dict]:
    """Check if plan feedback exists for the given domain/example_id.
    
    Returns:
        tuple: (feedback_exists, feedback_data)
    """
    if not args.enable_plan_feedback or not args.plan_feedback_file:
        return False, {}
    
    if not os.path.exists(args.plan_feedback_file):
        return False, {}
    
    try:
        feedback_cache = {}
        with open(args.plan_feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    task_id = entry.get('task_id')
                    if task_id:
                        feedback_cache[task_id] = entry
        
        task_key = f"{domain}/{example_id}"
        if task_key in feedback_cache:
            return True, feedback_cache[task_key]
        else:
            return False, {}
    except Exception as e:
        logger.error(f"Error reading plan feedback cache file: {e}")
        return False, {}

def save_plan_info(result_dir: str, args: argparse.Namespace, domain: str, example_id: str, plan_data: dict = None) -> None:
    """Save plan usage information and plan content to result directory."""
    plan_info = {
        "enable_plan": args.enable_plan,
        "plan_cache_dir": args.plan_cache_dir,
        "plan_exists": bool(plan_data),
        "domain": domain,
        "example_id": example_id,
    }
    
    if plan_data:
        plan_info.update({
            "plan_file": plan_data.get("plan_file", ""),
            "instruction": plan_data.get("instruction", ""),
        })
    
    # Save plan metadata
    plan_info_file = os.path.join(result_dir, "plan_info.json")
    with open(plan_info_file, "w", encoding="utf-8") as f:
        json.dump(plan_info, f, indent=2)
    
    # Save the plan text content to result directory
    if plan_data and "plan_text" in plan_data:
        try:
            plan_content_path = os.path.join(result_dir, "plan.txt")
            with open(plan_content_path, 'w', encoding='utf-8') as f:
                f.write(plan_data["plan_text"])
            logger.info(f"Saved plan content to {plan_content_path}")
        except Exception as e:
            logger.error(f"Failed to save plan content: {e}")

def save_plan_feedback_info(result_dir: str, args: argparse.Namespace, domain: str, example_id: str, feedback_data: dict = None) -> None:
    """Save plan feedback usage information and content to result directory."""
    plan_feedback_info = {
        "enable_plan_feedback": args.enable_plan_feedback,
        "plan_feedback_file": args.plan_feedback_file,
        "feedback_exists": bool(feedback_data),
        "domain": domain,
        "example_id": example_id,
    }
    
    if feedback_data:
        plan_feedback_info.update({
            "task_id": feedback_data.get("task_id", ""),
            "trajectory_count": feedback_data.get("trajectory_count", 0),
            "success_count": feedback_data.get("success_count", 0),
            "generated_at": feedback_data.get("generated_at", ""),
        })
    
    # Save plan_feedback metadata
    plan_feedback_info_file = os.path.join(result_dir, "plan_feedback_info.json")
    with open(plan_feedback_info_file, "w", encoding="utf-8") as f:
        json.dump(plan_feedback_info, f, indent=2)
    
    # Save the plan feedback content to result directory
    if feedback_data:
        try:
            # Save analysis
            if "analysis" in feedback_data:
                analysis_path = os.path.join(result_dir, "plan_feedback_analysis.txt")
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    f.write(feedback_data["analysis"])
                logger.info(f"Saved plan_feedback analysis to {analysis_path}")
            
            # Save actionable feedback
            if "feedback" in feedback_data:
                feedback_path = os.path.join(result_dir, "plan_feedback.txt")
                with open(feedback_path, 'w', encoding='utf-8') as f:
                    f.write(feedback_data["feedback"])
                logger.info(f"Saved plan feedback to {feedback_path}")
                
            # Save complete raw response for debugging
            if "raw_response" in feedback_data:
                raw_response_path = os.path.join(result_dir, "plan_feedback_raw_response.txt")
                with open(raw_response_path, 'w', encoding='utf-8') as f:
                    f.write(feedback_data["raw_response"])
                logger.info(f"Saved plan_feedback raw response to {raw_response_path}")
        except Exception as e:
            logger.error(f"Failed to save plan_feedback content: {e}")

def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list, engine_params, engine_params_for_grounding):
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
        from gui_agents.s3.agents.agent_s import AgentS3
        from gui_agents.s3.agents.grounding import OSWorldACI
        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=region,
            snapshot_name=snapshot_name,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type = "Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=True,
            client_password=getattr(args, 'client_password', ''),
        )
        grounding_agent = OSWorldACI(
            env=env,
            platform="linux",
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=engine_params_for_grounding,
            width=args.screen_width,
            height=args.screen_height,
        )
        agent = AgentS3(
            engine_params,
            grounding_agent,
            platform="linux",
            enable_plan=getattr(args, 'enable_plan', False),
            plan_cache_dir=getattr(args, 'plan_cache_dir', None),
            enable_plan_feedback=getattr(args, 'enable_plan_feedback', False),
            plan_feedback_file=getattr(args, 'plan_feedback_file', None),
        )
        active_environments.append(env)
        logger.info(f"Process {current_process().name} started.")
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
                    # In case args.examples_dir == "examples_clarified"
                    config_file = os.path.join(
                        args.test_config_base_dir, f"{args.examples_dir}/{domain}/{example_id}/{example_id}.json"
                    )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                # Apply environment perturbations if specified
                if getattr(args, 'perturbations', None):
                    from experiments.perturbations import get_perturbation_configs
                    perturbation_configs = get_perturbation_configs(args.perturbations)
                    example["config"] = perturbation_configs + example.get("config", [])
                    logger.info(f"[{current_process().name}] Applied perturbation set {args.perturbations}")

                instruction = example["instruction"]
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                
                # Check for plan existence and save plan info
                plan_exists, plan_data = check_plan_exists(args, domain, example_id)
                if args.enable_plan:
                    if plan_exists:
                        logger.info(f"[{current_process().name}][Plan]: Found for {domain}/{example_id}")
                    else:
                        logger.warning(f"[{current_process().name}][Plan]: Not found for {domain}/{example_id} (--enable_plan was used)")
                else:
                    logger.info(f"[{current_process().name}][Plan]: Disabled")
                
                # Save plan information to result directory
                save_plan_info(example_result_dir, args, domain, example_id, plan_data)
                
                # Check for plan feedback existence and save plan_feedback info
                feedback_exists, feedback_data = check_plan_feedback_exists(args, domain, example_id)
                if args.enable_plan_feedback:
                    if feedback_exists:
                        logger.info(f"[{current_process().name}][Plan Feedback]: Found feedback for {domain}/{example_id}")
                    else:
                        logger.warning(f"[{current_process().name}][Plan Feedback]: No feedback found for {domain}/{example_id} (--enable_plan_feedback was used)")
                else:
                    logger.info(f"[{current_process().name}][Plan Feedback]: Disabled")
                
                # Save plan_feedback information to result directory  
                save_plan_feedback_info(example_result_dir, args, domain, example_id, feedback_data)
                
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {instruction}")
                try:
                    if getattr(args, 'method', None) is not None:
                        # Create user simulator for methods that need it
                        user_sim = None
                        if args.method == 'clarify_upon_retry':
                            from experiments.user_simulator import UserSimulator
                            user_sim_engine_params = {
                                "engine_type": args.model_provider,
                                "model": getattr(args, 'user_sim_model', args.model),
                                "temperature": 1,
                            }
                            user_sim = UserSimulator(
                                example_config=example,
                                env=env,
                                engine_params=user_sim_engine_params,
                            )

                        lib_run_single.run_single_example_with_retries(
                            agent,
                            env,
                            example,
                            instruction,
                            args,
                            example_result_dir,
                            shared_scores,
                            method=args.method,
                            user_simulator=user_sim,
                            max_total_steps=getattr(args, 'max_total_steps', 100),
                            max_retries=getattr(args, 'max_retries', 3),
                        )
                    else:
                        lib_run_single.run_single_example(
                            agent,
                            env,
                            example,
                            args.max_steps,
                            instruction,
                            args,
                            example_result_dir,
                            shared_scores,
                        )
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(
                            json.dumps(
                                {"Error": f"{domain}/{example_id} - {e}"}
                            )
                        )
                        f.write("\n")
                    # Also save error info with plan status
                    error_info = {
                        "error": str(e),
                        "domain": domain,
                        "example_id": example_id,
                        "enable_plan": args.enable_plan,
                        "plan_exists": plan_exists if 'plan_exists' in locals() else False,
                    }
                    with open(os.path.join(example_result_dir, "error_info.json"), "w") as f:
                        json.dump(error_info, f, indent=2)
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
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

def signal_handler(signum, frame):
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
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
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--mask_time", action="store_true", 
                       help="Mask time display in Ubuntu desktop screenshots")

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

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=8)
    
    # plan mode config
    parser.add_argument(
        "--enable_plan", action="store_true", 
        help="Enable pre-generated plan mode"
    )
    parser.add_argument(
        "--plan_cache_dir", type=str, default=None,
        help="Directory containing pre-generated plans"
    )
    
    # plan feedback config
    parser.add_argument(
        "--enable_plan_feedback", action="store_true",
        help="Enable plan feedback mode"
    )
    parser.add_argument(
        "--plan_feedback_file", type=str, default=None,
        help="File containing plan feedback cache (JSONL format)"
    )

    # perturbation config
    parser.add_argument(
        "--perturbations", type=str, default=None, choices=["1", "2"],
        help="Apply environment perturbation set before each task (1 or 2)"
    )

    # user simulator / retry method config
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["binary_retry", "clarify_upon_retry"],
        help="Retry method: binary_retry (generic retry signal) or clarify_upon_retry (user simulator feedback)"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Max retries per rollout (default: 3)"
    )
    parser.add_argument(
        "--max_total_steps", type=int, default=100,
        help="Total step budget across all retries (default: 100)"
    )
    parser.add_argument(
        "--user_sim_model", type=str, default="gpt-5",
        help="Azure OpenAI deployment for user simulator (default: gpt-5)"
    )

    # lm config
    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
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
    parser.add_argument("--model_temperature", type=float, default=None, help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)")
    
    # grounding model config (required for s3 agent, not needed for uitars/anthropic)
    parser.add_argument("--ground_provider", type=str, default=None, help="The provider for the grounding model")
    parser.add_argument("--ground_url", type=str, default=None, help="The URL of the grounding model")
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model", type=str, default=None, help="The model name for the grounding model"
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=1920,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=1080,
        help="Height of screenshot image after processor rescaling",
    )

    args = parser.parse_args()

    return args


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")

    engine_params = {
        "engine_type": args.model_provider,
        "model": args.model,
        "base_url": getattr(args, 'model_url', ''),
        "api_key": getattr(args, 'model_api_key', ''),
        "temperature": getattr(args, 'model_temperature', None),
    }
    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": getattr(args, 'ground_url', ''),
        "api_key": getattr(args, 'ground_api_key', ''),
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        num_envs = args.num_envs
        processes = []
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, engine_params, engine_params_for_grounding),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores, engine_params, engine_params_for_grounding),
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        scores = list(shared_scores)
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            path = os.path.join(example_path, file)
                            if os.path.isdir(path):
                                import shutil
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    # save args to json in result_dir/action_space/observation_type/model/args.json
    path_to_args = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    test(args, test_file_list)
