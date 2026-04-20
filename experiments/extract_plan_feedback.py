#!/usr/bin/env python3

"""
Plan Feedback Extraction Script

This script implements step (2) of the plan feedback approach:
1. Analyzes multiple rollouts from run.py (step 1)
2. Generates diff captions (with caching)  
3. Creates comparative feedback using LLM judge
4. Stores feedback centrally for step (3)

Usage:
    python experiments/extract_plan_feedback.py --trial-number 0 --base-results-dir results/trials --judge-model gpt-5 --judge-engine azure --temperature 1
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import datetime

# Add repo root and osworld to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, 'osworld'))

try:
    from gui_agents.s3.core.mllm import LMMAgent
    from gui_agents.s3.utils.common_utils import call_llm_formatted, split_thinking_response
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import agent evaluation utilities. Some features may be limited.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d\x1b[1;33m] \x1b[0m%(message)s"
)
logger = logging.getLogger(__name__)

def load_one_instruction(task: str, android_world=False) -> str:
    """Load instruction for a task (copied from analyze_reliability_failures.py)"""
    if android_world:
        domain, example_id = task.split("/", 1)
        config_file = f"evaluation_examples/examples/{domain}/{example_id}.json"
    else:
        domain, example_id = task.split("/", 1)
        config_file = f"evaluation_examples/examples/{domain}/{example_id}.json"
        
        # Handle clarified examples structure
        if not os.path.exists(config_file):
            config_file = f"evaluation_examples/examples_clarified/{domain}/{example_id}/{example_id}.json"
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)
        return example["instruction"]
    except Exception as e:
        logger.error(f"Failed to load instruction for {task}: {e}")
        return f"Task: {task}"

def find_completed_tasks(rollout_dirs: List[str]) -> Dict[str, List[Dict]]:
    """Find all tasks that have completed results across rollout directories with deep recursive search"""
    logger.info("🔍 Finding completed tasks across rollout directories...")
    
    all_tasks = {}  # task_id -> list of results
    
    for rollout_dir in rollout_dirs:
        logger.info(f"  📂 Scanning {rollout_dir}...")
        
        if not os.path.exists(rollout_dir):
            logger.warning(f"    ⚠️  Directory not found: {rollout_dir}")
            continue
            
        # Recursively walk through all directories to find result.txt and traj.jsonl pairs
        for root, dirs, files in os.walk(rollout_dir):
            # Check if this directory contains both result.txt and traj.jsonl
            if "result.txt" in files and "traj.jsonl" in files:
                try:
                    result_file = os.path.join(root, "result.txt")
                    traj_file = os.path.join(root, "traj.jsonl")
                    
                    # Read the score
                    with open(result_file, 'r') as f:
                        score = float(f.read().strip())
                    
                    # Extract task identification from path
                    # The path structure should be: .../domain/task_id/...
                    rel_path = os.path.relpath(root, rollout_dir)
                    path_parts = rel_path.split(os.sep)
                    
                    # Known OSWorld domains to look for
                    known_domains = {
                        'chrome', 'libreoffice_calc', 'libreoffice_impress', 'libreoffice_writer',
                        'multi_apps', 'os', 'thunderbird', 'vlc', 'vs_code', 'gimp'
                    }
                    
                    # Find domain and task_id in the path
                    domain = None
                    task_id = None
                    
                    # First try to find known OSWorld domains
                    for i in range(len(path_parts) - 1):
                        potential_domain = path_parts[i]
                        potential_task_id = path_parts[i + 1]
                        
                        # If we find a known domain, use it
                        if potential_domain in known_domains:
                            domain = potential_domain
                            task_id = potential_task_id
                            break
                    
                    # Fallback: use last two non-empty path segments
                    if not domain or not task_id:
                        if len(path_parts) >= 2:
                            domain = path_parts[-2]
                            task_id = path_parts[-1]
                        else:
                            logger.warning(f"    ⚠️  Could not extract domain/task from path: {rel_path}")
                            continue
                    
                    # Skip onboard tasks
                    if task_id == "onboard":
                        continue
                    
                    full_task_id = f"{domain}/{task_id}"
                    
                    if full_task_id not in all_tasks:
                        all_tasks[full_task_id] = []
                    
                    all_tasks[full_task_id].append({
                        "rollout_dir": rollout_dir,
                        "task_dir": root,
                        "score": score,
                        "success": score >= 0.9
                    })
                    
                    logger.debug(f"    ✓ Found task: {full_task_id} in {root} (score: {score})")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️  Error processing {root}: {e}")
    
    # Filter to tasks that have multiple rollouts
    multi_rollout_tasks = {k: v for k, v in all_tasks.items() if len(v) > 1}
    
    logger.info(f"✅ Found {len(all_tasks)} total tasks, {len(multi_rollout_tasks)} with multiple rollouts")
    
    # Log some examples of found tasks for debugging
    if multi_rollout_tasks:
        example_tasks = list(multi_rollout_tasks.keys())[:3]
        for task_id in example_tasks:
            rollout_count = len(multi_rollout_tasks[task_id])
            logger.info(f"  📋 Example: {task_id} ({rollout_count} rollouts)")
    
    return multi_rollout_tasks



def load_ordered_fact_captions(task_dir: str, fact_captions_filename: str = "fact_captions.jsonl") -> List[str]:
    """Load existing fact captions in proper screenshot order (assumes facts already generated)"""
    fact_captions_file = os.path.join(task_dir, fact_captions_filename)
    
    if not os.path.exists(fact_captions_file):
        logger.warning(f"  ⚠️  No fact captions found at {fact_captions_file}")
        return []
    
    # Load all captions
    captions_data = []
    with open(fact_captions_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                captions_data.append(data)
    
    # Sort by screenshot number
    captions_data.sort(key=lambda x: x.get('screenshot_num', 0))
    
    # Extract the fact_answer text
    ordered_captions = []
    for data in captions_data:
        caption = f"Step {data.get('screenshot_num', 'unknown')}: {data['fact_answer']}"
        ordered_captions.append(caption)
    
    return ordered_captions

def load_trajectory_reasoning(task_dir: str) -> List[str]:
    """Load agent reasoning/thoughts from trajectory JSONL file"""
    traj_file = os.path.join(task_dir, "traj.jsonl")
    
    if not os.path.exists(traj_file):
        return []
    
    reasoning_steps = []
    with open(traj_file, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                step_data = json.loads(line)
                
                # Extract reasoning from trajectory fields
                reasoning_text = ""
                
                # Check for plan field (Agent S3 format)
                if "plan" in step_data:
                    reasoning_text = step_data["plan"]
                # Check for action.raw_response field (Sonnet format)
                elif "action" in step_data and isinstance(step_data["action"], dict) and "raw_response" in step_data["action"]:
                    reasoning_text = step_data["action"]["raw_response"]
                
                # Add the reasoning text if found
                if reasoning_text.strip():
                    reasoning_steps.append(reasoning_text.strip())
    
    return reasoning_steps

def extract_run_data(run_path: str) -> Optional[Dict]:
    """Extract trajectory data (fact captions and reasoning) from a run directory"""
    try:
        # Use the same loading logic as the main pipeline
        fact_captions = load_ordered_fact_captions(run_path)  # Model not needed for loading
        trajectory_reasoning = load_trajectory_reasoning(run_path)
        
        # Check if we got any data
        if not fact_captions and not trajectory_reasoning:
            logger.warning(f"No fact captions or reasoning found in {run_path}")
            return None
            
        logger.debug(f"  📊 Extracted {len(fact_captions)} fact captions and {len(trajectory_reasoning)} reasoning steps from {run_path}")
        
        return {
            'fact_captions': fact_captions,
            'trajectory_reasoning': trajectory_reasoning,
            'trajectory_length': len(fact_captions) if fact_captions else len(trajectory_reasoning)
        }
        
    except Exception as e:
        logger.error(f"Error extracting run data from {run_path}: {e}")
        return None

def find_historical_success(base_results_dir: str, trial_number: int, task_id: str) -> Optional[Dict]:
    """Find the most recent successful run for a task from previous trials using deep search"""
    # Search from most recent trial backwards
    for trial_idx in range(trial_number - 1, -1, -1):
        trial_dir = os.path.join(base_results_dir, f"trial{trial_idx}_runs")
        
        if not os.path.exists(trial_dir):
            continue
            
        logger.debug(f"    🔍 Searching trial {trial_idx} for historical success of {task_id}")
        
        # Look through rollout directories in this trial
        for run_dir in os.listdir(trial_dir):
            run_path = os.path.join(trial_dir, run_dir)
            if not os.path.isdir(run_path):
                continue
                
            # Use deep search like find_completed_tasks - recursively walk through directories
            for root, dirs, files in os.walk(run_path):
                # Check if this directory contains both result.txt and traj.jsonl
                if "result.txt" in files and "traj.jsonl" in files:
                    try:
                        result_file = os.path.join(root, "result.txt")
                        
                        # Read the score
                        with open(result_file, 'r') as f:
                            score = float(f.read().strip())
                        
                        # Only consider successes
                        if score <= 0.0:
                            continue
                            
                        # Extract task identification from path using same logic as find_completed_tasks
                        rel_path = os.path.relpath(root, run_path)
                        path_parts = rel_path.split(os.sep)
                        
                        # Known OSWorld domains to look for
                        known_domains = {
                            'chrome', 'libreoffice_calc', 'libreoffice_impress', 'libreoffice_writer',
                            'multi_apps', 'os', 'thunderbird', 'vlc', 'vs_code', 'gimp'
                        }
                        
                        # Find domain and task_id in the path
                        domain = None
                        extracted_task_id = None
                        
                        # First try to find known OSWorld domains
                        for i in range(len(path_parts) - 1):
                            potential_domain = path_parts[i]
                            potential_task_id = path_parts[i + 1]
                            
                            # If we find a known domain, use it
                            if potential_domain in known_domains:
                                domain = potential_domain
                                extracted_task_id = potential_task_id
                                break
                        
                        # Fallback: use last two non-empty path segments
                        if not domain or not extracted_task_id:
                            if len(path_parts) >= 2:
                                domain = path_parts[-2]
                                extracted_task_id = path_parts[-1]
                            else:
                                continue
                        
                        # Skip onboard tasks
                        if extracted_task_id == "onboard":
                            continue
                        
                        full_task_id = f"{domain}/{extracted_task_id}"
                        
                        # Check if this matches the task we're looking for
                        if full_task_id == task_id:
                            logger.debug(f"    ✅ Found historical success: {task_id} in trial {trial_idx}, score: {score}")
                            return {
                                "trial": trial_idx,
                                "run_dir": run_dir,
                                "task_dir": root,
                                "score": score,
                                "success": True
                            }
                            
                    except Exception as e:
                        logger.debug(f"    ⚠️  Error processing {root}: {e}")
                        continue
    
    logger.debug(f"    ❌ No historical success found for {task_id}")
    return None

def load_previous_feedback(base_results_dir: str, trial_number: int, task_id: str) -> Optional[Dict]:
    """Load previous feedback for a task from the most recent trial that has it"""
    for trial_idx in range(trial_number - 1, -1, -1):
        feedback_file = os.path.join(base_results_dir, f"trial{trial_idx}_feedback.jsonl")
        
        if not os.path.exists(feedback_file):
            continue
            
        try:
            with open(feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("task_id") == task_id:
                            # Add source trial information
                            entry['source_trial'] = trial_idx
                            return entry
        except Exception as e:
            logger.warning(f"Error reading feedback file {feedback_file}: {e}")
            continue
    
    # Return empty dict instead of None to simplify data processing
    return {'feedback': '', 'source_trial': None}




def prepare_plan_feedback_input(task_id: str, rollouts: List[Dict], model: str) -> Dict:
    """Prepare input data for plan_feedback analysis"""
    logger.info(f"📋 Preparing plan_feedback input for task: {task_id}")
    
    trajectories = {}
    
    for i, rollout in enumerate(rollouts):
        run_name = f"rollout_{i+1}"
        task_dir = rollout["task_dir"]
        
        # Load fact captions
        try:
            fact_captions = load_ordered_fact_captions(task_dir)
        except Exception as e:
            logger.error(f"  ❌ Failed to load fact captions for {run_name}: {e}")
            return None
        
        # Load trajectory reasoning  
        try:
            trajectory_reasoning = load_trajectory_reasoning(task_dir)
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to load trajectory reasoning for {run_name}: {e}")
            trajectory_reasoning = []
        
        trajectories[run_name] = {
            "success": rollout["success"],
            "score": rollout["score"],
            "fact_captions": fact_captions,
            "trajectory_reasoning": trajectory_reasoning,
            "task_dir": task_dir
        }
        
        status_icon = "✅" if rollout["success"] else "❌"
        logger.info(f"  📁 {run_name}: {status_icon} (score: {rollout['score']}) - {len(fact_captions)} visual facts, {len(trajectory_reasoning)} reasoning steps")
    
    return {
        "task_id": task_id,
        "trajectories": trajectories
    }

# Plan feedback analysis prompt
PLAN_FEEDBACK_ANALYSIS_PROMPT = """You are analyzing agent trajectories to provide plan feedback for improving future performance.

You are given multiple rollouts from the SAME policy on the SAME task:
- Some SUCCESSFUL rollouts (score > 0)  
- Some FAILED rollouts (score = 0)

{previous_feedback_section}

For each rollout, you will receive:
1. **Visual Changes**: Fact captions describing what changed between screenshots at each step
2. **Agent Reasoning**: The agent's internal thoughts and decision-making process at each step

Your goal is to identify what worked in successful runs versus what went wrong in failed runs, then provide actionable feedback for future attempts.

ANALYSIS APPROACH:

1. **Compare Trajectories Step-by-Step**
   - Compare both visual changes and agent reasoning between successful/failed runs
   - Identify where successful and failed runs diverge in actions AND thinking
   - Note different approaches taken by successful vs failed runs
   - Look for consistent patterns across successes and failures

2. **Key Success Factors**
   - What specific actions or strategies led to success?
   - What reasoning patterns did successful runs exhibit?
   - Were there critical steps that successful runs handled correctly?
   - What environmental awareness did successful runs demonstrate?

3. **Failure Analysis**  
   - What mistakes did failed runs make in actions or reasoning?
   - Were there flawed reasoning patterns or incorrect decision making?
   - Did failed runs misunderstand the task or environment?
   - Were there missed opportunities or incorrect decisions?

4. **Actionable Insights**
   - What should the agent think about differently next time?
   - What should the agent do differently next time?
   - What should the agent avoid based on failed attempts?
   - Are there general principles or heuristics that emerge?

OUTPUT FORMAT:

<analysis>
[Detailed step-by-step comparison of successful vs failed trajectories, analyzing both visual changes and agent reasoning, identifying key divergence points and patterns]
</analysis>

<feedback>
DO:
1. [Specific actionable instruction based on successful patterns]
2. [Another positive action or strategy to follow]
3. [Key behavior or approach to adopt]
...
(up to a max of 10 items and can stop earlier if not applicable)

DON'T:
1. [Specific mistake or behavior to avoid from failed runs]
2. [Common error pattern to prevent]
3. [Action or thinking pattern that leads to failure]
...
(up to a max of 10 items and can stop earlier if not applicable)

PLAN:
[Step-by-step execution plan extracted from successful runs - concrete sequence of actions that leads to success]
</feedback>"""

# All-failure with historical success prompt
ALL_FAILURE_WITH_HISTORY_PROMPT = """You are analyzing failed agent trajectories against a successful historical reference to generate corrective feedback.

SITUATION:
All current attempts failed, but we have a successful run from a previous trial to learn from.

HISTORICAL SUCCESSFUL RUN:
```
{historical_success_info}
```

CURRENT FAILED ATTEMPTS:
You will be shown multiple failed attempts from the current trial.

ANALYSIS GOAL:
Compare the failed attempts against the historical success to identify:
1. What the successful run did right that current attempts are missing
2. What systematic errors current attempts are making
3. How to guide future attempts toward the successful pattern

For each run, you will see:
1. **Visual Changes**: Fact captions describing what changed between screenshots at each step  
2. **Agent Reasoning**: The agent's internal thoughts and decision-making process at each step

ANALYSIS APPROACH:

1. **Success Pattern Analysis**
   - What made the historical run successful?
   - What key decisions, actions, or reasoning patterns led to success?
   - What environmental awareness did it demonstrate?

2. **Current Failure Analysis**
   - How do current failures differ from the successful pattern?
   - What systematic mistakes are being repeated?
   - Where do current attempts go wrong compared to the success?

3. **Corrective Strategy**
   - How can future attempts align with the successful pattern?
   - What specific changes in reasoning or actions are needed?
   - What should be avoided based on current failures?

OUTPUT FORMAT:

<analysis>
[Detailed comparison of historical success vs current failures, identifying key differences and systematic error patterns]
</analysis>

<feedback>
DO:
1. [Key strategy from the successful run to adopt]
2. [Specific action or approach that led to success]
3. [Reasoning pattern that worked in the successful case]
...
(up to a max of 10 items and can stop earlier if not applicable)

DON'T:
1. [Specific mistake in current attempts to avoid]
2. [Error pattern that differs from successful approach]
3. [Action or thinking that leads to failure vs success]
...
(up to a max of 10 items and can stop earlier if not applicable)

PLAN:
[Step-by-step execution plan extracted from the historical successful run - concrete sequence of actions to replicate]
</feedback>"""

# All-failure no history prompt  
ALL_FAILURE_NO_HISTORY_PROMPT = """You are analyzing failed agent trajectories to generate diagnostic feedback when no successful examples exist.

SITUATION:
All attempts have failed, and there are no successful runs from previous trials to reference.

CURRENT FAILED ATTEMPTS:
You will be shown multiple failed attempts. Your goal is to identify the most promising approach and provide guidance to help future attempts succeed.

ANALYSIS GOAL:
Since all attempts failed, focus on:
1. Which failure showed the most promise/progress
2. What systematic errors are preventing success
3. What fundamental strategies might lead to success

For each failed run, you will see:
1. **Visual Changes**: Fact captions describing what changed between screenshots at each step
2. **Agent Reasoning**: The agent's internal thoughts and decision-making process at each step

ANALYSIS APPROACH:

1. **Comparative Failure Analysis**
   - Which attempt made the most progress before failing?
   - What different approaches were tried?
   - What common error patterns appear across attempts?

2. **Root Cause Identification**
   - What fundamental issues are causing failures?
   - Are failures due to misunderstanding, poor execution, or environment issues?
   - What capabilities seem to be missing?

3. **Diagnostic Strategy**
   - What should future attempts focus on first?
   - What basic principles or approaches might help?
   - What specific errors should be avoided?

OUTPUT FORMAT:

<analysis>
[Detailed analysis of failure patterns, identifying the most promising approach and systematic issues preventing success]
</analysis>

<feedback>
DO:
1. [Basic strategy that might lead to success]
2. [Fundamental approach to try based on most promising failure]
3. [Key capability or awareness to develop]
...
(up to a max of 10 items and can stop earlier if not applicable)

DON'T:
1. [Common error pattern to avoid across all attempts]
2. [Systematic mistake that prevents success]
3. [Flawed approach or reasoning to prevent]
...
(up to a max of 10 items and can stop earlier if not applicable)
</feedback>"""

def format_historical_success_info(historical_data: Dict, historical_success: Dict) -> str:
    """Format historical success information for the prompt"""
    trial_num = historical_success['trial']
    score = historical_success['score']
    
    info = f"TRIAL {trial_num} SUCCESS (Score: {score}):\n\n"
    
    # Add visual changes if available
    if historical_data.get('fact_captions'):
        info += "Visual Changes:\n"
        for i, caption in enumerate(historical_data['fact_captions'], 1):
            info += f"Step {i}: {caption}\n"
        info += "\n"
    
# Add agent reasoning if available  
        if historical_data.get('trajectory_reasoning'):
            info += "Agent Reasoning:\n"
            for i, reasoning in enumerate(historical_data['trajectory_reasoning'], 1):
                info += f"Step {i}: {reasoning}\n"
    
    return info

def get_scenario_name(num_successes: int, historical_success: Optional[Dict], previous_feedback: Optional[str]) -> str:
    """Get a descriptive name for the plan_feedback scenario"""
    if num_successes > 0:
        if previous_feedback:
            return "iterative_plan_feedback"
        else:
            return "standard_plan_feedback"
    elif historical_success:
        return "all_failure_with_history"
    else:
        return "all_failure_diagnostic"

def plan_feedback_formatter(response: str) -> Tuple[bool, str]:
    """Custom formatter for plan feedback responses"""
    try:
        # Check for required tags
        if "<analysis>" not in response or "</analysis>" not in response:
            return False, "Missing <analysis> tags"
        
        if "<feedback>" not in response:# or "</feedback>" not in response:
            return False, "Missing <feedback> tags"
        
        # Extract sections
        analysis = response.split("<analysis>")[1].split("</analysis>")[0].strip()
        feedback = response.split("<feedback>")[1].strip()#.split("</feedback>")[0].strip()
        
        if not analysis:
            return False, "Empty analysis section"
            
        if not feedback:
            return False, "Empty feedback section"
        
        # Check for DO/DON'T structure
        if "DO:" not in feedback or "DON'T:" not in feedback:
            return False, "Feedback must contain both DO: and DON'T: sections"
        
        # Note: PLAN: section is optional (not present in all_failure_no_history scenario)
        
        return True, ""
    except Exception as e:
        return False, f"Error parsing response: {str(e)}"

async def generate_plan_feedback(task_data: Dict, judge: LMMAgent, previous_feedback: Optional[str] = None, historical_success: Optional[Dict] = None) -> Dict:
    """Generate plan feedback for a task with multi-trial scenario handling"""
    logger.info(f"🤖 Generating plan feedback for: {task_data['task_id']}")
    
    # Get task instruction
    instruction = load_one_instruction(task_data['task_id'])
    
    # Determine scenario and select appropriate prompt
    successes = [name for name, traj in task_data["trajectories"].items() if traj["success"]]
    failures = [name for name, traj in task_data["trajectories"].items() if not traj["success"]]
    
    if len(successes) > 0:
        # Scenario 1: Mixed outcomes - compare successes vs failures
        if previous_feedback and previous_feedback.strip():
            # Include previous feedback section for iterative plan_feedback
            previous_feedback_section = f"""PREVIOUS FEEDBACK:
```
{previous_feedback}
```

ANALYSIS GOAL:
The previous feedback was applied, but the results show it may need refinement. You need to:
1. Identify which parts of the previous feedback were helpful vs harmful
2. Understand why the feedback led to the current outcomes  
3. Refine the feedback based on new evidence
"""
            scenario = "iterative_plan_feedback"
            logger.info(f"  📝 Using iterative plan_feedback (refining previous feedback)")
        else:
            # No previous feedback section for standard plan_feedback
            previous_feedback_section = ""
            scenario = "standard_plan_feedback"
            logger.info(f"  📝 Using standard plan_feedback (first feedback)")
            
        system_prompt = PLAN_FEEDBACK_ANALYSIS_PROMPT.format(previous_feedback_section=previous_feedback_section)
            
    elif historical_success:
        # Scenario 2: All failures but we have historical success
        # Format historical success info
        historical_info = format_historical_success_info(
            historical_success.get('run_data', {}), 
            historical_success
        )
        system_prompt = ALL_FAILURE_WITH_HISTORY_PROMPT.format(historical_success_info=historical_info)
        scenario = "all_failure_with_history"
        logger.info(f"  📝 Using all-failure with historical success from trial {historical_success['trial']}")
        
    else:
        # Scenario 3: All failures with no historical success
        system_prompt = ALL_FAILURE_NO_HISTORY_PROMPT
        scenario = "all_failure_diagnostic"
        logger.info(f"  📝 Using all-failure diagnostic mode (no historical success)")
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    messages.append({
        "role": "user",
        "content": f"Task: {instruction}\n\nAnalyze these rollouts from the same policy:"
    })
    
    # Add trajectory data with both visual diffs and agent reasoning
    for run_name, traj in task_data["trajectories"].items():
        status = "SUCCESSFUL" if traj["success"] else "FAILED" 
        
        content = f"\n=== {run_name.upper()} ({status}) ===\nScore: {traj['score']}\n\n"
        
        # Add visual fact captions if available
        if traj["fact_captions"]:
            content += "Visual Changes (Fact Captions):\n"
            for caption in traj["fact_captions"]:
                content += f"- {caption}\n"
            content += "\n"
        
        # Add agent reasoning if available
        if traj["trajectory_reasoning"]:
            content += "Agent Reasoning:\n"
            for reasoning in traj["trajectory_reasoning"]:
                content += f"- {reasoning}\n"
        
        messages.append({"role": "user", "content": content})
    
    messages.append({
        "role": "user", 
        "content": "Please analyze these trajectories and provide actionable plan feedback."
    })
    
    # Generate response with retries
    logger.info(f"  🔄 Calling LLM for {task_data['task_id']}...")
    response = await asyncio.to_thread(
        call_llm_formatted,
        judge,
        [plan_feedback_formatter], 
        messages=messages
    )
    
    # Parse response with better error handling
    try:
        if "<analysis>" not in response or "</analysis>" not in response:
            raise ValueError(f"Response missing <analysis> tags. Response preview: {response[:500]}...")
        if "<feedback>" not in response or "</feedback>" not in response:
            raise ValueError(f"Response missing <feedback> tags. Response preview: {response[:500]}...")
            
        analysis = response.split("<analysis>")[1].split("</analysis>")[0].strip()
        feedback = response.split("<feedback>")[1].split("</feedback>")[0].strip()
        
        if not analysis.strip():
            raise ValueError("Parsed analysis is empty")
        if not feedback.strip():
            raise ValueError("Parsed feedback is empty")
            
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse LLM response for {task_data['task_id']}: {e}")
        logger.error(f"Raw response: {response}")
        raise
    
    return {
        "task_id": task_data["task_id"],
        "analysis": analysis,
        "feedback": feedback,
        "scenario": scenario,
        "trajectory_count": len(task_data["trajectories"]),
        "success_count": len(successes),
        "failure_count": len(failures),
        "has_previous_feedback": previous_feedback is not None,
        "has_historical_success": historical_success is not None,
        "generated_at": datetime.datetime.now().isoformat(),
        "raw_response": response,
        "llm_messages": messages,
        "system_prompt": system_prompt
    }

async def save_metadata(result: Dict, metadata_dir: str, write_lock: asyncio.Lock) -> None:
    """Save LLM prompt and response metadata for debugging"""
    if not metadata_dir:
        return
        
    try:
        async with write_lock:
            # Create task-specific directory
            task_id = result["task_id"]
            safe_task_id = task_id.replace("/", "_")
            task_metadata_dir = os.path.join(metadata_dir, safe_task_id)
            os.makedirs(task_metadata_dir, exist_ok=True)
            
            # Save scenario info
            scenario_info = {
                "task_id": task_id,
                "scenario": result["scenario"],
                "trajectory_count": result["trajectory_count"],
                "success_count": result["success_count"],
                "failure_count": result["failure_count"],
                "has_previous_feedback": result["has_previous_feedback"],
                "has_historical_success": result["has_historical_success"],
                "generated_at": result["generated_at"]
            }
            
            with open(os.path.join(task_metadata_dir, "scenario_info.json"), "w") as f:
                json.dump(scenario_info, f, indent=2)
            
            # Save system prompt
            with open(os.path.join(task_metadata_dir, "system_prompt.txt"), "w") as f:
                f.write(result["system_prompt"])
            
            # Save formatted conversation
            with open(os.path.join(task_metadata_dir, "conversation.txt"), "w") as f:
                f.write(f"=== PLAN FEEDBACK GENERATION ===\n")
                f.write(f"Task: {task_id}\n")
                f.write(f"Scenario: {result['scenario']}\n")
                f.write(f"Generated: {result['generated_at']}\n\n")
                
                for i, message in enumerate(result["llm_messages"]):
                    role = message["role"].upper()
                    content = message["content"]
                    f.write(f"=== {role} MESSAGE {i+1} ===\n")
                    f.write(f"{content}\n\n")
            
            # Save raw LLM response  
            with open(os.path.join(task_metadata_dir, "llm_response.txt"), "w") as f:
                f.write(result["raw_response"])
            
            # Save structured response
            structured_response = {
                "analysis": result["analysis"],
                "feedback": result["feedback"]
            }
            with open(os.path.join(task_metadata_dir, "structured_response.json"), "w") as f:
                json.dump(structured_response, f, indent=2)
                
            logger.debug(f"  💾 Saved metadata for {task_id} to {task_metadata_dir}")
            
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to save metadata for {task_id}: {e}")

async def process_single_task(
    task_id: str, 
    rollouts: List[Dict], 
    judge: LMMAgent, 
    semaphore: asyncio.Semaphore,
    output_file: str,
    write_lock: asyncio.Lock,
    model: str,
    trial_number: int,
    base_results_dir: str,
    metadata_dir: Optional[str] = None,
    force: bool = False
) -> Optional[Dict]:
    """Process a single task for plan feedback with multi-trial support"""
    async with semaphore:
        try:
            # Check if already processed (unless forced)
            if not force and os.path.exists(output_file):
                # Check if this task already exists in output
                try:
                    with open(output_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                existing = json.loads(line)
                                if existing.get("task_id") == task_id:
                                    logger.info(f"  ⏭️  Task {task_id} already processed, skipping")
                                    return None
                except Exception:
                    pass  # Continue processing if can't check existing
            
            # Classify current rollouts
            successes = [r for r in rollouts if r["success"]]
            failures = [r for r in rollouts if not r["success"]]
            
            logger.info(f"  📊 Task {task_id}: {len(successes)} successes, {len(failures)} failures")
            
            # Load previous feedback if it exists
            previous_feedback = load_previous_feedback(base_results_dir, trial_number, task_id)
            has_previous_feedback = previous_feedback.get('feedback') and previous_feedback.get('feedback').strip()
            
            if has_previous_feedback:
                logger.info(f"  📖 Found previous feedback from trial {previous_feedback.get('source_trial', 'unknown')}")
            
            # Handle different scenarios
            if len(successes) > 0:
                # Check if ALL rollouts are successful 
                if len(failures) == 0:
                    if has_previous_feedback:
                        # All successful + have previous feedback - reuse it
                        logger.info(f"  🔄 All rollouts successful, reusing previous feedback from trial {previous_feedback.get('source_trial', 'unknown')}")
                        
                        result = {
                            "task_id": task_id,
                            "analysis": previous_feedback.get("analysis", ""),
                            "feedback": previous_feedback.get("feedback", ""),
                            "scenario": "all_success_reused",
                            "trajectory_count": len(rollouts),
                            "success_count": len(successes),
                            "failure_count": len(failures),
                            "has_previous_feedback": True,
                            "has_historical_success": False,
                            "generated_at": datetime.datetime.now().isoformat(),
                            "raw_response": f"[REUSED FROM TRIAL {previous_feedback.get('source_trial', 'unknown')}] " + previous_feedback.get("raw_response", ""),
                            "llm_messages": [],
                            "system_prompt": f"[REUSED FROM TRIAL {previous_feedback.get('source_trial', 'unknown')}]",
                            "reused_from_trial": previous_feedback.get('source_trial')
                        }
                        
                        # Save result
                        async with write_lock:
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(result) + "\n")
                        
                        # Save metadata if requested
                        if metadata_dir:
                            await save_metadata(result, metadata_dir, write_lock)
                        
                        logger.info(f"  ✅ Reused feedback for {task_id}")
                        return result
                    else:
                        # All successful + no previous feedback - write empty feedback entry
                        logger.info(f"  📝 All rollouts successful with no previous feedback - writing empty feedback entry")
                        
                        # Create empty feedback result 
                        result = {
                            "task_id": task_id,
                            "analysis": "All rollouts were successful with no previous feedback to refine.",
                            "feedback": "",
                            "scenario": "all_success_no_feedback",
                            "trajectory_count": len(rollouts),
                            "success_count": len(successes),
                            "failure_count": len(failures),
                            "has_previous_feedback": False,
                            "has_historical_success": False,
                            "generated_at": datetime.datetime.now().isoformat(),
                            "raw_response": "[EMPTY FEEDBACK - ALL SUCCESS]",
                            "llm_messages": [],
                            "system_prompt": "[EMPTY FEEDBACK - ALL SUCCESS]"
                        }
                        
                        # Save result
                        async with write_lock:
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(result) + "\n")
                        
                        # Save metadata if requested
                        if metadata_dir:
                            await save_metadata(result, metadata_dir, write_lock)
                        
                        logger.info(f"  ✅ Wrote empty feedback for {task_id}")
                        return result
                
                # Scenario 1: Mixed outcomes - generate new feedback
                logger.info(f"  ✅ Mixed outcomes - generating plan feedback")
                rollouts_to_use = rollouts
                historical_success = None
                
            elif len(failures) > 0:
                # All failures - check for historical success
                historical_success = find_historical_success(base_results_dir, trial_number, task_id)
                
                if historical_success:
                    logger.info(f"  📚 All failures + historical success from trial {historical_success['trial']}")
                else:
                    logger.info(f"  🔍 All failures + no historical success - diagnostic mode")
                rollouts_to_use = failures
                        
            else:
                logger.info(f"  ⏭️  Task {task_id} has no eligible rollouts (single success), skipping")
                return None
            
            # Prepare input data
            task_data = prepare_plan_feedback_input(task_id, rollouts_to_use, model)
            if task_data is None:
                logger.error(f"  ❌ Failed to prepare input for {task_id}")
                return None
                
            # Load historical success data if needed
            if historical_success:
                # Load the trajectory data for the historical success
                historical_run_data = extract_run_data(historical_success['task_dir'])  # Changed from run_path to task_dir
                if historical_run_data:
                    historical_success['run_data'] = historical_run_data
            
            # Generate feedback  
            result = await generate_plan_feedback(
                task_data, 
                judge, 
                previous_feedback=previous_feedback.get('feedback') if has_previous_feedback else None,
                historical_success=historical_success
            )
            
            # Save result
            async with write_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + "\n")
            
            # Save metadata if requested
            if metadata_dir:
                await save_metadata(result, metadata_dir, write_lock)
            
            logger.info(f"  ✅ Generated {result['scenario']} feedback for {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {task_id}: {type(e).__name__}: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error(f"  📍 Traceback: {traceback.format_exc()}")
            return None

def find_rollout_directories(base_results_dir: str, trial_number: int) -> List[str]:
    """Automatically find rollout directories for a given trial"""
    trial_runs_dir = os.path.join(base_results_dir, f"trial{trial_number}_runs")
    
    if not os.path.exists(trial_runs_dir):
        logger.error(f"❌ Trial runs directory not found: {trial_runs_dir}")
        return []
    
    rollout_dirs = []
    for item in os.listdir(trial_runs_dir):
        item_path = os.path.join(trial_runs_dir, item)
        if os.path.isdir(item_path):
            rollout_dirs.append(item_path)
    
    logger.info(f"📂 Found {len(rollout_dirs)} rollout directories in {trial_runs_dir}")
    return rollout_dirs

async def main():
    parser = argparse.ArgumentParser(description="Extract plan feedback from rollouts")
    
    parser.add_argument("--trial-number", type=int, required=True,
                       help="Current trial number (e.g., 0, 1, 2)")
    parser.add_argument("--base-results-dir", required=True,
                       help="Base results directory containing all trials")
    parser.add_argument("--output", "-o", 
                       help="Output file for feedback cache (default: trial{N}_feedback.jsonl)")
    parser.add_argument("--judge-model", required=True,
                       help="Model for generating feedback")
    parser.add_argument("--judge-engine", required=True,
                       help="Engine for judge model")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Maximum concurrent analyses")
    parser.add_argument("--force", action="store_true",
                       help="Force re-analysis of existing tasks")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for LLM judge (default: 0.7)")
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if not args.output:
        args.output = os.path.join(args.base_results_dir, f"trial{args.trial_number}_feedback.jsonl")
    
    # Auto-generate metadata directory
    metadata_dir = os.path.join(args.base_results_dir, f"trial{args.trial_number}_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    logger.info("🚀 Starting plan feedback extraction...")
    logger.info(f"📊 Trial number: {args.trial_number}")
    logger.info(f"📁 Base results directory: {args.base_results_dir}")
    logger.info(f"💾 Output file: {args.output}")
    logger.info(f"📝 Metadata directory: {metadata_dir}")
    
    # Automatically find rollout directories
    rollout_dirs = find_rollout_directories(args.base_results_dir, args.trial_number)
    if not rollout_dirs:
        logger.error("❌ No rollout directories found")
        sys.exit(1)
    
    logger.info(f"📂 Rollout directories: {rollout_dirs}")
    
    # Find completed tasks
    completed_tasks = find_completed_tasks(rollout_dirs)
    
    if not completed_tasks:
        logger.info("❌ No tasks with multiple rollouts found")
        sys.exit(0)
    
    logger.info(f"📊 Found {len(completed_tasks)} tasks with multiple rollouts")
    
    # Initialize judge
    judge = LMMAgent(engine_params={
        "model": args.judge_model,
        "engine_type": args.judge_engine,
        "temperature": args.temperature,
        "max_tokens": 12228
    }, system_prompt=PLAN_FEEDBACK_ANALYSIS_PROMPT)
    
    # Create semaphore and lock
    semaphore = asyncio.Semaphore(args.max_concurrent)
    write_lock = asyncio.Lock()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process tasks
    tasks = []
    for task_id, rollouts in completed_tasks.items():
        task = process_single_task(
            task_id, rollouts, judge, semaphore, 
            args.output, write_lock, args.judge_model, 
            args.trial_number, args.base_results_dir, 
            metadata_dir, args.force
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    logger.info(f"🔄 Processing {len(tasks)} tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    failed = sum(1 for r in results if isinstance(r, Exception))
    skipped = len(results) - successful - failed
    
    logger.info(f"✅ Completed! {successful} successful, {skipped} skipped, {failed} failed")
    
    if successful > 0:
        logger.info(f"💾 Plan feedback saved to: {args.output}")
        logger.info(f"📈 Ready for step (3) - rerunning with feedback in context!")

if __name__ == "__main__":
    asyncio.run(main())