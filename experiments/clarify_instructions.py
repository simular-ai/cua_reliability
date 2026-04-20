#!/usr/bin/env python3
"""
Script to clarify OSWorld evaluation instructions by combining JSON configs,
cached files, and evaluator function implementations to generate improved instructions.

Usage:
    # Process all examples listed in a test JSON file (parallel):
    python experiments/clarify_instructions.py evaluation_examples/test_all.json --model gpt-5 --output-dir evaluation_examples/examples_clarified

    # Process a single example JSON file:
    python experiments/clarify_instructions.py evaluation_examples/examples/chrome/abc123/abc123.json --model gpt-5

    # Convert existing vlm_prompt.json files to readable .txt:
    python experiments/clarify_instructions.py --convert-prompts
"""

import argparse
import json
import os
import sys
import ast
import inspect
import time
import re
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from PIL import Image
import pandas as pd
import base64
import openai
from openai import AzureOpenAI

# Base paths (relative to repo root)
_REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = str(_REPO_ROOT / "evaluation_examples" / "examples")
CACHE_DIR = str(_REPO_ROOT / "cache")
EVALUATORS_DIR = str(_REPO_ROOT / "osworld" / "desktop_env" / "evaluators")
OUTPUT_DIR = str(_REPO_ROOT / "evaluation_examples" / "examples_clarified")  # default, overridden by --output-dir

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",  # Use the latest API version
)


def find_function_in_files(func_name: str, search_dir: str) -> Optional[str]:
    """Find function implementation in Python files."""
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Parse the AST to find function definitions
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                                # Extract the function source code
                                lines = content.split('\n')
                                start_line = node.lineno - 1
                                end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                                func_source = '\n'.join(lines[start_line:end_line])
                                return func_source
                    except SyntaxError:
                        continue
                        
                except Exception as e:
                    continue
    return None


def get_file_content_for_llm(file_path: str) -> Optional[Dict[str, Any]]:
    """Get file content in LLM-friendly format."""
    try:
        # Get file extension and size
        ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        # Skip very large files (>10MB)
        if file_size > 10 * 1024 * 1024:
            return {'type': 'metadata', 'content': f'File too large ({file_size} bytes)', 'format': ext}
        
        # Handle images
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            return {
                'type': 'image_url', 
                'image_url': {'url': f'data:image/{ext[1:]};base64,{img_data}'}
            }
        
        # Handle text files
        elif ext in ['.txt', '.json', '.xml', '.csv', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {'type': 'text', 'text': f'File content ({ext}):\\n{content}'}
        
        # Handle spreadsheets (convert to text)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            csv_content = df.to_csv(index=False)
            return {'type': 'text', 'text': f'Spreadsheet data ({ext}):\\n{csv_content}'}
        
        # Other files - just metadata
        else:
            return {'type': 'metadata', 'content': f'File format {ext} not displayable ({file_size} bytes)'}
            
    except Exception as e:
        return {'type': 'metadata', 'content': f'Error reading file: {str(e)}'}


def extract_cache_files(example_config: Dict[str, Any], example_id: str) -> List[Dict[str, str]]:
    """Extract information about cache files referenced in the example."""
    cache_files = []
    cache_path = os.path.join(CACHE_DIR, example_id)
    
    # Get files from config downloads
    if 'config' in example_config:
        for config_item in example_config['config']:
            if config_item.get('type') == 'download':
                files = config_item.get('parameters', {}).get('files', [])
                for file_info in files:
                    url = file_info.get('url', '')
                    vm_path = file_info.get('path', '')
                    # Try to find the cached version
                    cache_filename = os.path.basename(vm_path)
                    cache_file_path = os.path.join(cache_path, cache_filename)
                    if os.path.exists(cache_file_path):
                        file_content = get_file_content_for_llm(cache_file_path)
                        cache_files.append({
                            'type': 'input_file',
                            'url': url,
                            'vm_path': vm_path,
                            'cache_path': cache_file_path,
                            'description': f'Input file downloaded to {vm_path}',
                            'content': file_content
                        })
    
    # Get expected/ground truth files from evaluator
    if 'evaluator' in example_config:
        evaluator = example_config['evaluator']
        
        # Handle expected files
        if 'expected' in evaluator:
            expected = evaluator['expected']
            if isinstance(expected, dict):
                expected_list = [expected]
            elif isinstance(expected, list):
                expected_list = [item for item in expected if isinstance(item, dict)]
            else:
                expected_list = []  # Skip non-dict/non-list expected values
                
            for exp in expected_list:
                if isinstance(exp, dict) and exp.get('type') == 'cloud_file':
                    url = exp.get('path', '')
                    dest = exp.get('dest', '')
                    # Ensure dest is a string, not a list
                    if isinstance(dest, list):
                        dest = dest[0] if dest else ''
                    if dest:  # Only proceed if dest is a non-empty string
                        cache_file_path = os.path.join(cache_path, dest)
                        if os.path.exists(cache_file_path):
                            file_content = get_file_content_for_llm(cache_file_path)
                            cache_files.append({
                                'type': 'ground_truth_file', 
                                'url': url,
                                'cache_path': cache_file_path,
                                'description': f'Expected output/ground truth file: {dest}',
                                'content': file_content
                            })
    
    return cache_files


def get_func_implementations(func_spec: Union[str, List[str]]) -> str:
    """Get function implementations for single or multiple functions."""
    if isinstance(func_spec, str):
        func_names = [func_spec]
    else:
        func_names = func_spec
    
    implementations = []
    for func_name in func_names:
        if func_name == "infeasible":
            implementations.append("# Function: infeasible\n# This indicates the task is marked as impossible/infeasible")
            continue
            
        impl = find_function_in_files(func_name, EVALUATORS_DIR)
        if impl:
            implementations.append(f"# Function: {func_name}\n{impl}")
        else:
            implementations.append(f"# Function: {func_name}\n# Implementation not found")
    
    return "\n\n" + "="*80 + "\n\n".join(implementations)


def generate_clarification_prompt(example_config: Dict[str, Any], cache_files: List[Dict[str, str]], 
                                func_implementations: str) -> Dict[str, Any]:
    """Generate OpenAI format messages for VLM clarification."""
    
    # Base text prompt
    base_prompt = f"""You are improving the clarity of an OSWorld evaluation task instruction. The current instruction is vague and doesn't provide enough detail for an AI agent to know exactly what to do based on the evaluation criteria.

The goal is to make MINIMAL clarifications while preserving the natural, human-like tone. Do NOT rewrite as step-by-step instructions. Only clarify ambiguous details that are checked by the evaluator but missing from the instruction.

## Current Task Configuration:
```json
{json.dumps(example_config, indent=2)}
```

## Evaluator Function Implementation(s):
```python
{func_implementations}
```

## Task:

Analyze the instruction: "{example_config.get('instruction', '')}"

Identify what the evaluator checks that isn't clear from the instruction, then provide a minimally clarified version that:
1. **Keeps the natural human tone** - should sound like a human said it
2. **Only adds missing details** that the evaluator checks for
3. **Specifies exact file names/formats/locations** if the evaluator expects them
4. **Does NOT become a step-by-step procedure** - stay conversational

Additional criteria:
1. The clarified instruction will replace the original instruction - make sure to retain essential details to avoid losing current context (e.g. do not omit links).
2. For tasks that have exact file checks, be very careful in accidentally adding or preventing information to the instruction that would cause it to no longer match the evaluator's criteria.

Format your response as:
<thoughts>
[Think through what's ambiguous and what the evaluator actually checks for]
</thoughts>

<answer>
[The minimally clarified instruction in natural language]
</answer>
"""
    
    # Create OpenAI format messages
    content = [{'type': 'text', 'text': base_prompt}]
    
    # Add file contents
    for cache_file in cache_files:
        if 'content' in cache_file and cache_file['content']:
            if cache_file['content']['type'] in ['text', 'image_url']:
                content.append(cache_file['content'])
            else:
                # Add metadata as text
                content.append({
                    'type': 'text', 
                    'text': f"\\n**{cache_file['type'].title()}**: {cache_file['description']}\\nFile info: {cache_file['content']['content']}"
                })
    
    return {'messages': [{'role': 'user', 'content': content}]}


# Rate limiting for API calls
api_call_lock = threading.Lock()
last_api_call = {'time': 0}

def call_vlm_for_clarification(openai_messages: Dict[str, Any], model: str, max_retries: int = 3, rate_limit: float = 5.0) -> Optional[str]:
    """Call OpenAI VLM to get clarified instruction with rate limiting."""

    with api_call_lock:
        time_since_last = time.time() - last_api_call['time']
        if time_since_last < rate_limit:
            time.sleep(rate_limit - time_since_last)
        last_api_call['time'] = time.time()
    
    # GPT-5 has a much larger context window, so minimal truncation needed
    content = openai_messages["messages"][0]["content"]
    if isinstance(content, list):
        total_chars = sum(len(item.get('text', '')) for item in content if item.get('type') == 'text')
        if total_chars > 5000000:  # Very high limit since GPT-5 can handle much more
            print(f"  ⚠️ Prompt extremely long ({total_chars} chars), doing minimal truncation...")
            # Only truncate the largest individual text blocks
            new_content = []
            for item in content:
                if item.get('type') == 'text':
                    text = item.get('text', '')
                    if len(text) > 1000000:  # Only truncate truly massive blocks
                        new_content.append({
                            'type': 'text',
                            'text': text[:500000] + "\\n\\n... [TRUNCATED DUE TO EXTREME LENGTH] ..."
                        })
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            openai_messages["messages"][0]["content"] = new_content
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=openai_messages["messages"],
                max_completion_tokens=128000,
                temperature=1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️ API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    """Extract thoughts and answer from VLM response that uses <thoughts> and <answer> tags."""
    try:
        # Extract thoughts section
        thoughts = full_response.split("<thoughts>")[-1].split("</thoughts>")[0].strip()
        
        # Extract answer section  
        answer = full_response.split("<answer>")[-1].split("</answer>")[0].strip()
        
        return answer, thoughts
    except Exception as e:
        return full_response, ""


def parse_clarified_instruction(response_text: str) -> Optional[str]:
    """Extract clarified instruction from VLM response using thoughts/answer tags."""
    if not response_text:
        return None
    
    # Try to parse thoughts/answer format first
    answer, thoughts = split_thinking_response(response_text)
    if answer and answer != response_text:  # Successfully extracted answer
        return answer.strip()
    
    # Fallback: return the whole response
    return response_text.strip()


def openai_messages_to_readable_text(openai_messages: Dict[str, Any]) -> str:
    """Convert OpenAI messages format to human-readable text."""
    readable_text = ""
    
    for message in openai_messages.get('messages', []):
        role = message.get('role', 'unknown')
        content = message.get('content', [])
        
        readable_text += f"=== {role.upper()} MESSAGE ===\n\n"
        
        if isinstance(content, str):
            readable_text += content + "\n\n"
        elif isinstance(content, list):
            for item in content:
                if item.get('type') == 'text':
                    readable_text += item.get('text', '') + "\n\n"
                elif item.get('type') == 'image_url':
                    readable_text += f"[IMAGE: {item.get('image_url', {}).get('url', 'N/A')[:100]}...]\n\n"
        
    return readable_text.strip()


def save_clarified_example(example_config: Dict[str, Any], clarified_instruction: str, 
                         relative_path: str, openai_messages: Dict[str, Any], vlm_response: str):
    """Save the clarified example in domain-organized structure with debugging metadata."""
    example_id = example_config.get('id', 'unknown')
    original_instruction = example_config.get('instruction', '')
    
    # Create domain/example_id directory structure
    domain_dir = os.path.dirname(relative_path)
    example_dir = os.path.join(OUTPUT_DIR, domain_dir, example_id)
    os.makedirs(example_dir, exist_ok=True)
    
    # Update the instruction in the config
    updated_config = example_config.copy()
    updated_config['instruction'] = clarified_instruction
    
    # Save the clarified JSON
    example_json_path = os.path.join(example_dir, f"{example_id}.json")
    with open(example_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_config, f, indent=2)
    
    # Save debugging metadata
    prompt_path = os.path.join(example_dir, "vlm_prompt.txt")
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(openai_messages_to_readable_text(openai_messages))
    
    response_path = os.path.join(example_dir, "vlm_response.txt")
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(vlm_response)
    
    # Save before/after instructions for easy comparison
    before_path = os.path.join(example_dir, "instruction_before.txt")
    with open(before_path, 'w', encoding='utf-8') as f:
        f.write(original_instruction)
    
    after_path = os.path.join(example_dir, "instruction_after.txt")
    with open(after_path, 'w', encoding='utf-8') as f:
        f.write(clarified_instruction)
    
    return example_json_path


def convert_existing_json_prompts_to_txt():
    """Convert existing vlm_prompt.json files to vlm_prompt.txt for better readability."""
    converted_count = 0
    
    if not os.path.exists(OUTPUT_DIR):
        print("No examples_clarified directory found.")
        return
    
    print(f"Converting existing vlm_prompt.json files to readable vlm_prompt.txt...")
    
    for root, dirs, files in os.walk(OUTPUT_DIR):
        if 'vlm_prompt.json' in files:
            json_path = os.path.join(root, 'vlm_prompt.json')
            txt_path = os.path.join(root, 'vlm_prompt.txt')
            
            try:
                # Read the JSON file
                with open(json_path, 'r', encoding='utf-8') as f:
                    openai_messages = json.load(f)
                
                # Convert to readable text and save
                readable_text = openai_messages_to_readable_text(openai_messages)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(readable_text)
                
                # Remove the old JSON file
                os.remove(json_path)
                converted_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed to convert {json_path}: {str(e)}")
    
    print(f"  ✅ Converted {converted_count} vlm_prompt files to readable text format")
    return converted_count


def load_test_examples(test_json_path: str) -> List[str]:
    """Load example IDs from test JSON file - handles both list and domain-organized formats."""
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        example_ids = []
        
        # Handle domain-organized format (like test_nogdrive.json)
        if isinstance(test_data, dict) and all(isinstance(v, list) for v in test_data.values()):
            for domain, ids in test_data.items():
                example_ids.extend(ids)
        # Handle simple list format
        elif isinstance(test_data, list):
            for item in test_data:
                if isinstance(item, dict) and 'id' in item:
                    example_ids.append(item['id'])
                elif isinstance(item, str):
                    example_ids.append(item)
        
        return example_ids
    except Exception as e:
        print(f"Error loading test file {test_json_path}: {str(e)}")
        return []


def find_example_json_path(example_id: str) -> Optional[str]:
    """Find the JSON file path for a given example ID."""
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for file in files:
            if file == f"{example_id}.json":
                return os.path.join(root, file)
    return None


def process_single_example(json_file_path: str) -> Dict[str, Any]:
    """Process a single example JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            example_config = json.load(f)
        
        example_id = example_config.get('id', '')
        if not example_id:
            return {'error': 'No ID found in example'}
        
        # Get evaluator function(s) 
        evaluator = example_config.get('evaluator', {})
        func_spec = evaluator.get('func')
        
        if not func_spec:
            return {'error': 'No evaluator function specified'}
        
        # Get function implementations
        func_implementations = get_func_implementations(func_spec)
        
        # Extract cache files
        cache_files = extract_cache_files(example_config, example_id)
        
        # Generate clarification messages for OpenAI API
        openai_messages = generate_clarification_prompt(example_config, cache_files, func_implementations)
        
        # Get relative path for saving
        relative_path = os.path.relpath(json_file_path, EXAMPLES_DIR)
        
        return {
            'example_id': example_id,
            'original_instruction': example_config.get('instruction', ''),
            'evaluator_func': func_spec,
            'cache_files': cache_files,
            'openai_messages': openai_messages,
            'example_config': example_config,
            'relative_path': relative_path,
            'status': 'success'
        }
        
    except Exception as e:
        return {'error': f'Failed to process {json_file_path}: {str(e)}'}


def is_example_already_processed(example_id: str) -> bool:
    """Check if an example has already been successfully processed."""
    for domain in os.listdir(EXAMPLES_DIR):
        example_dir = os.path.join(OUTPUT_DIR, domain, example_id)
        json_file = os.path.join(example_dir, f"{example_id}.json")
        if os.path.exists(json_file):
            return True
    return False


def process_single_example_parallel(example_id: str, total_count: int, current_index: int, model: str, rate_limit: float = 5.0) -> Tuple[bool, str]:
    """Process a single example for parallel execution. Returns (success, message)."""
    try:
        # Check if already processed
        if is_example_already_processed(example_id):
            return True, f"[{current_index}/{total_count}] ⏭️ {example_id}: Already processed (cached)"
        
        # Find the JSON file for this example
        json_file_path = find_example_json_path(example_id)
        if not json_file_path:
            return False, f"[{current_index}/{total_count}] ❌ {example_id}: JSON file not found"
        
        # Process the example
        result = process_single_example(json_file_path)
        
        if 'error' in result:
            return False, f"[{current_index}/{total_count}] ❌ {example_id}: {result['error']}"
        
        # Call VLM for clarification
        vlm_response = call_vlm_for_clarification(result['openai_messages'], model, rate_limit=rate_limit)

        if not vlm_response:
            return False, f"[{current_index}/{total_count}] ❌ {example_id}: VLM call failed"
        
        # Parse clarified instruction
        clarified_instruction = parse_clarified_instruction(vlm_response)
        
        if not clarified_instruction:
            return False, f"[{current_index}/{total_count}] ❌ {example_id}: Failed to parse clarified instruction"
        
        # Save results
        output_path = save_clarified_example(
            result['example_config'], 
            clarified_instruction, 
            result['relative_path'],
            result['openai_messages'],
            vlm_response
        )
        
        # Extract domain from output path for cleaner display
        domain = output_path.split('/')[-3] if '/' in output_path else 'unknown'
        
        return True, f"[{current_index}/{total_count}] ✅ {example_id} ({domain})"
        
    except Exception as e:
        return False, f"[{current_index}/{total_count}] ❌ {example_id}: Unexpected error - {str(e)}"


def main():
    """Main function to process examples specified in test JSON."""
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Clarify OSWorld evaluation instructions using a VLM.")
    parser.add_argument("input", nargs="?", help="Test JSON file or single example JSON path")
    parser.add_argument("--model", default="gpt-5", help="Azure OpenAI deployment name (default: gpt-5)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for clarified examples")
    parser.add_argument("--max-workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--rate-limit", type=float, default=5.0, help="Minimum seconds between API calls (default: 5)")
    parser.add_argument("--convert-prompts", action="store_true", help="Convert existing vlm_prompt.json files to .txt")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir

    # Handle conversion mode
    if args.convert_prompts:
        convert_existing_json_prompts_to_txt()
        return

    if not args.input:
        parser.print_help()
        return

    input_path = args.input
    max_workers = args.max_workers
    model = args.model
    rate_limit = args.rate_limit

    # Check if it's a test JSON file or single example
    if input_path.endswith('.json') and not input_path.startswith('OSWorld/'):
        # Process examples from test JSON
        print(f"Loading examples from test file: {input_path}")
        example_ids = load_test_examples(input_path)
        
        if not example_ids:
            print("No example IDs found in test file")
            return
        
        # Check for already processed examples
        already_processed = [eid for eid in example_ids if is_example_already_processed(eid)]
        to_process = [eid for eid in example_ids if not is_example_already_processed(eid)]
        
        print(f"Found {len(example_ids)} total examples")
        print(f"Already processed (cached): {len(already_processed)}")
        print(f"To process: {len(to_process)}")
        print(f"Using {max_workers} parallel workers")
        print(f"Output directory: {OUTPUT_DIR}")
        print("\n" + "="*60)
        
        if not to_process:
            print("✅ All examples already processed!")
            return
        
        # Create base output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        success_count = 0
        error_count = 0
        
        # Process examples in parallel (only unprocessed ones)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(process_single_example_parallel, example_id, len(to_process), i+1, model, rate_limit): example_id
                for i, example_id in enumerate(to_process)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_id):
                example_id = future_to_id[future]
                try:
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                    print(message)
                except Exception as e:
                    error_count += 1
                    print(f"❌ Unexpected error for {example_id}: {str(e)}")
        
        print("\n" + "="*60)
        print(f"🏁 Processing complete:")
        print(f"  ⏭️ Cached: {len(already_processed)}")
        print(f"  ✅ New success: {success_count}")
        print(f"  ❌ Errors: {error_count}")
        print(f"  📁 Output: {OUTPUT_DIR}")
        total_success = len(already_processed) + success_count
        total_processed = len(example_ids)
        print(f"  📊 Overall success rate: {total_success/total_processed*100:.1f}% ({total_success}/{total_processed})")
        
    else:
        # Process single file for testing (original behavior)
        json_file = input_path
        if not json_file.endswith('.json'):
            json_file = os.path.join(EXAMPLES_DIR, json_file + '.json')
        
        print(f"Processing single file: {json_file}")
        result = process_single_example(json_file)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Example ID: {result['example_id']}")
            print(f"Original: {result['original_instruction']}")
            print(f"Evaluator: {result['evaluator_func']}")
            print(f"Cache files: {len(result['cache_files'])}")
            
            # Call VLM
            print("Calling VLM...")
            vlm_response = call_vlm_for_clarification(result['openai_messages'], model, rate_limit=rate_limit)

            if vlm_response:
                clarified = parse_clarified_instruction(vlm_response)
                print(f"\nClarified: {clarified}")
                
                # Save results
                output_path = save_clarified_example(
                    result['example_config'], 
                    clarified, 
                    result['relative_path'],
                    result['openai_messages'],
                    vlm_response
                )
                print(f"Saved to: {output_path}")
            else:
                print("VLM call failed")


if __name__ == "__main__":
    main()