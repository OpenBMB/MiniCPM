import contextlib
import io
import json
import os
import re
import sys
import traceback

import fire
from vllm import LLM, SamplingParams

max_turns = 5
system_prompt_template = """You are an AI Agent who is proficient in solve complicated task. 
Each step you should wirte executable code to fulfill user query. Any Response without code means the task is completed and you do not have another chance to submit code

You are equipped with a codeinterpreter. You can give the code and get the execution result of your code. You should use the codeinterpreter in the following format: 
<|execute_start|>
```python

<your code>

```
<|execute_end|>


WARNING:Do not use cv2.waitKey(0) cv2.destroyAllWindows()!!! Or the program will be destoried

Each round, your answer should ALWAYS use the following format(Each of your response should contain code, until you complete the task):


Analyse:(Analyse the message you received and plan what you should do)  

This Step Todo: One Subtask need to be done at this step  

Code(WARNING:MAKE SURE YOU CODE FOLLOW THE FORMAT AND WRITE CODE OR THE TASK WILL BE FAILED): 
<|execute_start|>
```python

<your code>


```
<|execute_end|>


You will got the result of your code after each step. When the code of previous subtask is excuted successfully, you can write and excuet the code for next subtask
When all the code your write are executed and you got the code result that can fulfill the user query, you should summarize the previous analyse process and make a formal response to user, The response should follow this format:
WARNING:MAKE SURE YOU GET THE CODE EXECUTED RESULT THAT FULFILLED ALL REQUIREMENT OF USER BEFORE USE "Finished"
Finished: <Answer to user query>

Some notice:
1. When you want to draw a plot, use plt.savefig() and print the image path in markdown format instead of plt.show()
2. Save anything to ./output folder
3. End the process whenever you complete the task, When you do not have Action(Code), Use: Finished: <summary the analyse process and make response>
4. Do not ask for user input in your python code.
"""

def execute_code(code):

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Note here we simplely imitate notebook output.
    # if you want to run more complex tasks, try to use nbclient to run python code
    lines = code.strip().split('\n')
    last_expr = lines[-1].strip()

    if '=' in last_expr:
        value = last_expr.split('=')[0].strip()
        code += f"\nprint({value})"
    
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        try:
            # execute code here
            exec(code)
        except Exception as e:
            return {'output': stdout_capture.getvalue(), 'error': str(e)}
    
    return {'output': stdout_capture.getvalue(), 'error': stderr_capture.getvalue()}

class DemoLLM:
    def __init__(self, model_path):
        # Initialize default sampling parameters
        params_dict = {
            "n": 1,
            "best_of": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.02,
            "temperature": 1.0,
            "top_p": 0.85,
            "top_k": -1,
            "use_beam_search": False,
            "length_penalty": 1.0,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 300,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        
        # Create a SamplingParams object
        self.sampling_params = SamplingParams(**params_dict)
        
        # Initialize the language model
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True
        )

    def apply_template(self, messages):
        """Formats messages into a prompt string for the LLM."""
        formatted_messages = [
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in messages
        ]
        formatted_messages.append("<|im_start|>assistant\n")
        return ''.join(formatted_messages)

    def generate(self, messages):
        """Generates a response from the LLM based on the input messages."""
        raw_input = self.apply_template(messages)
        response = self.llm.generate(raw_input, self.sampling_params)
        if response:
            return response[0].outputs[0].text
        return None

def extract_code(text):
    """ Extracts Python code blocks from the given text. """
    # Define a regular expression pattern to match Python code blocks
    pattern = r'```python\s+(.*?)\s+```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return matches

def process(model_path):
    """
    Processes interactions with the DemoLLM using provided model path.

    Args:
        model_path (str): The path to the language model directory.
    """

    # Initialize the language model
    llm = DemoLLM(model_path)

    # Define initial messages
    messages = [
        {"role": "system", "content": system_prompt_template},
        {"role": "user", "content": "2 的 100 次方是多少？"},
    ]

    for index in range(max_turns):
        print(f"Turn {index+1} start...")

        # Generate response from the LLM
        raw_resp = llm.generate(messages)
        print(f"Raw response: {raw_resp}")

        # Check if the response contains the termination keyword
        if "Finished" in raw_resp:
            break

        # Extract code from the raw response
        code_list = extract_code(raw_resp)

        if not code_list:
            break

        # Execute the extracted code
        code_str = code_list[-1]
        run_result = execute_code(code_str)
        executor_response = run_result['output'] if run_result['error'] == "" else run_result['error']
        print(f"Code execution result: {run_result}")

        # Append the execution result to the messages
        messages.append({"role": "user", "content": executor_response})


if __name__ == "__main__":
    fire.Fire(process)