
import re
import json
import copy

# BASE_SURVEY_STRUCTURE = """
# # Title: A survey of ...
# # Introduction: None.
# # Section 1: None.
#     ## Subsection 1 (if needed): None.
#     ## Subsection 2 (if needed): None.
#         ### Subsubsection 1 (if needed): None.
#         ### Subsubsection 2 (if needed): None.
#         ### ...
# # Section 2: None.
# # ...
# # Conclusion: None.
# """


class SurveyManager:
    BASE_SURVEY_STRUCTURE = {
        "title": "",
        "abstract": "",
        "introduction": {
            "content": ""
        },
        "sections": [],
        "conclusion": ""
    }
    
    def __init__(self):
        pass
  
    @staticmethod
    def parse_update_pos(update_pos):
        """
        (1) "title", "abstract", "introduction", or "conclusion"
        (2) "section-i/subsection-j/..."
        
        """
        if update_pos in ["title", "abstract", "introduction", "conclusion","plan"]:
            return update_pos
        else: 
            keys = update_pos.split("/")
            if len(keys) == 1:  # Section-?
                i = int(keys[0].lower().split("section-")[-1])
                return f"section-{i}"
            elif len(keys) == 2: # Section-?/Subsection-?
                i = int(keys[0].lower().split("section-")[-1])
                j = int(keys[1].lower().split("subsection-")[-1])
                return f"section-{i}/subsection-{j}"
            elif len(keys) == 3: # Section-?/Subsection-?/Subsubsection-?
                i = int(keys[0].lower().split("section-")[-1])
                j = int(keys[1].lower().split("subsection-")[-1])
                k = int(keys[2].lower().split("subsubsection-")[-1])
                return f"section-{i}/subsection-{j}/subsubsection-{k}"
            else:
                raise ValueError("unsupported update_pos keys")
         
    @staticmethod
    def _to_one_line(string):
        if isinstance(string, dict):
            if "content" in string and string["content"]:
                return SurveyManager._to_one_line(string["content"])
                # return SurveyManager._to_one_line(string["content"])
            else:
                return "[PLAN] " + string.get("plan", "").replace("\n", " ").strip()
        if not string:
            return ""
        else:
            return string#.replace("\n", " ")
    
    @staticmethod
    def convert_survey_dict_to_str(current_survey):
        string = ""
        if current_survey == {}:
            return "There is no survey."
        # title
        try:
            content = SurveyManager._to_one_line(current_survey["title"])
            string += f"# {content}\n"
        except:
            string += f"# Title: None\n"
        
        # abstract
        try:
            content = SurveyManager._to_one_line(current_survey["abstract"])
            string += f"## Abstract\n{content}\n"
        except:
            string += f"## Abstract\nNone\n"
        
        # introduction
        try:
            content = SurveyManager._to_one_line(current_survey["introduction"])
            string += f"## Introduction\n{content}\n"
        except:
            string += f"## Introduction\nNone\n"
        
        # sections
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                title_key = "name" if "name" in section else "title"
                name, content = section[title_key], SurveyManager._to_one_line(section)
                # string += f"# Section-{i+1} [{name}]: {content}\n"
                string += f"## {name}\n{content}\n"

                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        name, content = subsection[title_key], SurveyManager._to_one_line(subsection)
                        # string += f"    ## Subsection-{j+1} [{name}]: {content}\n"
                        string += f"### {name}\n{content}\n"

                        if "subsubsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsubsections"]):
                                name, content = subsubsection[title_key], SurveyManager._to_one_line(subsubsection)
                                # string += f"        ### Subsubsection-{k+1} [{name}]: {content}\n"
                                string += f"#### {name}\n{content}\n"
        
        
        # conclusion
        try:
            content = SurveyManager._to_one_line(current_survey["conclusion"])
            string += f"## Conclusion\n{content}\n"
        except:
            string += f"## Conclusion:\nNone\n"
        
        return string
    
    @staticmethod
    def _abbr_one_line(string, abbr=True):
        if isinstance(string, dict):
            if "content" in string and string["content"]:
                return SurveyManager._abbr_one_line(string["content"], abbr=abbr)
            elif "plan" in string:
                return "[PLAN] " + string["plan"].replace("\n", " ").strip()
            else:
                return ""
        else:
            if not string:
                return ""
            else:
                if abbr and len(string) > 50:
                    return "[OK] " + string.replace("\n", " ").strip()[:50] + "..."
                else:
                    return "[OK] " + string.replace("\n", " ").strip()

    @staticmethod
    def convert_survey_dict_to_abbr_str(current_survey):
        string = ""
        if current_survey == {}:
            return "There is no survey."
        # title
        try:
            content = SurveyManager._abbr_one_line(current_survey["title"], abbr=False)
            string += f"# Title: {content}\n"
        except:
            string += f"# Title: None\n"
        # abstract
        try:
            content = SurveyManager._abbr_one_line(current_survey["abstract"], abbr=False)
            string += f"# Abstract: {content}\n"
        except:
            string += f"# Abstract: None\n"
        
        # introduction
        try:
            content = SurveyManager._abbr_one_line(current_survey["introduction"])
            string += f"# Introduction: {content}\n"
        except:
            string += f"# Introduction: None\n"
        
        # sections
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                title_key = "name" if "name" in section else "title"
                name, content = section[title_key], SurveyManager._abbr_one_line(section)
                string += f"# Section-{i+1} [{name}]: {content}\n"

                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        name, content = subsection[title_key], SurveyManager._abbr_one_line(subsection)
                        string += f"    ## Subsection-{j+1} [{name}]: {content}\n"

                        if "subsubsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsubsections"]):
                                name, content = subsubsection[title_key], SurveyManager._abbr_one_line(subsubsection)
                                string += f"        ### Subsubsection-{k+1} [{name}]: {content}\n"
        
        # conclusion
        try:
            content = SurveyManager._abbr_one_line(current_survey["conclusion"])
            string += f"# Conclusion: {content}\n"
        except:
            string += f"# Conclusion: None\n"
        
        return string

    @staticmethod
    def update_one_section(sections, i, content):
        # i -= 1
        if i >= 0 and i <= (len(sections)-1):
            sections[i]["content"] = content
            return True
        else:
            # print("update fail!")
            return False
    
    @staticmethod
    def update_current_survey(current_survey, answer) -> bool:
        """
        update_pos: "section-i/subsection-j/subsubsection-k"
        """
        # if answer == {}:
        #     return True
        try:
            update_pos, content = answer["update"], answer["content"]

            if update_pos == "plan":
                # current_survey = content
                if current_survey == {}:
                    for k,v in content.items():
                        current_survey[k] = copy.deepcopy(v)
                else:
                    return False
            elif update_pos in ["conclusion", "abstract"]:
                if update_pos not in current_survey:
                    # print("update fail!")
                    return False
                current_survey[update_pos] = content

            elif update_pos == "introduction":
                if update_pos not in current_survey:
                    # print("update fail!")
                    return False
                current_survey[update_pos] = {"content": content}

            else:
                keys = update_pos.split("/")
                if len(keys) == 1:  # Section-?
                    i = int(keys[0].lower().split("section-")[-1])-1
                    return SurveyManager.update_one_section(current_survey["sections"], i, content)

                elif len(keys) == 2: # Section-?/Subsection-?
                    i = int(keys[0].lower().split("section-")[-1])-1
                    j = int(keys[1].lower().split("subsection-")[-1])-1
                    try:
                        return SurveyManager.update_one_section(current_survey["sections"][i]["subsections"], j,  content)
                    except:
                        # print("update fail!")
                        return False

                elif len(keys) == 3: # Section-?/Subsection-?/Subsubsection-?
                    i = int(keys[0].lower().split("section-")[-1])-1
                    j = int(keys[1].lower().split("subsection-")[-1])-1
                    k = int(keys[2].lower().split("subsubsection-")[-1])-1
                    try:
                        return SurveyManager.update_one_section(current_survey["sections"][i]["subsections"][j]["subsubsections"], k, content)  # 禁用第四级
                    except:
                        # print("update fail!")
                        return False
                else:
                    # print("update fail!")
                    # print("unsupported update_pos keys")
                    return False
                    # raise ValueError("unsupported update_pos keys")
        except:
            # print("update fail!")
            return False
            # print("answer is not a valid json object.")
            # print(answer)
            # raise ValueError("answer is not a valid json object.")
        return True
    

from prompts import *
class PromptManger:
    system_prompt = SYSTEM_PROMPT_0415_BUFFER
    user_prompt_v0 = USER_PROMPT_v0_0424_BUFFER
    user_prompt = USER_PROMPT_0415_BUFFER





class BufferManager:
    """
        Used to manage prompts/responses generated during the Rollout phase, providing data support for subsequent training.
        batch_rollout_data = [
            {
                query (or env_id): # Uniquely identifies a query or environment, [input parameter].
                *running_id: # Uniquely identifies a single rollout. For cases where a query or environment is repeated multiple times, the query can be the same, but running_id will not repeat.
                state: { # Indicates whether the process is finished.
                    "score": 0.0,
                    "done": True / False
                    "current_survey": dict  # Structured data.
                }    
                trajectory: [  # Organizes all data into a multi-turn interaction format.
                    {
                        step: int, 0~?, # The first step, usually includes some init_info or plan.
                        original_response: str, The raw output from the model, which may have various formatting issues.
                        answer_thought:  str, # Encapsulated using the <think>...</think> block.
                        answer: {
                            "original_str": str
                            "update": str,
                            "name": str,
                            "content": str,
                            "inclusions": list, # Extracted independently?
                        }
                        tool_call_thought:  str, # Encapsulated using the <think>...</think> block.
                        tool_call: {
                            "original_str": str, # Encapsulated using the <tool_call>...</tool_call> block, used for tool invocation. In the survey setting, it is either "done" to end the task or "search".
                            "tool_name": str  # done or search.
                            "keywords": list[str], Extracted search keywords from tool_call, otherwise none.
                        }
                        *papers: list[str], # Top-n papers retrieved via the search engine. Required if using the Agent-Summary-1 for collaborative optimization; otherwise, not needed.
                        cites: list[str], # References cited by the model, which may include multiple citations.
                        summarys: list[str], # Summaries of papers generated using Agent-Summary-1. Must include BIBKEY.
                        *prompt_for_generator: str, # The prompt input to the generator at the current step. Required if using Agent-Summary-2 for generation and collaborative optimization; otherwise, not needed.
                    },
                    ...
                    
                ]
                
            },
            ...
        ]
        
        """
    def __init__(self, prompts, repeat_n: int=1):
        # self.config = config
        self.step = 0
        self.batch_rollout_data = []
        self.running_ids = []  # active envs
        batch_size = prompts.batch['input_ids'].size(0)
        uids = prompts.non_tensor_batch['uid']
        querys = prompts.non_tensor_batch['raw_prompt'].copy()
        ground_truths = prompts.non_tensor_batch['ground_truth']
        # print(querys)
        new_querys = []
        for i_batch in range(batch_size):
            raw_prompt_i_batch = querys[i_batch][-1]["content"]
            new_querys.append(raw_prompt_i_batch)
        querys = new_querys
        
        assert len(querys) == len(uids)
        for query, uid, ground_truth in zip(querys, uids, ground_truths):
            
            now_survey = {}
            
            for _ in range(repeat_n):
                self.batch_rollout_data.append({
                    "query": query,
                    "uid": uid,
                    "state": {
                        # "score": 0.0, # only for debug 
                        # "format_score": None,   # will update at last step
                        "done": False,
                        "current_survey": {}
                    },
                    "trajectory": [],
                    "history_messages": [],
                })        
    
    @staticmethod
    def _build_system_prompt():
        prompt = PromptManger.system_prompt
        return prompt
    @staticmethod
    def _build_user_prompt_v0(query, current_survey):
        # query 
        prompt = PromptManger.user_prompt_v0.replace("<user_query>", query)
        
        # add template
        prompt = prompt.replace("<init_survey>", SurveyManager.convert_survey_dict_to_abbr_str(current_survey))
        return prompt
        
    @staticmethod
    def _build_user_prompt(query, current_survey, trajs):
        last_traj = trajs[-1]
        # query 
        prompt = PromptManger.user_prompt.replace("<user_query>", query)
        
        # add current survey
        prompt = prompt.replace("<current_survey>", SurveyManager.convert_survey_dict_to_abbr_str(current_survey))
        
        # current plan
        if last_traj["tool_call_thought"] == "":
            prompt = prompt.replace("<last_step_thought>", "Your last thought is not available, please give new plan")
        else:
            prompt = prompt.replace("<last_step_thought>", last_traj["tool_call_thought"])
        prompt = prompt.replace("<last_step_tool_call>", json.dumps(last_traj["tool_call"]))
    
        # summarys
        for traj in reversed(trajs):
            if len(traj["summarys"]) > 0:
                break
        summary_num = len(traj["summarys"])
           
        if summary_num == 0:
            prompt = prompt.replace("<summarys>", "There is no result.")
        else:
            prompt = prompt.replace("<summarys>", f"There are {summary_num} results:\n\n" + "\n\n".join(traj["summarys"]))
            
        return prompt
    
    @staticmethod
    def _build_user_prompt_force_correct(query, current_survey, trajs):
        if current_survey == {}:
            # gen plan
            now_section = "plan"
            # trajs[-1]["tool_call_thought"] = "Next I will provide the plan. "
        else:
            now_section = ""
            if isinstance(current_survey["abstract"],dict) and "content" not in current_survey["abstract"]:
                now_section = "abstract"
            elif "content" not in current_survey["introduction"]:
                now_section = "introduction"
            elif "sections" in current_survey:
                for section in current_survey["sections"]:
                    if "content" not in section:
                        now_section = "section-{}".format(current_survey["sections"].index(section) + 1)
                        break
                    elif "subsections" in section:
                        for subsection in section["subsections"]:
                            if "content" not in subsection:
                                now_section = "section-{}/subsection-{}".format(
                                    current_survey["sections"].index(section) + 1,
                                    section["subsections"].index(subsection) + 1
                                )
                                break
                            elif "subsubsections" in subsection:
                                for subsubsection in subsection["subsubsections"]:
                                    if "content" not in subsubsection:
                                        now_section = "section-{}/subsection-{}/subsubsection-{}".format(
                                            current_survey["sections"].index(section) + 1,
                                            section["subsections"].index(subsection) + 1,
                                            subsection["subsubsections"].index(subsubsection) + 1
                                        )
                                        break
                                if now_section:
                                    break
                        if now_section:
                            break
            
            elif isinstance(current_survey["conclusion"],dict) and "content" not in current_survey["conclusion"]:
                now_section = "conclusion"
            else:
                trajs[-1]["tool_call_thought"] = "Next I will finalize the survey."
        if now_section != "":
            trajs[-1]["tool_call_thought"] = f"Next I will provide {now_section}"
        for traj in reversed(trajs):
            if len(traj["summarys"]) > 0:
                break
        summary_num = len(traj["summarys"])
        if now_section == "plan" and summary_num == 0:
            trajs[-1]["tool_call_thought"] = "I need to get enough information."
            
        return BufferManager._build_user_prompt(query, current_survey, trajs)
    
    @staticmethod
    def _check_finalize(query, current_survey, trajs):
        if current_survey == {}:
            # gen plan
            return False
            # trajs[-1]["tool_call_thought"] = "Next I will provide the plan. "
        else:
            now_section = ""
            if isinstance(current_survey["abstract"],dict) and "content" not in current_survey["abstract"]:
                now_section = "abstract"
            elif "content" not in current_survey["introduction"]:
                now_section = "introduction"
            elif "sections" in current_survey:
                for section in current_survey["sections"]:
                    if "content" not in section:
                        now_section = "section-{}".format(current_survey["sections"].index(section) + 1)
                        break
                    elif "subsections" in section:
                        for subsection in section["subsections"]:
                            if "content" not in subsection:
                                now_section = "section-{}/subsection-{}".format(
                                    current_survey["sections"].index(section) + 1,
                                    section["subsections"].index(subsection) + 1
                                )
                                break
                            elif "subsubsections" in subsection:
                                for subsubsection in subsection["subsubsections"]:
                                    if "content" not in subsubsection:
                                        now_section = "section-{}/subsection-{}/subsubsection-{}".format(
                                            current_survey["sections"].index(section) + 1,
                                            section["subsections"].index(subsection) + 1,
                                            subsection["subsubsections"].index(subsubsection) + 1
                                        )
                                        break
                                if now_section:
                                    break
                        if now_section:
                            break
            
            elif isinstance(current_survey["conclusion"],dict) and "content" not in current_survey["conclusion"]:
                now_section = "conclusion"
            # else:
            #     trajs[-1]["tool_call_thought"] = "Next I will finalize the survey."
        if now_section != "":
            return False
            
        return True

    # rule-based method: query, plan, paragraphs -> prompt -> thought, paragraph, action
    def build_prompt_for_generator(self):
        total_messages = []
        self.running_ids = []
        for running_id, data in enumerate(self.batch_rollout_data):
            if data["state"]["done"]:
                pass
            else:
                if len(data["trajectory"]) == 0:  # first prompt
                    user_prompt = BufferManager._build_user_prompt_v0(data["query"],
                                                                      data["state"]["current_survey"])
                else:
                    if data["trajectory"][-1]["update_success"]:
                        user_prompt = BufferManager._build_user_prompt(data["query"], 
                                                                      data["state"]["current_survey"],
                                                                      data["trajectory"])
                    else:
                        # user_prompt = data["history_messages"][-1][1]["content"]
                        user_prompt = BufferManager._build_user_prompt_force_correct(data["query"], 
                                                                      data["state"]["current_survey"],
                                                                      data["trajectory"])
                messages = [
                    {
                        "role": "system",
                        "content": BufferManager._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]
                data["history_messages"].append(messages)
                total_messages.append(messages)
                self.running_ids.append(running_id)     # update running ids
        return total_messages
    
    def update_all_scores(self, scores):
        assert len(scores) == len(self.batch_rollout_data)
        for score, log in zip(scores, self.batch_rollout_data):
            log["state"]["score"] = score
    
    def update_all_format_scores(self, scores):
        assert len(scores) == len(self.batch_rollout_data)
        for score, log in zip(scores, self.batch_rollout_data):
            log["state"]["format_score"] = score
   
    
    def update_trajectory(self, model_responses, env_feedbacks):
        """
        model_response: original_response, thought, paragraph, tool_call, format_reward
        env_feedback: done, search_keywards, abstracts, outcome_reward
        """
        assert len(self.running_ids) == len(model_responses)
        assert len(self.running_ids) == len(env_feedbacks)
        
        for running_id, response, feedback in zip(self.running_ids, model_responses, env_feedbacks):
            # update state
            self.batch_rollout_data[running_id]["state"]["done"] = feedback["done"]   # if True, finalize the task
            
            update_success = False
            if response["true"]:
                if self.batch_rollout_data[running_id]["state"]["current_survey"] != {}:
                    if len(response["answer"]) != 0:  # no empty dict or start
                        update_success = SurveyManager.update_current_survey(
                            self.batch_rollout_data[running_id]["state"]["current_survey"],
                            response["answer"])
                else:
                    # Search Then Write
                    if len(response["answer"]) != 0 and "There is no result" not in self.batch_rollout_data[running_id]["history_messages"][-1][1]["content"]:
                        update_success = SurveyManager.update_current_survey(
                            self.batch_rollout_data[running_id]["state"]["current_survey"],
                            response["answer"])
                    elif "There is no result" in self.batch_rollout_data[running_id]["history_messages"][-1][1]["content"] and len(response["answer"]) == 0:
                        update_success = True


            self.batch_rollout_data[running_id]["trajectory"].append({
                "step": self.step,
                "original_response": response["original_response"],
                "answer_thought": response["answer_thought"],
                "answer": response["answer"],
                "tool_call_thought": response["tool_call_thought"],
                "tool_call": response["tool_call"],
                "search_keywords": feedback["search_keywords"],
                "summarys": feedback["summarys"],
                "update_success": update_success and response["true"],
            })
          
            
            self.batch_rollout_data[running_id]["history_messages"][-1].append({
                "role": "assistant",
                "content": response["original_response"],
            })
            
            if self.batch_rollout_data[running_id]["state"]["done"]:
                real_done = BufferManager._check_finalize(self.batch_rollout_data[running_id]["query"],
                                                         self.batch_rollout_data[running_id]["state"]["current_survey"],
                                                         self.batch_rollout_data[running_id]["trajectory"])
                if not real_done:
                    self.batch_rollout_data[running_id]["state"]["done"] = False

    
    @staticmethod
    def match_reference(text:str):
        reg = r"\\\w*cite(?!style)\w*\{(.+?)\}"
        placeholder_reg = re.compile(r"^#\d+$")
        reg_bibkeys = re.findall(reg, text)
        bibkeys = set()
        for bibkey in reg_bibkeys:
            single_bib = bibkey.split(",")
            for bib in single_bib:
                if not placeholder_reg.match(bib):
                    bib = bib.strip()
                    if bib and bib != "*":
                        bibkeys.add(bib)

        reg = r"\\nocite{(.+?)\}"
        reg_bibkeys = re.findall(reg, text)
        for bibkey in reg_bibkeys:
            single_bib = bibkey.split(",")
            for bib in single_bib:
                if not placeholder_reg.match(bib):
                    bib = bib.strip()
                    if bib and bib != "*":
                        bibkeys.remove(bib)

        ref_key_list = list(bibkeys)
        return ref_key_list
    
    @staticmethod
    def parse_generator_response(response):
        """
        1. 解析失败： step + 1, 重新生成, 给出提示
        2. 解析成功：
            2.1 tool_call == search(keywords)  发送post请求
            2.2 tool_call == done  结束任务
            
        **standard format**
        
        Current Update:
        <think> [Your Thoughts]: str </think>
        <answer> {"update": str, "content": str}: dict </answer>
        
        Next Plan:
        <think> [Your Thoughts]: str </think>
        <tool_call> {"tool": "search", "arguments": {}}: dict</tool_call>
        """
        extracted_result = {
            "original_response": response
        }
        
        try:
            current_update = response.split("Current Update:")[-1].split("Next Plan:")[0]
        except:
            current_update = response
        
        # pattern
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        tool_pattern = r"<tool_call>(.*?)</tool_call>"

        # extract information from current_update

        think_match = re.search(think_pattern, current_update, re.DOTALL)  # 多行提取
        if think_match:
            think = think_match.group(1)
            think = think.strip()
        else:
            think = ""
        extracted_result["answer_thought"] = think
        
        answer_match = re.search(answer_pattern, current_update, re.DOTALL)  # 多行提取
        has_answer = False
        if answer_match:
            answer = answer_match.group(1)
            answer = answer.strip()
            try:
                answer = json.loads(answer)
                if not answer == {}:
                    assert isinstance(answer["update"], str)
                    answer["update"] = SurveyManager.parse_update_pos(answer["update"])
                    if answer["update"] == "plan":
                        
                        assert isinstance(answer["content"], dict)
                        plan = answer["content"]
                        assert isinstance(plan, dict)
                        plan.pop("instruction",None)
                        keys = ["abstract", "introduction", "conclusion","sections","title"]
                        for key in keys:
                            assert key in plan
                        for key in plan:
                            assert key in keys
                            if key == "sections":
                                assert isinstance(plan[key], list)
                                for section in plan[key]:
                                    assert isinstance(section, dict)
                                    assert "plan" in section
                                    assert "title" in section
                                    assert isinstance(section["plan"], str)
                                    assert isinstance(section["title"], str)
                                    assert section["title"] != "Methodology"  # 不能是Methodology，WIP
                                    if "subsections" in section:
                                        assert isinstance(section["subsections"], list)
                                        for subsection in section["subsections"]:
                                            assert isinstance(subsection, dict)
                                            assert "plan" in subsection
                                            assert "title" in subsection
                                            assert isinstance(subsection["plan"], str)
                                            assert isinstance(subsection["title"], str)
                                            if "subsubsections" in section:
                                                assert isinstance(subsection["subsubsections"], list)
                                                for subsubsection in subsection["subsubsections"]:
                                                    assert isinstance(subsubsection, dict)
                                                    assert "plan" in subsubsection
                                                    assert "title" in subsubsection
                                                    assert isinstance(subsubsection["plan"], str)
                                                    assert isinstance(subsubsection["title"], str)
                            elif key == "title":
                                assert isinstance(plan[key], str) 
                            else:
                                assert isinstance(plan[key], dict)
                                assert "plan" in plan[key]
                                if key not in ["abstract", "conclusion", "introduction"]:
                                    assert "title" in plan[key]
                    else:
                        assert isinstance(answer["content"], str)
                has_answer = True
            except:
                answer = {}
        else:
            answer = {}
        extracted_result["answer"] = answer
        
        # extract information from next_plan
        
        try:
            next_plan = response.split("Next Plan:")[1]
        except:
            try:
                next_plan = response.split("</answer>")[1]
            except:
                next_plan = response
    
        think_match = re.search(think_pattern, next_plan, re.DOTALL)  # 多行提取
        if think_match:
            think = think_match.group(1)
            think = think.strip()
        else:
            think = ""
        extracted_result["tool_call_thought"] = think
        
        tool_match = re.search(tool_pattern, next_plan, re.DOTALL)  # 多行提取
        has_tool_call = False
        if tool_match:
            tool_text = tool_match.group(1)
            tool_text = tool_text.strip()
            try:
                tool_call = json.loads(tool_text)
                assert tool_call["name"] in ["search_engine", "finalize"]
                if tool_call["name"] == "search_engine":
                    assert isinstance(tool_call["arguments"]["query"], list)
                has_tool_call = True
            except:
                tool_call = {}
        else:
            
            tool_call = {}
            
        extracted_result["tool_call"] = tool_call     

        extracted_result["true"] = has_answer and has_tool_call
        reg = r"[\u4e00-\u9fa5]"
        has_chinese = re.search(reg, response) is not None
        extracted_result["true"] = extracted_result["true"] and not has_chinese
        
        return extracted_result
  

class BufferManager_V2(BufferManager):

    def __init__(self, querys, repeat_n=1):
        # self.config = config
        self.step = 0
        self.batch_rollout_data = []
        self.running_ids = []  # active envs
        
        for uid, query in enumerate(querys):
            print("CURRENT QUERY: ", query)
            for _ in range(repeat_n):
                self.batch_rollout_data.append({
                    "query": query,
                    "uid": f"query_{uid}",
                    "state": {
                        # "score": 0.0, # only for debug 
                        # "format_score": None,   # will update at last step
                        "done": False,
                        "current_survey": {}
                    },
                    "trajectory": [],
                    "history_messages": []
                })

