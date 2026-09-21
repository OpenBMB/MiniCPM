import os
import json
from argparse import ArgumentParser


def normalize_value(value) -> object:
    if isinstance(value, str):
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            return value.lower().strip()
    return value

def validate_prediction(pred_json):
    return (
        pred_json is not None 
    )

def evaluate_function_calls(test_data: list, eval_data: list) :
    stats = {
        "total": 0,
        "function_correct": 0,
        "argument_name_correct": 0,
        "argument_value_correct": 0,
        "invalid_gt": 0,
        "invalid_pred": 0
    }

    function_accuracy_errors = []
    argument_name_accuracy_errors = []
    argument_value_accuracy_errors = []
    
    for i in range(len(test_data)):
        stats["total"] += 1
        try:
            pred_data = test_data[i]

            gt_data = eval_data[i]
        except:
            breakpoint()

        if not validate_prediction(pred_data): 
            stats["invalid_pred"] += 1

            continue
        
        if isinstance(pred_data,list):
            find = False
            for data in pred_data:
                if data["name"] == gt_data["name"]:
                    pred_data = data
                    find = True
                    break
            if not find:
                pred_data = pred_data[0]
                
        if pred_data["name"] == gt_data["name"]:
            stats["function_correct"] += 1
            
            pred_args = pred_data.get("arguments", {})
            gt_args = gt_data.get("arguments", {})
            
            try:
                if set(pred_args.keys()) == set(gt_args.keys()):
                    stats["argument_name_correct"] += 1

                    all_values_match = True
                    for key in gt_args.keys():
                        normalized_pred = normalize_value(pred_args[key])
                        normalized_gt = normalize_value(gt_args[key])
                        
                        if normalized_pred != normalized_gt:
                            all_values_match = False
                            argument_value_accuracy_errors.append({"prediction": pred_data, "ground_truth": gt_data})
                            break
                    
                    if all_values_match:
                        stats["argument_value_correct"] += 1
                else:
                    argument_name_accuracy_errors.append({"prediction": pred_data, "ground_truth": gt_data})
            except Exception as e:
                print(e)
                continue
        else:
            function_accuracy_errors.append({"prediction": repr(pred_data["name"]), "ground_truth": repr(gt_data["name"])})


    return {
        "function_accuracy": round(stats["function_correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
        "argument_name_accuracy": round(stats["argument_name_correct"] / stats["total"], 4) if stats["function_correct"] > 0 else 0.0,
        "argument_value_accuracy": round(stats["argument_value_correct"] / stats["total"], 4) if stats["argument_name_correct"] > 0 else 0.0,
        "total_samples": stats["total"],
        "function_correct": stats["function_correct"],
        "argument_name_correct": stats["argument_name_correct"],
        "argument_value_correct": stats["argument_value_correct"],
        "invalid_ground_truth": stats["invalid_gt"],
        "invalid_predictions": stats["invalid_pred"]
    }

gt_example = [
    {
      "name": "getWeather",
      "arguments": {
        "city": "杭州",
        "extensions": "all"
      }
    }
]

if __name__ == "__main__":
    argument_parsr = ArgumentParser()
    argument_parsr.add_argument("--input_path",type=str,required=True)
    args = argument_parsr.parse_args()
    with open(args.input_path,"r") as f:
        data = json.load(f)
    print(evaluate_function_calls(test_data=[data[-1].get("function_call")],eval_data=gt_example))