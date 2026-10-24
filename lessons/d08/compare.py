import torch
import math
from config import (
    BASELINE_PATH,
    INTERRUPT_PATH,
    RESUME_PATH,
    BASELINE_JSON_PATH,
    INTERRUPT_JSON_PATH,
    RESUME_JSON_PATH,
    RESUME_CHKPT_PATH
)

checkpoint_baseline = torch.load(
    BASELINE_PATH,
    map_location = "cpu",
    weights_only=True
)

checkpoint_interrupt = torch.load(
    INTERRUPT_PATH,
    map_location = "cpu",
    weights_only=True
)

checkpoint_resume = torch.load(
    RESUME_PATH,
    map_location = "cpu",
    weights_only=True
)

# print(checkpoint_baseline["optimizer_state"].keys())
# print(checkpoint_baseline["optimizer_state"]["state"][0].keys())
# print(checkpoint_baseline["optimizer_state"]["state"][0]["step"])
# print(checkpoint_baseline["optimizer_state"]["param_groups"][0].keys())
# print(checkpoint_baseline["optimizer_state"]["param_groups"][0]["lr"])
# print(checkpoint_baseline["optimizer_state"]["param_groups"][0]["weight_decay"])
# print(checkpoint_baseline["optimizer_state"]["param_groups"][0]["betas"])

# print(checkpoint_interrupt["optimizer_state"].keys())
# print(checkpoint_interrupt["optimizer_state"]["state"][0].keys())
# print(checkpoint_interrupt["optimizer_state"]["state"][0]["step"])
# print(checkpoint_interrupt["optimizer_state"]["param_groups"][0].keys())
# print(checkpoint_interrupt["optimizer_state"]["param_groups"][0]["lr"])
# print(checkpoint_interrupt["optimizer_state"]["param_groups"][0]["weight_decay"])
# print(checkpoint_interrupt["optimizer_state"]["param_groups"][0]["betas"])

# print(checkpoint_resume["optimizer_state"].keys())
# print(checkpoint_resume["optimizer_state"]["state"][0].keys())
# print(checkpoint_resume["optimizer_state"]["state"][0]["step"])
# print(checkpoint_resume["optimizer_state"]["param_groups"][0].keys())
# print(checkpoint_resume["optimizer_state"]["param_groups"][0]["lr"])
# print(checkpoint_resume["optimizer_state"]["param_groups"][0]["weight_decay"])
# print(checkpoint_resume["optimizer_state"]["param_groups"][0]["betas"])

def compare_batch_order(ckpt_1, ckpt_2):
    same_order = (ckpt_1["batch_order"] == ckpt_2["batch_order"])
    print(f"same_batch_order = {same_order}")

    return same_order

def compare_model_parameters(ckpt_1, ckpt_2):
    same_parameters = []
    for name, parameter_tensor_a in ckpt_1["model_state"].items():
        parameter_tensor_b = ckpt_2["model_state"][name]
        same_parameters.append(torch.allclose(parameter_tensor_a, parameter_tensor_b, rtol = 1e-5, atol = 1e-6))

    return all(same_parameters)

def compare_optimizer_state(ckpt_1, ckpt_2):
    opt_a = ckpt_1["optimizer_state"]
    opt_b = ckpt_2["optimizer_state"]
    same_groups = (opt_a["param_groups"] == opt_b["param_groups"])
    same_opt_parameters = []
    print(f"same_param_groups = {same_groups}")

    for param_id, parameter_state_a in opt_a["state"].items():
        parameter_state_b = opt_b["state"][param_id]
        for param_state_id, parameter_state_a_parameter_a in parameter_state_a.items():
            parameter_state_b_parameter_b = parameter_state_b[param_state_id]
            same_opt_parameters.append(torch.allclose(parameter_state_a_parameter_a, parameter_state_b_parameter_b, rtol = 1e-5, atol = 1e-6))

    return same_groups, all(same_opt_parameters)

def compare_history(ckpt_1, ckpt_2):
    same_history = []
    history_a = ckpt_1["history"]
    history_b = ckpt_2["history"]
    for history_id, history_record_a in enumerate(history_a):
        history_record_b = history_b[history_id]
        for record_id, record_a in history_record_a.items():
            record_b = history_record_b[record_id]
            same = math.isclose(
                record_a, 
                record_b,
                rel_tol = 1e-5,
                abs_tol = 1e-6
            )
            same_history.append(same)

    return same_history

def main():
    same_batch_order = compare_batch_order(checkpoint_baseline, checkpoint_resume)
    print(f"same_batch_order = {same_batch_order}")
    same_model_parameters = compare_model_parameters(checkpoint_baseline, checkpoint_resume)
    print(f"same_model_parameters = {same_model_parameters}")
    same_optimizer_groups, same_optimizer_parameter = compare_optimizer_state(checkpoint_baseline, checkpoint_resume)
    print(f"same_optimizer_groups = {same_optimizer_groups}; same_optimizer_parameter = {same_optimizer_parameter}")
    same_history = compare_history(checkpoint_baseline, checkpoint_resume)
    print(f"same_history = {same_history}")

if __name__ == "__main__":
    main()