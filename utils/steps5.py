# -*- coding: utf-8 -*-
import enum


# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

class CotTagBase(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    ACTION = "ACTION:"

class PromptManager5step(object):
                     
    def __init__(self, history_adaptive=False):
        # Intialize subtask history
        self.history_adaptive = history_adaptive
        self.history_idx = len(CotTagBase) - 1 
        self.start_key = "TASK:"
        self.end_key = "MOVE REASONING:"
        self.subtask_history = ['']
        self.highlevel_frequency = 5
        self.update_counter = self.highlevel_frequency -1

    def update_history(self, generated_text):
        """ Update subtask history based on 
            Args:
                - index: if index is specified, only extract that subtask
        """
        if self.update_counter != self.highlevel_frequency - 1:
            return
        start_idx = generated_text.find(self.start_key)
        end_idx = generated_text.find(self.end_key)
        subtask_text =  generated_text[start_idx+len(self.start_key):end_idx]
        if subtask_text != self.subtask_history[-1]:
            self.subtask_history.append(subtask_text)
            print(f"\033[91m subtext {self.start_key}{subtask_text}\033[0m") 

    def generate_prompts(self, task_description):
        """ Generate batch prompts, with history
        """
        prompts = []
        prompt = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: "
        if self.update_counter != 0:
            prompt = prompt + self.start_key + self.subtask_history[-1] + self.end_key # Use updated history
            self.update_counter = self.update_counter - 1
        else:
            prompt = prompt + self.start_key 
            self.update_counter = self.highlevel_frequency - 1
        prompts.append(prompt)
        # print(f"\033[93mprompt: {prompt}\033[0m")
        # print(f"\033[91m Promts Length {len(prompts)}\033[0m")
        return prompts
    


def hf_to_vllm(vla, processor, cfg):
    import vllm
    # Get imbeddings
    if vla.input_embds is None:
        vla.input_embds = vla.language_model.get_input_embeddings()

    # Save language model temporarily
    vllm_model_path = f"tmp/{cfg.pretrained_checkpoint.replace('/', '_')}-vllm"
    vla.language_model.save_pretrained(vllm_model_path)
    processor.save_pretrained(vllm_model_path)

    # Load language model with VLLM
    if hasattr(vla, "language_model"):
        del vla.language_model
    # TODO: check vllm load mode, check settings, memory
    # check if async engine is enabled
    if not hasattr(cfg, 'async_engine') or not cfg.async_engine:
        if not hasattr(cfg, 'quantization'):
            vla.language_model = vllm.LLM(vllm_model_path, 
                                        trust_remote_code=True, 
                                        gpu_memory_utilization=0.7, 
                                        preemption_mode='swap', 
                                        swap_space = 10, 
                                        enable_chunked_prefill = True, 
                                        enable_prefix_caching = True, 
                                        )
        else:
            vla.language_model = vllm.LLM(vllm_model_path, 
                                        trust_remote_code=True, 
                                        gpu_memory_utilization=0.7, 
                                        preemption_mode='swap', 
                                        swap_space = 10, 
                                        enable_chunked_prefill = True, 
                                        enable_prefix_caching = True, 
                                        quantization="bitsandbytes", 
                                        load_format="bitsandbytes"
                                        )
    else:
        vla.language_model  = vllm.AsyncLLMEngine.from_engine_args(
                vllm.AsyncEngineArgs(
                    model= vllm_model_path,
                    gpu_memory_utilization=0.8,
                    preemption_mode='swap',
                    swap_space=12,
                    disable_log_requests=True,
                    enable_prefix_caching=True,
                    # enable_sleep_mode=True,
                )
        )
    return vla
