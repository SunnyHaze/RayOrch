from dataflow.serving import LocalModelLLMServing_sglang, LocalVLMServing_vllm

class SGLangServingOP():
    def __init__(self, 
                 hf_model_name_or_path,
                 sgl_tp_size=4,
                 sgl_dp_size=1,
                 sgl_mem_fraction_static: float = 0.9,
                 ):
        self.hf_model_name_or_path = hf_model_name_or_path
        self.sgl_tp_size = sgl_tp_size
        self.sgl_dp_size = sgl_dp_size
        self.sgl_mem_fraction_static = sgl_mem_fraction_static
        self.llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path=self.hf_model_name_or_path,
            tp_size=self.sgl_tp_size,
            dp_size=self.sgl_dp_size,
            mem_fraction_static=self.sgl_mem_fraction_static
        )
    def run(self, input_prompts):
        res = self.llm_serving.generate_from_input(input_prompts)
        # add a tag before each output to indicate which operator it comes from
        tagged_res = [f"SGLangOP: {output}" for output in res]
        return tagged_res
    

class vLLMServingOP():
    def __init__(
            self,
            hf_model_name_or_path,
            vllm_tensor_parallel_size=4,
            vllm_gpu_memory_utilization=0.9,
    ):
        self.llm_serving = LocalVLMServing_vllm(
            hf_model_name_or_path=hf_model_name_or_path,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
    def run(self, input_prompts):
        res = self.llm_serving.generate_from_input(input_prompts)
        # add a tag before each output to indicate which operator it comes from
        tagged_res = [f"vLLMOP: {output}" for output in res]
        return tagged_res