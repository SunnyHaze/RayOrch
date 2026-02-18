import os
from time import time
import ray
from typing import List, Tuple, Any

from rayorch import RayModule  # 用你现有 RayModule
# 如果你有 Dispatch 也可用，但这里用单 actor 更像“最简单demo”

from ops import SGLangServingOP, vLLMServingOP


def _wrap_prompts(texts: List[str]) -> List[str]:
    # 保证始终是 list[str]
    return texts if isinstance(texts, list) else [str(texts)]


class PingPongServingPipeline:
    """
    两个 Serving 来回“打乒乓” ≥4 次：
      S -> V -> S -> V -> S -> V -> S -> V
    每一步都明确改变任务/语言/格式，让 demo 看起来“变化明显、合理、优雅”。
    """

    def __init__(
        self,
        *,
        model_path: str,
        sglang_env: str,
        vllm_env: str,
        # 你可按显存改这些
        sgl_tp: int = 1,
        vllm_tp: int = 1,
        sgl_mem: float = 0.90,
        vllm_mem: float = 0.90,
    ):
        # SGLang actor
        self.sgl = RayModule(
            SGLangServingOP,
            env=sglang_env,
            replicas=1,
            num_gpus_per_replica=1.0,
        ).pre_init(
            hf_model_name_or_path=model_path,
            sgl_tp_size=sgl_tp,
            sgl_dp_size=1,
            sgl_mem_fraction_static=sgl_mem,
        )

        # vLLM actor
        self.vllm = RayModule(
            vLLMServingOP,
            env=vllm_env,
            replicas=1,
            num_gpus_per_replica=1.0,
        ).pre_init(
            hf_model_name_or_path=model_path,
            vllm_tensor_parallel_size=vllm_tp,
            vllm_gpu_memory_utilization=vllm_mem,
        )

        import time
        t0 = time.time()
        print("[DRIVER] pre_init done at", t0, flush=True)


    def _step(self, op_name: str, op, prompts: List[str]) -> List[str]:
        out = op(_wrap_prompts(prompts))
        # 你的 OP 已经加了 tag（SGLangOP: / vLLMOP:），这里再加一层 step tag 便于 demo 观感
        return [f"[{op_name}] {x}" for x in out]

    def __call__(self, user_prompt: str) -> Tuple[List[str], List[str]]:
        """
        返回：
          - final_outputs: 最后一步 vLLM 的输出 list[str]
          - transcript: 全链路每一步的输出记录（用于 demo 展示）
        """
        transcript: List[str] = []

        # Round 0: seed prompt
        p0 = [f"""You are a helpful assistant.

TASK: Turn the following into a short demo story about "two model servers ping-ponging prompts".
Constraints:
- Keep it concrete and technical.
- Mention Ray actors and separate conda envs.
- Keep it under 120 words.

INPUT:
{user_prompt}
"""]
        transcript.append("=== Seed Prompt ===\n" + p0[0])

        # Round 1: SGLang -> EN summary + bullets
        p1 = [f"""Rewrite the INPUT into:
1) A 2-sentence English summary
2) Exactly 3 bullet points
Be concise. No extra text.

INPUT:
{p0[0]}
"""]
        o1 = self._step("SGLang R1 (EN summary+bullets)", self.sgl, p1)
        transcript.append("=== R1 Output ===\n" + "\n".join(o1))

        # Round 2: vLLM -> 中文 + 标题
        p2 = [f"""把下面内容翻译成中文，并添加一个简短标题。
要求：标题一行，正文保持原有结构（2句总结 + 3个要点）。

TEXT:
{chr(10).join(o1)}
"""]
        o2 = self._step("vLLM R2 (ZH title)", self.vllm, p2)
        transcript.append("=== R2 Output ===\n" + "\n".join(o2))

        # Round 3: SGLang -> 日文（丁寧語）+ 注意事项
        p3 = [f"""次の内容を日本語（丁寧語）に変換してください。
さらに末尾に「注意事項」を1行だけ追加してください（例：実行環境のnumpyを揃える等）。
形式は読みやすく。

TEXT:
{chr(10).join(o2)}
"""]
        o3 = self._step("SGLang R3 (JA polite + note)", self.sgl, p3)
        transcript.append("=== R3 Output ===\n" + "\n".join(o3))

        # Round 4: vLLM -> 西班牙语 tweet 风格
        p4 = [f"""Convierte el texto al español con estilo de tweet:
- Máx 2 tweets (separados por '---')
- Mantén términos técnicos (Ray, actor, conda env)
- Incluye 2 hashtags relevantes

TEXT:
{chr(10).join(o3)}
"""]
        o4 = self._step("vLLM R4 (ES tweets)", self.vllm, p4)
        transcript.append("=== R4 Output ===\n" + "\n".join(o4))

        # Round 5: SGLang -> 英文 key takeaways + action items
        p5 = [f"""Convert the content into:
- Key takeaways (3 bullets)
- Action items (2 bullets)
English only, crisp.

TEXT:
{chr(10).join(o4)}
"""]
        o5 = self._step("SGLang R5 (takeaways/actions EN)", self.sgl, p5)
        transcript.append("=== R5 Output ===\n" + "\n".join(o5))

        # Round 6: vLLM -> 中文口播脚本（带停顿）
        p6 = [f"""把下面英文内容改写成中文「口播脚本」：
- 适合30秒讲完
- 用【停顿】标记停顿
- 语气自然，面向工程师demo

TEXT:
{chr(10).join(o5)}
"""]
        o6 = self._step("vLLM R6 (ZH spoken script)", self.vllm, p6)
        transcript.append("=== R6 Output ===\n" + "\n".join(o6))

        # Round 7: SGLang -> YAML 结构化
        p7 = [f"""Turn this into YAML with keys:
title, context, steps (list), risks (list), envs (dict with sglang, vllm)
Keep values short.

TEXT:
{chr(10).join(o6)}
"""]
        o7 = self._step("SGLang R7 (YAML)", self.sgl, p7)
        transcript.append("=== R7 Output ===\n" + "\n".join(o7))

        # Round 8: vLLM -> 最终总结 + 文本对比表（漂亮 demo）
        p8 = [f"""You are preparing a demo README snippet.
1) Write a short, polished final summary (<=120 words)
2) Then a plain-text comparison table with 3 rows:
   - Runtime env isolation
   - Prompt ping-pong depth
   - Output diversity
Columns: Feature | SGLang | vLLM

Use the YAML as the source of truth.

YAML:
{chr(10).join(o7)}
"""]
        o8 = self._step("vLLM R8 (final summary + table)", self.vllm, p8)
        transcript.append("=== R8 Output (FINAL) ===\n" + "\n".join(o8))

        return o8, transcript


if __name__ == "__main__":
    import os
    import time
    from datetime import datetime

    ray.init(ignore_reinit_error=True)

    SGLANG_ENV = "df_sglang_mxc"
    VLLM_ENV = "df-vllm-mxc"

    MODEL_PATH = "/vepfs-mlp2/c20250602/500050/models/Qwen3-0.6B/qwen/Qwen3-0.6B"

    pipe = PingPongServingPipeline(
        model_path=MODEL_PATH,
        sglang_env=SGLANG_ENV,
        vllm_env=VLLM_ENV,
        sgl_tp=1,
        vllm_tp=1,
        sgl_mem=0.90,
        vllm_mem=0.90,
    )

    final_out, transcript = pipe(
        "We want a minimal RayOrch demo that shows two different serving backends calling each other."
    )

    print("\n================ FINAL OUTPUT ================\n")
    print("\n".join(final_out))

    print("\n================ TRANSCRIPT (ALL STEPS) ================\n")
    for t in transcript:
        print(t)
        print()

    # ========= 保存日志 =========
    log_dir = "./cache"
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join(log_dir, f"pingpong_demo_{ts}.txt"))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("========== FINAL OUTPUT ==========\n\n")
        f.write("\n".join(final_out))
        f.write("\n\n========== TRANSCRIPT ==========\n\n")
        for t in transcript:
            f.write(t)
            f.write("\n\n")

    print("\nLogs saved to:", log_path)

    time.sleep(1)
    ray.shutdown()