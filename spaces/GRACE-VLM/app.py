import threading

import gradio as gr
import spaces
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "ForeverBlue/Qwen3-VL-2B-GRACE-BF16"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda").eval()
generation_lock = threading.Lock()


@spaces.GPU(duration=120)
def answer(image, prompt, max_new_tokens):
    if image is None:
        raise gr.Error("Please upload an image first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a question about the image.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with generation_lock, torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    return processor.decode(
        outputs[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


with gr.Blocks(title="GRACE-VLM Demo") as demo:
    gr.Markdown(
        """
        # 🦢 GRACE-VLM
        **An ICML 2026 Qwen3-VL-2B student distilled from an 8B teacher.**

        Upload an image and ask a question. This online demo runs the GRACE
        BF16 checkpoint on ZeroGPU. The primary deployment checkpoint uses
        [real AWQ-packed INT4](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ)
        and retains 98% of the GRACE BF16 benchmark average.
        """
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        with gr.Column():
            prompt_input = gr.Textbox(
                value="Describe this image in detail.",
                label="Question",
                lines=3,
            )
            token_input = gr.Slider(
                minimum=32,
                maximum=256,
                value=128,
                step=32,
                label="Maximum new tokens",
            )
            run_button = gr.Button("Run GRACE", variant="primary")
    output = gr.Textbox(label="GRACE response", lines=10)
    gr.Markdown(
        "[Paper](https://arxiv.org/abs/2601.22709) · "
        "[Code](https://github.com/ForeverBlue816/GRACE) · "
        "[Models](https://huggingface.co/collections/ForeverBlue/grace)"
    )
    run_button.click(
        fn=answer,
        inputs=[image_input, prompt_input, token_input],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
