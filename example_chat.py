from typing import List, Optional
import fire
import gradio as gr
from llama import Dialog, Llama

def setup_generator(ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int):
    return Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

def generate_response(generator, history, temperature=0.6, top_p=0.9, max_gen_len=None):
    dialog = [{"role": role, "content": message} for role, message in history if role == "user"]
    results = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return results[0]['generation']['content']

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = setup_generator(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    def chat_interface(user_input, history):
        history = history or []
        history.append(("user", user_input))
        response = generate_response(generator, history, temperature, top_p, max_gen_len)
        history.append(("RadAssistant", response))
        return history, history

    with gr.Blocks(css="""
        .gradio-container {font-family: 'Arial', sans-serif;}
        .gradio-chatbox {background-color: #f7f7f7; border: 1px solid #ccc; padding: 10px; border-radius: 5px;}
        .user {color: #0000ff;}
        .RadAssistant {color: #ff0000;}
    """) as demo:
        gr.Markdown("""
        ## Chat with RadAssistant
        ### An AI assistant powered by LLaMA
        """)
        chatbot = gr.Chatbot(label="Chat", show_label=False)
        user_input = gr.Textbox(placeholder="Type your message here...", show_label=False)
        send_button = gr.Button("Send")

        def clear_input():
            return ""

        send_button.click(
            chat_interface,
            inputs=[user_input, chatbot],
            outputs=[chatbot, chatbot],
        )
        
        send_button.click(
            clear_input,
            inputs=[],
            outputs=[user_input],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)  # Bind to all network interfaces

    # Ensure the process group is properly destroyed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)
