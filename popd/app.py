import gradio as gr
import popd.ocr as ocr_lib
import popd.authenticity as auth_lib
import popd.utils as utils

# 动态加载私有库中的 prompt.txt 文件
default_prompt = utils.load_text_from_file("popd/prompt.txt")
sample_json = utils.load_text_from_file("popd/sample_json.json")
sample_authenticity = {'Real': 0, 'Fake': 0}

with gr.Blocks() as demo:
    gr.Markdown("""<center><h1>Proof Of Payment Detection<br><h4>(Test Demo - accuracy varies by model)""")
    with gr.Row(equal_height=True, max_height=500):
        input_img = gr.Image(type='pil', label="Proof Image", show_label=True, height=510)
        with gr.Column():
            output_label = gr.Label(label="Image Authenticity", value=sample_authenticity)
            output_text = gr.JSON(label="Payment Info", show_label=True, value=sample_json)
    submit_btn = gr.Button(value="Detect")

    with gr.Accordion("Advance Options", open=False):
        with gr.Row():
            ocr_model_selector = gr.Dropdown(choices=list(ocr_lib.ocr_models), label="OCR Model", value="Qwen/Qwen2-VL-7B-Instruct")
            authenticity_model_selector = gr.Dropdown(choices=list(auth_lib.authenticity_models), label="Authenticity Model", value="otaku840726/autotrain-realfake-swin-20")
        with gr.Row():
            temperature = gr.Slider(0.0, 2.0, value=0.3, label="Temperature", info="Choose between 0.0 and 2.0")
            min_p = gr.Slider(0.0, 1.0, value=0.3, label="Min P", info="Choose between 0.0 and 2.0")
        text_input = gr.Textbox(label="Prompt", value=default_prompt, lines=15)

    submit_btn.click(ocr_lib.run_ocr_cf, [input_img, text_input, temperature, min_p, ocr_model_selector], [output_text])
    submit_btn.click(auth_lib.run_authenticity, [input_img, authenticity_model_selector], [output_label])

demo.queue(api_open=False)
demo.launch(debug=True)
