"""Interface file"""

import platform

import gradio as gr


def say_filename(file):
    if file is not None:
        is_windows = platform.system() == "Windows"
        separator = "\\" if is_windows else "/"

        filename = file.name.split(separator)[-1]
        return {"filename": filename}
    else:
        return {"message": "The input file is empty."}


demo = gr.Interface(
    fn=say_filename,
    inputs="file",
    outputs="json",
)


if __name__ == "__main__":
    demo.launch(debug=True)
