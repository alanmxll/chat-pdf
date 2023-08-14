import gradio as gr


def say_filename(file):
    if file is not None:
        filename = file.name.split("/")[-1]
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
