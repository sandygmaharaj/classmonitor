from fastcore.all import *
from fastai.learner import *
from fastai.vision.all import *
import gradio as gr

categories = ('confused','attentive', 'bored', 'interested', 'frustrated', 'thoughtful')
learn_inf = load_learner('export.pkl')

def classify_image(img):
    pred,idx,prods = learn_inf.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch