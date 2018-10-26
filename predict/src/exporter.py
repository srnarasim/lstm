from sap.mlf.serving.python.exporter.mlf_exporter import MLFExporter
import keras
import pandas as pd
from seq2seq_utils import load_text_processor
from seq2seq_utils import Seq2Seq_Inference

def predict(inputs=None, assetFilesPath=None):
    outputs = detect(inputs['body-text'], assetFilesPath+"/seq2seq_model.h5", assetFilesPath+"/title_pp.dpkl", assetFilesPath+"/body_pp.dpkl")
    return outputs

def export_model():
    m = MLFExporter()
    m.save_python_model("title-gen", predict, requirements_file="requirements.txt", asset_list=["seq2seq_model.h5", "title_pp.dpkl", "body_pp.dpkl"])


def detect( inputs, input_model_h5, input_title_preprocessor_dpkl, input_body_preprocessor_dpkl):
    # Load model, preprocessors.
    seq2seq_Model = keras.models.load_model(input_model_h5)
    num_encoder_tokens, body_pp = load_text_processor(input_body_preprocessor_dpkl)
    num_decoder_tokens, title_pp = load_text_processor(input_title_preprocessor_dpkl)

    # Prepare inference.
    seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                    decoder_preprocessor=title_pp,
                                    seq2seq_model=seq2seq_Model)

    # Output predictions for n random rows in the test set.
    return seq2seq_inf.generate_issue_title( input[0])

export_model()
