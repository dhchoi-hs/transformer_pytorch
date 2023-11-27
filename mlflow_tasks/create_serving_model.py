import os
import json
import mlflow
from mlflow.pyfunc import PythonModel
import torch
from bpe.preprocess_tweetD import preprocess_text
from bpe.bpe_codec_char import encode_bpe_char
from BPE_char_dict_PILE_15589 import vocab


class PredictModel(PythonModel):
    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(
            context.artifacts['best_val_acc_model'], map_location="cpu")
        self.device = torch.device('cuda:1')
        self.model.to(self.device)
        self.model.eval()

        # vocab_file = 'BPE_char_dict_PILE_15589.json'
        # with open(vocab_file, 'rt', encoding='utf8') as f:
        #     self.vocab = json.load(f)

    def predict(self, context, model_input, params=None):
        input_text = str(model_input['text'])
        tokens = encode_bpe_char(vocab, preprocess_text, input_text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            o = self.model(tokens.unsqueeze(0))

        return o.cpu().numpy()


mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run(run_id='b86642ad8b1b425fa35bbd43ba986104') as run:
    mlflow.pyfunc.log_model(
        artifact_path='finetuning_predict',
        artifacts={
            'best_val_acc_model': mlflow.get_artifact_uri('best_val_acc_model')
        },
        python_model=PredictModel(),
        code_path=[
            os.path.join(os.path.dirname(__file__), '../models'),
            os.path.join(os.path.dirname(__file__), '../model'),
            os.path.join(os.path.dirname(__file__), '../checkpoint.py'),
            os.path.join(os.path.dirname(__file__), '../configuration.py'),
            os.path.join(os.path.dirname(__file__), '../configuration_fine_tuning.py'),
            os.path.join(os.path.dirname(__file__), '../dataset_loader'),
            os.path.join(os.path.dirname(__file__), 'bpe')]
    )
