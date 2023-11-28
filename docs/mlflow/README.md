# mlflow
* using mlflow api from tracking to serving

## run mlflow server
* install mlflow
    ```bash
    pip install mlflow
    ```

* run mlflow server in mlflow data directory.
    ```bash
    cd mlflow_data/
    mlflow server --host 0.0.0.0 --port 5000
    ```

## model tracking
1. set tracking uri and experiment name.
    ```python
    import mlflow

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('BERT_pretrain')
    ```

2. run mlflow with context manager. 
    ```python
    with mlflow.start_run(run_name='vocab10k'):
        train_loop()
    ```
    * or
    ```python
    mlflow.start_run()
    train_loop()
    mlflow.end_run()
    ```

3. log parameters.
    ```python
    params = {
        'batch_size': 128
        'layers': 6,
        'd_model': 1024,
        'heads': 8,
        'ff': 2048
    }
    mlflow.log_params(params)
    ```

4. log metrics
    ```python
    mlflow.log_metric('Loss/valid', metrics['loss'], step)
    mlflow.log_metric('Acc/valid', metrics['acc'], step)
    ```
    * using dictionary:
    ```python
    mlflow.log_metrics(
        {
            'Loss/valid': metrics['loss'],
            'Acc/valid': metrics['acc']
        },
        step
    )
    ```

4. log model
    ```python
    if best_val_acc < metrics['acc']:
        best_val_acc = metrics['acc']
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path='best_val_acc_model')
        get_logger().info('best acc %f model is saved.', best_val_acc)
    ```

## log predict model

```python
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
            context.artifacts['best_val_acc_model'], map_location="cpu")  # key/value in context.artifacts from artifacts argument in mlflow.pyfunc.log_model()
        self.device = torch.device('cuda:1')
        self.model.to(self.device)
        self.model.eval()

        # vocab_file = 'bpe/BPE_char_dict_PILE_15589.json'
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


with mlflow.start_run(run_id='b86642ad8b1b425fa35bbd43ba986104') as run:  # get run_id from web page.
    mlflow.pyfunc.log_model(
        artifact_path='finetuning_predict',  # save artifact path
        artifacts={  # key/value used in context of PredictModel
            'best_val_acc_model': mlflow.get_artifact_uri('best_val_acc_model')
        },
        python_model=PredictModel(),
        code_path=[  # all files in directory are uploaded to load on serving.
            os.path.join(os.path.dirname(__file__), '../components'),
            os.path.join(os.path.dirname(__file__), '../models'),
            os.path.join(os.path.dirname(__file__), '../checkpoint.py'),
            os.path.join(os.path.dirname(__file__), '../configuration.py'),
            os.path.join(os.path.dirname(__file__), '../configuration_fine_tuning.py'),
            os.path.join(os.path.dirname(__file__), '../dataset_loader'),
            os.path.join(os.path.dirname(__file__), 'bpe'),
            os.path.join(os.path.dirname(__file__), 'BPE_char_dict_PILE_15589.py')]

    )
```

## serving model
* run serve
    ```bash
    MLFLOW_TRACKING_URI="http://127.0.0.1:5000" mlflow models serve -m "models:/bert_tweet_disaster/4" --no-conda --port 5001
    ```
    * model name format is 'models:/{registered_model_name}/{version}'
      * you can also use the full path of artifact.

* send text to be inferred.
    ```bash
    curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{"inputs": {"text": "the houses are burning!!"}}'
    ```
    * received: {"predictions": [[0.611606240272522]]}
