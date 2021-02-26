# Speaker-Representation-Learning

This repository contains the code for training unsupervised speaker representation learning models with LibriSpeech Dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

Download Noise Dataset with [wavencoder](https://github.com/shangeth/wavencoder)
```python
import wavencoder
wavencoder.utils.download_noise_dataset('path-to-noise-dir')
```


## Usage

### Download the Dataset
```bash
# Download and Extract Librispeech train-clean-360, dev-clean, test-clean
python download_dataset.py
```

### Prepare the Dataset for training
```bash
# Prepare the Librispeech Dataset for Speaker Representation learning training
python prepare_repr_dataset.py
```

### Training
```bash
# dev run the training loop
python train.py --dev=True --data_root='lirispeech-dataset-path'

# Train the model
python train.py --data_root='lirispeech-dataset-path' --epochs=1000
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)