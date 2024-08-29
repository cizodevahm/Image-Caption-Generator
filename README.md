# Image Caption Generator
This project is an image caption generator that uses a deep learning model to generate captions for images. The model is trained using the Flickr8k dataset and leverages a pre-trained Xception model for feature extraction and an LSTM network for sequence processing.

## Requirements
- Python 3.x
- NumPy
- Pillow
- Keras
- TensorFlow
- tqdm

## Installation
To install the required packages, run:
```bash
pip install numpy pillow keras tensorflow tqdm
```

## Usage
**Training**
1. Prepare the Dataset:
  - Download the Flickr8k dataset [1](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) [2](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip) and place the images in a directory.
  - Ensure the captions file is in the correct format and location.
2. Run the Notebook:
  - Open the Jupyter Notebook ```training_caption_generator.ipynb```.
  - Follow the steps in the notebook to preprocess the data, extract features, and train the model.

**Testing**
1. Run the Testing Script:
 - Use the provided testing_caption_generator.py script to generate captions for new images.
 - Example usage:
  ```bash
   python testing_caption_generator.py --image <path_to_image>
  ```

## Functions
**Training Functions**

- ```load_doc(filename)```: Loads a text file into memory.
- ```all_img_captions(filename)```: Retrieves all images with their captions.
- ```cleaning_text(captions)```: Cleans the text by lowercasing, removing punctuation, and filtering out 
words with numbers.
- ```text_vocabulary(descriptions)```: Builds a vocabulary of all unique words.
- ```save_descriptions(descriptions, filename)```: Saves all descriptions to a file.
- ```extract_features(directory)```: Extracts features from images using the Xception model.
- ```data_generator(descriptions, features, tokenizer, max_length)```: Generates input-output sequence 
pairs for training.
- ```define_model(vocab_size, max_length)```: Defines the image captioning model architecture.

**Testing Functions**

- ```extract_features(filename, model)```: Extracts features from an image using the Xception model.
- ```word_for_id(integer, tokenizer)```: Maps an integer to a word using the tokenizer.
- ```generate_desc(model, tokenizer, photo, max_length)```: Generates a caption for an image using the 
trained model.

## License
This project is licensed under the MIT License.
