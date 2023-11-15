### Hand-Written-Digit-Classifier

#### Usage:
1. **Training a New Model:**
   - Set `train_new_model` to `True` in the script.
   - Run the script to train the model on the MNIST dataset.
   - The trained model will be saved as `test-model.model`.

2. **Using Pre-trained Model:**
   - Set `train_new_model` to `False` in the script.
   - The script will load the pre-trained model from `test-model.model`.

3. **Classifying Custom Images:**
   - Place custom digit images in the `digits` folder as `digit1.png`, `digit2.png`, etc.
   - Run the script to classify the custom images.

#### Requirements:
- TensorFlow
- OpenCV
- Matplotlib

#### Instructions:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Hand-Written-Digit-Classifier.git
   cd Hand-Written-Digit-Classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python main.py
   ```

Feel free to customize this template based on the specific details of your project. Provide more details on the installation process, usage, and any additional functionalities. Additionally, you can include a license section, contributor guidelines, and any other information you find relevant.
