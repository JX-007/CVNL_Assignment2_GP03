# Project Setup Guides

This README contains **two separate guides**:

1. **Guide A:** Google Colab Local Runtime + CUDA (Model Training)
2. **Guide B:** Streamlit Application Setup (Model Demo)

Use **Guide A** if you are training or re-running the model.
Use **Guide B** if you are only running the Streamlit app.

---

# Guide A: Google Colab Local Runtime + CUDA

This guide explains how to run Google Colab notebooks on your **local machine** with **GPU (CUDA) support**.

---

## A1. Requirements

* NVIDIA GPU
* Python 3.9+
* NVIDIA drivers installed
* CUDA Toolkit (11.8 or 12.1)

Check GPU:

```bash
nvidia-smi
```

---

## A2. Install Colab Local Runtime Extension

1. Open Google Chrome
2. Go to Chrome Web Store
3. Search **Colab Local Runtime**
4. Add extension

This allows Colab to run notebooks locally instead of on Google servers.

---

## A3. Start Jupyter Notebook Locally

```bash
pip install jupyter
jupyter notebook
```

You will see:

```
http://localhost:8888/?token=xxxx
```

Keep this terminal open.

---

## A4. Connect Colab to Local Runtime

1. Open your notebook in Google Colab
2. Click **Connect**
3. Select **Connect to local runtime**
4. Paste the localhost URL
5. Connect

Colab is now using your local machine.

---

## A5. Install PyTorch with CUDA

Choose the version that matches your CUDA installation.

**CUDA 12.1 (recommended):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (older GPUs):**

```bash
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## A6. Install Required Libraries

```bash
pip install datasets scikit-learn seaborn matplotlib pandas numpy
```

---

## A7. Verify CUDA

Run inside the notebook:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is available and your GPU name appears, setup is complete.

---

# Guide B: Streamlit Application Setup (Model Demo)

This guide explains how to run the **Streamlit web application** that demonstrates the trained model.

This guide **does not require CUDA**.

---

## B1. What is Streamlit?

Streamlit is a Python library that lets you create simple web apps using only Python.
It is commonly used to demo machine learning models.

---

## B2. Install Required Packages

Install Streamlit and related libraries:

```bash
pip install streamlit torch numpy Pillow
```

Or, if provided:

```bash
pip install -r requirements.txt
```

---

## B3. Project Folder Structure

```
my_changi_project/
│
├── changi_ops_assistant.py
├── requirements.txt
├── README.md
├── check_setup.py
├── run_app.bat
├── run_app.sh
│
├── checkpoints/
│   └── intent_best.pth
│
└── data/
    └── intent_vocab.json
```

---

## B4. Getting the Model Files

The model files are generated from the training notebook.

Steps:

1. Open `rnn_intent.ipynb`
2. Run all cells
3. Training completes
4. Files created:

   * `checkpoints/intent_best.pth`
   * `data/intent_vocab.json`
5. Copy these folders into the Streamlit project directory

---

## B5. Run the Streamlit App

From the project folder:

```bash
streamlit run changi_ops_assistant.py
```

The app opens at:

```
http://localhost:8501
```

---

## B6. Alternative Ways to Run

**Windows:**

* Double-click `run_app.bat`

**Mac/Linux:**

```bash
chmod +x run_app.sh
./run_app.sh
```

**VS Code Terminal:**

```bash
streamlit run changi_ops_assistant.py
```

---

## B7. Check Setup Before Running

```bash
python check_setup.py
```

This verifies:

* Required packages
* Model files
* Folder structure

---

## B8. Using the App

**Sidebar:**

* Model information
* Feature selection (RNN / CNN)
* Credits

**Main Area:**

* Text input
* Predicted intent
* Confidence score
* Routing suggestion

Example inputs:

* “What gate does my flight depart from?”
* “Where can I get a taxi?”
* “How much is a ticket to Tokyo?”

---

## B9. Stopping the App

Press:

```
Ctrl + C
```

in the terminal.

---

## Common Issues

**Streamlit not recognised**

```bash
pip install streamlit
```

**Torch not installed**

```bash
pip install torch
```

**Port already in use**

```bash
streamlit run changi_ops_assistant.py --server.port 8502
```
