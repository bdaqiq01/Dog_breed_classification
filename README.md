# Dog Breed Classification

A TensorFlow-based image classification model for identifying dog breeds using the Stanford Dogs dataset. The notebook covers data loading, annotation parsing, preprocessing, model training, and evaluation.

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell)**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (cmd)**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux**
  ```bash
  source venv/bin/activate
  ```

### 2. Install dependencies

With the environment activated, install everything at once from the requirements file:

```bash
pip install -r requirment.txt
```

### 3. Add `data/` to `.gitignore`

The dataset is large and should not be committed. Make sure your `.gitignore` includes:

```
data/
```

This is already present in the repo's `.gitignore`, so no action is needed if you cloned the project. If you initialized a fresh repo, add the line above before committing.

### 4. Run the notebook

```bash
jupyter notebook model_develpment.ipynb
```
