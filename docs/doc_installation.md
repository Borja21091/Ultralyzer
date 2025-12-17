# Installation

## System Requirements

Ultralyzer is compatible with Linux, Windows, and MacOS systems that support Python 3.12 (tested) or higher. The software leverages hardware acceleration for image processing and GUI rendering, so a system with a dedicated GPU is recommended for optimal performance, although not strictly necessary.

We recommend using a virtual environment (e.g., Conda) to manage dependencies and avoid conflicts with other Python packages on your system. You can start by installing [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

## Installation Steps

<details>

<summary>Windows</summary>

1. **Clone the Repository**:

   Open Anaconda Prompt and run:

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer.git
   cd Ultralyzer
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. **Install Dependencies**:

   Run the following command from the command line:

   ```bash
   install.bat
   ```

4. **Run the Application**:

   ```bash
   python src/ultralyzer/main.py
   ```

</details>

<br>

<details>

<summary>Linux & MacOS</summary>

1. **Clone the Repository**:

   Open a terminal and run:

   ```bash
   git clone https://github.com/Borja21091/Ultralyzer.git
   cd Ultralyzer
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   conda create -n ultralyzer_env python=3.12
   conda activate ultralyzer_env
   ```

3. **Install Dependencies**:

   Run the following command in the terminal:

    ```bash
    chmod +x install.sh
    bash install.sh
    ```

4. **Run the Application**:

   ```bash
   python src/ultralyzer/main.py
   ```

</details>
