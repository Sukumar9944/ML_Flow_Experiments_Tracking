## ML_Flow_Experiments_Tracking

This project showcases the use of MLflow for tracking machine learning experiments and managing models. It uses an ElasticNet model to predict the quality of wine based on various features.

## Getting Started

Follow these steps to set up and run the project in a virtual environment:

### Prerequisites

- Python 3.x installed on your system.
- [Virtualenv](https://pypi.org/project/virtualenv/) for creating a virtual environment.

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   
2. Activate the virtual environment
   venv\Scripts\activate

3. Install the required dependencies : pip install -r requirements.txt

4. Running the Project
- Now that you have set up the virtual environment and installed the dependencies, you can run the project:

- Execute the main script : python train.py

- The script will train the ElasticNet model and print evaluation metrics.

- The MLflow tracking server can be started using the following command to visualize the experiment results: mlflow ui
