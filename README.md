# MLOps

## Create pipelines for machine learning models to ensure they are retrainable, manageable and controllable:

### Machine learning models usually will require multiple execution for performance comparison and monitoring purposes.
Step1: Break a jupyter notebook into smaller modules (Easier for unit testing and debugging)

Step2: Use logging to keep track of the steps and changes in each execution.

Step3: Initialize dvc environment and create dvc.yaml file to execute those python files in order.

Step4: Store the history metrics in the json file for comparison for the next nth execution.

Step5: Better model will store in MLFlow for versioning of performance and parameters.

Step6: Deploy using BentoML by API service or docker.


The whole model can be reproduced by a command with `dvc.yaml` and `src` folder:
```
$ dvc repro dvc.yaml
```

Execute the MLFlow web application:
```
$ mlflow ui
```

![Untitled Diagram drawio](https://user-images.githubusercontent.com/102400483/200590588-fe467efc-4d19-4db3-b5e6-b06968e77246.png)
