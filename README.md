# Generalization-in-Deep-Learning
Please first make sure you have properly installed Cuda on your device. The requirements.txt lists
all the requirements for this implementation. You can install all the requirements by running
```
pip3 install -r requirements.txt
```
in the prompt under your code folder.

To train a ResNet18 on CIFAR-10, use the following command (in a Windows environment):

```
python cifar_trainer.py --arch resnet18 --coeff -1.0 --dataset cifar10 --save-dir folder_name --gpu 0
```

To check the test error and disagreement rate, please run the following command:
```
.\eval.py --path1 folder_path_of_model_1 --path2 folder_path_of_model_2 --gpu 0 --output_name test_name
```

Please note that the first value displayed after running the command is the test accuracy in %.
