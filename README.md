# Generalization-in-Deep-Learning
Please first make sure you have properly installed Cuda on your device. The requirements.txt lists
all the requirements for this implementation. You can install all the requirements by running ’pip3
install -r requirements.txt’ in the prompt under your code folder.
To train a ResNet18 on CIFAR-10, use the following command (in Windows environment):


python c i f a r t r a i n e r . py −− a r c h r e s n e t 1 8 −− c o e f f −1.0
−− d a t a s e t c i f a r 1 0 −−save − d i r f o l d e r n ame −−gpu 0
To check test error and disagreement rate, please run the following command:
. \ e v a l . py −− p a t h 1 f o l d e r p a t h o f mo d e l 1
−− p a t h 2 f o l d e r p a t h o f mo d e l 2 −−gpu 0 −− o u t p u t n ame t e s t n ame
It is worth noting that the first value displayed after running the command is the test accuracy in %.
