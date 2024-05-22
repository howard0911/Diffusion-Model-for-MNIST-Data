# Diffusion Model for MNIST Data

![alt text](/assets/output_63_0.png)

In this demo, I am going to implement the paper Denoising Diffusion Probabilistic Models, which presented high-quality image synthesis results using diffusion probabilistic models in 2020, using MNIST data, which is well-formatted handwritten data.

## Theory

In this demo, I try two different losses to train the model. The first is to model the conditional mean. The second is to model the error. 

1. $L(X_j, \theta) = \frac{|X_{t_j-1} - \mu(X_{t_j}, t_j ; \theta)|^2}{2(1 - \alpha_{t_j})}$
2. $L(X_j, \theta) = |\epsilon_j - e(X_{t_j}, t_j; \theta)|^2$

The Unet outputs are $\mu(X_{t_j}, t_j ; \theta)$ and $e(X_{t_j}, t_j; \theta)$ respectively. 

## Implementation

In Part 1, the model is used to directly model the conditional mean
$\mu(x_t,t;\theta)$ given the input and time step. This means the model learns to predict the mean of the next state given the current state and the time step.

In Part 2, the model is used to directly model the error term $e(x_t,t;\theta)$ given the input and time step. This means the model learns to predict the error term that is added to the current state to generate the next state according to the diffusion process.

Observable differences between the two versions include:

- Model Output: In Part 1, the model outputs the conditional mean, while in Part 2, the model outputs the error term.
- Sampling Process: In Part 1, the sampling process involves generating samples based on the predicted mean and adding noise according to the diffusion process. In Part 2, the sampling process involves generating samples based on the predicted error term and adding noise according to the diffusion process.
- Loss Function: The loss function used for training differs between the two versions. In Part 1, the loss function is based on the squared difference between the predicted mean and the ground truth next state. In Part 2, the loss function is based on the squared difference between the predicted error term and the actual error term generated during the diffusion process.
- Sample Images: From the trained models in two problems, we can observe somehow handwritten digits, but we got a more clear image in the error model.
- Models Size: 4.3 MB for one ScoreNet model.
- Training Time: Around 20 secs for one epoch.

You can obtain more details in the Jupyter Notebook by running the code.

## Reference

- Theory Background: The Denoising Diffusion Probabilistic Models: [Link](https://arxiv.org/pdf/2006.11239.pdf)
- Calculation: Understanding Diffusion Models: A Unified Perspective: [Link](https://arxiv.org/pdf/2208.11970)
