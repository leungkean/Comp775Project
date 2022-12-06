# Comp775Project
Image Segmentation using Both Supervised and Unsupervised Active Feature Acquisition

## Initialization
Follow the `afa_guide.txt` for intialization.

## Training the Surrogate Model
To train the surrogate model for AIR (unsupervised AFA) just run `train_air_deformer_surrogate.py`.

## Training the Agent using PPO
To train the agent using PPO run the scripts `train_ppo_air_surrogate.py` and `train_gym_ppo.py`.
`train_ppo_air_surrogate.py` is for the AIR task while `train_gym_ppo.py` is for the supervised AFA task.

## My code contributions
I mainly contributed by modifying the code so as to be able train an agent using a pretrained
U-Net model for supervised AFA.
