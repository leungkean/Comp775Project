# Comp775Project
Image Segmentation using Both Supervised and Unsupervised Active Feature Acquisition

## Initialization
Follow the `afa_guide.txt` for intialization.

## Training the Surrogate Model
To train the surrogate model for AIR (unsupervised AFA) just run `train_air_deformer_surrogate.py`.

## Training the Agent using PPO
To train the agent using PPO run the scripts `train_ppo_air_surrogate.py` and `train_gym_ppo.py`.
`train_ppo_air_surrogate.py` is for the AIR task while `train_gym_ppo.py` is for the supervised AFA task.

## Evaluating the RL Agent
To evaluate the agent run the scripts `eval_air_surrogate.py` and `eval_gym_ppo.py`.

## My code contributions
I mainly contributed by modifying the code so as to be able train an agent using a pretrained
U-Net model for supervised AFA found in `train_gym_ppo.py` and `eval_gym_ppo.py`. I also implemented the U-Net model in `afa/network/segment`
in `unet.py`.

## Notes
- `afa` is the main driver for active feature acquisition
- Here `afa` and `bax` are required libraries for `afa`.

