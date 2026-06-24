import os
from stable_baselines3 import PPO
from cartpole_env import CartPoleCustomEnv

def main():
    # 1. Setup paths for saving the model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ppo_cartpole")

    # 2. Initialize the environment
    print("Initializing environment...")
    env = CartPoleCustomEnv()

    # 3. Create the PPO agent
    # MlpPolicy: A standard neural network (Multi-Layer Perceptron)
    # verbose=1: Prints training metrics (like average reward) to the console
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,        # longer rollouts -> the agent sees full swing-up attempts
        batch_size=64,
        gamma=0.99,          # value a few seconds of upright balance ahead
        gae_lambda=0.95,
        ent_coef=0.01,       # encourage exploration to discover the swing-up
    )

    # 4. Train the agent
    total_timesteps = 1_000_000
    print(f"Starting training ({total_timesteps:,} timesteps)...")
    model.learn(total_timesteps=total_timesteps)

    # 5. Save the weights
    print(f"Saving the model to {model_path}...")
    model.save(model_path)

    print("Training finished successfully!")
    env.close()

if __name__ == "__main__":
    main()