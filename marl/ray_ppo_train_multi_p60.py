from env_proxy_multi_60 import EnvProxy, obs_space, act_space
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print



algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=50)
    .resources(num_cpus_per_worker=1, num_gpus=1)
    .framework('torch')
    .environment(env=EnvProxy, env_config={'n_neural': 60}, disable_env_checking=True)
    .multi_agent(policies={
        'ps_policy': (None, obs_space, act_space, {'gamma': 0.95})
    }, policy_mapping_fn=lambda _: "ps_policy")
    .build()
)
# algo.restore("/home/work1/ray_results/***/")
for i in range(100000):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")