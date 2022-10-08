import numpy as np
import yaml

import os

import matplotlib.pyplot as plt


class TestDataManager:
    def __init__(self, n_envs, n_policies, n_eps, policies, save_path):

        self.n_envs = n_envs
        self.n_policies = n_policies
        self.n_eps = n_eps

        self.policies = policies
        self.save_path = save_path

        self.n_eps_envs = np.zeros(n_envs)

        self.total_qts_keys = [
            "avg_plc_timing",
            "n_infeasible",
            "n_deadlocked",
            "n_collisions",
            "n_finished",
            "n_timeout",
            "max_step_reward",
        ]
        self.episode_qts_keys = [
            "rewards",
            "steps",
            "env_ids",
            "status",
            "free_cells",
            "seed",
        ]

        self.total_qts = {key: np.zeros((n_policies, 1)) for key in self.total_qts_keys}
        self.episode_qts = {
            key: [[] for i in range(n_policies)] for key in self.total_qts_keys
        }

        self.reset()

    def reset(self):
        self.rewards = [[] for i in range(self.n_envs)]

        self.actions = []

        # self.episode_nn_processing_times = []

    def step(self, reward, infos, dones, ig_plc, plc_id, nn_runtime):
        if ig_plc != "rl_model":
            ig_expert_runtime = (
                sum([infos[i]["ig_expert_runtime"] for i in range(self.n_envs)])
                / self.n_envs
            )
            # Running average
            self._running_average(ig_expert_runtime, self.total_qts["avg_plc_timing"], plc_id)
        else:
            self._running_average(nn_runtime, self.total_qts["avg_plc_timing"], plc_id)

        self.total_qts["n_infeasible"][ig_plc] += sum(
            [infos[i]["is_infeasible"] for i in range(self.n_envs)]
        )
        self.total_qts["max_step_reward"][plc_id] = max(np.max(reward), self.total_qts["max_step_reward"][plc_id])

        for i in range(self.n_envs):

            self.rewards[i].append(np.squeeze(reward[i]))

            if dones[i].any():

                scenario_seed = infos[i]["scenario_seed"]

                # TODO What's that?
                # first condition must be for learning method
                # avoid repeated scenario being logged?
                if plc_id == 0 or scenario_seed not in self.episode_qts["seed"][plc_id]:
                    self.episode_qts["seed"][plc_id].append(scenario_seed)

                    if infos[i]["in_collision"]:
                        self.total_qts["n_collisions"][plc_id] += 1
                    if infos[i]["finished_coverage"]:
                        self.total_qts["n_finished"][plc_id] += 1
                    if infos[i]["deadlocked"]:
                        self.total_qts["n_deadlocked"][plc_id] += 1
                    if infos[i]["ran_out_of_time"]:
                        self.total_qts["n_timeout"][plc_id] += 1

                    self.episode_qts["steps"][plc_id].append(infos[i]["step_num"])
                    self.episode_qts["free_cells"][plc_id].append(infos[i]["n_free_cells"])
                    self.episode_qts["rewards"][plc_id].append(np.sum(self.rewards[i]))
                    self.episode_qts["status"][plc_id].append(
                        0
                        if infos[i]["ran_out_of_time"] or infos[i]["in_collision"]
                        else 1
                    )
                    self.episode_qts["env_ids"][plc_id].append([i, infos[i]["n_episodes"]])
                    self.n_eps_envs[i] = infos[i]["n_episodes"]
                self.rewards[i] = []

                eps_number = len(self.episode_qts["rewards"][plc_id])
                if eps_number % 5 == 0 and eps_number > 0:
                    print(
                        "Episode "
                        + str(eps_number)
                        + " completed with policy "
                        + ig_plc
                    )

    def _running_average(self, new_val, avg_list, key):
        avg_list[key] += (new_val - avg_list[key])/(len(avg_list[key])+1)

    def get_finished_episodes_number(self, plc_id):
        return len(self.episode_qts["rewards"][plc_id])

    def plc_postprocessing(self, plc_id):

        for key in self.episode_qts.keys():
            self.episode_qts[key][plc_id] = self.episode_qts[key][plc_id][: self.n_eps]

    def postprocessing(self):

        avg_rewards = np.mean(np.asarray(self.episode_qts["rewards"]).transpose(), axis=0)
        min_rewards = np.min(np.asarray(self.episode_qts["rewards"]).transpose(), axis=0)
        max_rewards = np.max(np.asarray(self.episode_qts["rewards"]).transpose(), axis=0)
        std_rewards = np.std(np.asarray(self.episode_qts["rewards"]).transpose(), axis=0)

        avg_steps = np.mean(np.asarray(self.episode_qts["steps"]).transpose(), axis=0)

        results_dict = {
            "ig_policies": self.policies,
            "avg_rewards": avg_rewards.tolist(),
            "min_rewards": min_rewards.tolist(),
            "max_rewards": max_rewards.tolist(),
            "std_rewards": std_rewards.tolist(),
            "avg_steps": avg_steps.tolist(),
        }
        for key, value in self.total_qts.items():
            results_dict[key] = value.tolist()

        with open(os.path.join(self.save_path, "results.yml"), "w") as f:
            yaml.dump(results_dict, f)

    def _print_summary(self, results_dict):
        pass

    def save_eps_rewards(self):

        for i in range(self.n_policies):
            ig_plc = self.policies[i]
            output = np.column_stack([
                np.asarray(self.episode_qts[key][i]) for key in self.episode_qts_keys
            ])
            np.savetxt(
                os.path.join(self.save_path, "eps_" + ig_plc + ".csv"),
                output,
                delimiter=",",
            )
        np.savetxt(
            os.path.join(self.save_path, "rewards.csv"), self.episode_qts["rewards"], delimiter=","
        )

    def rewards_plot(self):

        fig = plt.figure()
        plt.rc("font", size=10)
        fig.set_size_inches(8, 8)
        ax = fig.add_subplot(111)
        ax.violinplot(self.episode_qts["rewards"], showmeans=True, showextrema=True)
        # ax.set_xlabel('timesteps')
        ax.set_ylabel("episode rewards")
        ax.set_xticks([i + 1 for i in range(len(self.policies))])
        ax.set_xticklabels(self.policies)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.legend()
        fig.tight_layout()

        # dateObj = datetime.now()
        # timestamp = dateObj.strftime("%Y%m%d_%H%M%S")
        fig.savefig(os.path.join(self.save_path, "rewards.png"))
