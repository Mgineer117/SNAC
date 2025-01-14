import numpy as np
import torch
import json
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# from utils import *
from utils.call_network import call_sfNetwork
from utils.get_all_states import get_grid_tensor
from utils.plotter import Plotter
from utils.sampler import OnlineSampler
from utils.eigenvector import get_eigenvectors
from utils.call_env import call_env
from models.evaulators.base_evaluator import DotDict

COLORS = {
    "SNAC": "red",
    "SNAC+": "red",
    "SNAC++": "red",
    "SNAC+++": "red",
    "EigenOption": "blue",
    "EigenOption+": "blue",
    "EigenOption++": "blue",
    "EigenOption+++": "blue",
    "CoveringOption": "green",
    "PPO": "magenta",
}

LINESTYLES = {
    "EigenOption": "-",
    "EigenOption+": (0, (5, 10)),
    "EigenOption++": ":",
    "EigenOption+++": "--",
}

MARKERS = {
    "EigenOption": {
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 15,
    },
    "EigenOption+": {
        "marker": "s",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 15,
    },
    "EigenOption++": {
        "marker": "X",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 15,
    },
    "EigenOption+++": {
        "marker": "^",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 15,
    },
}

LABEL = {
    "EigenOption": f"Top n",
    "EigenOption+": f"CVS",
    "EigenOption++": f"CRS",
    "EigenOption+++": f"TRS",
}


def init_args(algo_name: str, num_vector: str):
    model_dir = "log/eval_log/model_for_eval/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.import_sf_model = True
    args.s_dim = tuple(args.s_dim)

    args.algo_name = algo_name
    args.env_name = "Maze"
    args.num_vector = num_vector
    args.device = torch.device("cpu")

    print(f"Algo name: {args.algo_name}")
    print(f"Env name: {args.env_name}")
    print(f"Num vector: {args.num_vector}")

    return args


def get_vectors(args):
    plotter = Plotter(
        grid_size=args.grid_size,
        img_tile_size=args.img_tile_size,
        device=args.device,
    )

    sampler = OnlineSampler(
        training_envs=env,
        state_dim=args.s_dim,
        feature_dim=args.sf_dim,
        action_dim=args.a_dim,
        hc_action_dim=args.num_vector + 1,
        agent_num=args.agent_num,
        min_cover_option_length=args.min_cover_option_length,
        episode_len=args.episode_len,
        batch_size=4096,
        min_batch_for_worker=args.min_batch_for_worker,
        cpu_preserve_rate=args.cpu_preserve_rate,
        num_cores=args.num_cores,
        gamma=args.gamma,
        verbose=False,
    )

    option_vals, options, _ = get_eigenvectors(
        env,
        sf_network,
        sampler,
        plotter,
        args,
        draw_map=False,
    )

    return option_vals, options


def get_grid(args):
    grid, pos, loc = get_grid_tensor(env, grid_type=args.grid_type)

    return grid, pos


def get_feature_matrix(feaNet, grid, pos, args):
    features = torch.zeros(args.grid_size, args.grid_size, args.sf_dim)

    for x, y in zip(pos[0], pos[1]):
        # # Load the image as a NumPy array
        img = grid.copy()
        if args.env_name in ("FourRooms", "Maze"):
            img[x, y, :] = 10  # 10 is an agent
        else:
            img[x, y, 1] = 1  # 1 is blue agent
            img[x, y, 2] = 2  # alive agent

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).to(torch.float32)
            agent_pos = torch.tensor([[x, y]]).to(
                torch.float32
            )  # .to(self._dtype).to(self.device)
            phi, _ = feaNet(img, agent_pos)
        features[x, y, :] = phi

    return features.numpy()


def get_similarity_metric(features, option_vals, options, pos, args):
    """This sweeps possible blue agent states to
    compute all options for each feature then average"""
    total_dissimilarity = 0
    total_diss_dict = {}
    feature_num = len(pos[0])

    if args.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
        # parameters
        vector_dividend = int(args.num_vector / 2)
        feature_dividend = int(args.sf_dim / 2)

        reward_options = options[:vector_dividend, :]
        state_options = options[vector_dividend:, :]

        values = [option_vals[i] for i in range(args.num_vector)]
        val_pairs = list(itertools.combinations(values, 2))

        reward_vectors = [reward_options[i, :] for i in range(vector_dividend)]
        state_vectors = [state_options[i, :] for i in range(vector_dividend)]

        reward_pairs = list(itertools.combinations(reward_vectors, 2))
        state_pairs = list(itertools.combinations(state_vectors, 2))

        reward_features = features[:, :, :feature_dividend]
        state_features = features[:, :, feature_dividend:]

        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = reward_features[x, y, :]  # F dim feature is ready
            for i, (v1, v2) in enumerate(reward_pairs):
                dissimilarity = np.abs(np.dot(current_features, (v1 - v2)))
                for key in val_pairs[i]:
                    try:
                        total_diss_dict[str(round(key.item(), 3))] += dissimilarity
                    except:
                        total_diss_dict[str(round(key.item(), 3))] = dissimilarity
                # sweep through every options for each feature
                current_dissimilarity += dissimilarity
            total_dissimilarity += current_dissimilarity / len(reward_pairs)

        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = state_features[x, y, :]  # F dim feature is ready
            for j, (v1, v2) in enumerate(state_pairs):
                dissimilarity = np.abs(np.dot(current_features, (v1 - v2)))
                for key in val_pairs[i + j + 1]:
                    try:
                        total_diss_dict[str(round(key.item(), 3))] += dissimilarity
                    except:
                        total_diss_dict[str(round(key.item(), 3))] = dissimilarity
                # sweep through every options for each feature
                current_dissimilarity += dissimilarity
            total_dissimilarity += current_dissimilarity / len(state_pairs)
    else:
        values = [option_vals[i] for i in range(args.num_vector)]
        vectors = [options[i, :] for i in range(args.num_vector)]
        val_pairs = list(itertools.combinations(values, 2))
        pairs = list(itertools.combinations(vectors, 2))
        # parameters
        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = features[x, y, :]  # F dim feature is ready
            for i, (v1, v2) in enumerate(pairs):
                dissimilarity = np.abs(np.dot(current_features, (v1 - v2)))
                for key in val_pairs[i]:
                    try:
                        total_diss_dict[str(round(key.item(), 3))] += dissimilarity
                    except:
                        total_diss_dict[str(round(key.item(), 3))] = dissimilarity
                # sweep through every options for each feature
                current_dissimilarity += dissimilarity
            total_dissimilarity += current_dissimilarity / len(pairs)

    total_dissimilarity /= feature_num
    for k, v in total_diss_dict.items():
        total_diss_dict[k] /= feature_num * (args.num_vector - 1)

    return total_dissimilarity, total_diss_dict


if __name__ == "__main__":
    algo_names = ["EigenOption", "EigenOption+", "EigenOption++", "EigenOption+++"]
    num_vectors = [6, 12, 24, 36, 48, 60, 72, 84]  # , 24, 48]

    mean_diss_dict = {}
    for algo_name in algo_names:
        mean_diss_list = []
        for num_vector in num_vectors:
            args = init_args(algo_name=algo_name, num_vector=num_vector)

            env = call_env(args)
            sf_network = call_sfNetwork(args)

            grid, pos = get_grid(args)
            feature_matrix = get_feature_matrix(sf_network.feaNet, grid, pos, args)

            n = 3
            mean_diss = 0
            dict_list = []
            for i in range(n):
                option_vals, options = get_vectors(args)
                diss, diss_dict = get_similarity_metric(
                    feature_matrix, option_vals, options, pos, args
                )
                mean_diss += diss / n
                dict_list.append(diss_dict)
            mean_diss_list.append(mean_diss)

            # Organize data by keys
            data_by_key = {key: [] for key in range(num_vector)}
            for d in dict_list:
                i = 0
                for _, value in d.items():
                    data_by_key[i].append(value)
                    i += 1

            # Compute mean and std dev for each key (if needed)
            means = {key: np.mean(values) for key, values in data_by_key.items()}
            std_devs = {key: np.std(values) for key, values in data_by_key.items()}

            # Prepare data for boxplot
            boxplot_data = [values for key, values in sorted(data_by_key.items())]
            plt.figure(figsize=(12, 8))  # Width=12, Height=8 (adjust as needed)
            plt.boxplot(boxplot_data, patch_artist=True)
            plt.title(f"Mean dissimilarity: {mean_diss} for {args.algo_name}")
            plt.tight_layout()
            plt.savefig(f"data_{args.algo_name}_{num_vector}.png")
            plt.close()

        mean_diss_dict[algo_name] = mean_diss_list

    print(mean_diss_dict)

    # Plot the results
    for k, v in mean_diss_dict.items():
        plt.plot(
            num_vectors,
            v,
            label=f"{LABEL[k]}",
            color=COLORS[k],
            linestyle=LINESTYLES[k],
            **MARKERS[k],
        )

    plt.xlabel("Number of Vectors/ Clusters", fontsize=20)
    plt.ylabel("Mean Dissimilarity per State", fontsize=16)
    plt.xticks(num_vectors, labels=[str(x) for x in num_vectors], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    # plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"FourRooms cluster.png")
    plt.close()
