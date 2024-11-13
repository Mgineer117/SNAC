import numpy as np
import torch
import json
import itertools

from utils import *
from utils.call_env import call_env
from models.evaulators.base_evaluator import DotDict


def init_args(num_vector):
    model_dir = "log/eval_log/model_for_eval/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.import_sf_model = True
    args.s_dim = tuple(args.s_dim)
    args.device = torch.device("cpu")

    args.num_vector = num_vector

    print(f"Algo name: {args.algo_name}")
    print(f"Env name: {args.env_name}")

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
        min_option_length=args.min_option_length,
        min_cover_option_length=args.min_cover_option_length,
        episode_len=args.episode_len,
        episode_num=args.episode_num,
        num_cores=args.num_cores,
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
    if args.env_name == "FourRooms" or args.env_name == "LavaRooms":
        grid, pos, loc = get_grid_tensor(env, grid_type=args.grid_type)
    else:  # args.env_name == "CtF1v1" or args.env_name == "CtF1v2"
        grid, pos, loc = get_grid_tensor2(env, grid_type=args.grid_type)

    return grid, pos


def get_feature_matrix(feaNet, grid, pos, args):
    features = torch.zeros(args.grid_size, args.grid_size, args.sf_dim)

    for x, y in zip(pos[0], pos[1]):
        # # Load the image as a NumPy array
        img = grid.copy()
        if args.env_name == "FourRooms" or args.env_name == "LavaRooms":
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


def get_similarity_metric(features, options, pos, args):
    """This sweeps possible blue agent states to
    compute all options for each feature then average"""
    # print(f"option dims: {options.shape}")
    # print(f"feature dim: {features[0,0,:].shape}")
    # print(f"vector dim: {(options[0,:] - options[1, :]).shape}")

    total_dissimilarity = 0
    feature_num = len(pos[0])

    if args.algo_name == "SNAC":
        # parameters
        vector_dividend = int(args.num_vector / 2)
        feature_dividend = int(args.sf_dim / 2)

        reward_options = options[:vector_dividend, :]
        state_options = options[vector_dividend:, :]

        reward_vectors = [reward_options[i, :] for i in range(vector_dividend)]
        state_vectors = [state_options[i, :] for i in range(vector_dividend)]

        reward_pairs = list(itertools.combinations(reward_vectors, 2))
        state_pairs = list(itertools.combinations(state_vectors, 2))

        reward_features = features[:, :, :feature_dividend]
        state_features = features[:, :, feature_dividend:]

        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = reward_features[x, y, :]  # F dim feature is ready
            for v1, v2 in reward_pairs:
                # sweep through every options for each feature
                current_dissimilarity += np.abs(np.dot(current_features, (v1 - v2)))
            total_dissimilarity += current_dissimilarity / len(reward_pairs)

        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = state_features[x, y, :]  # F dim feature is ready
            for v1, v2 in state_pairs:
                # sweep through every options for each feature
                current_dissimilarity += np.abs(np.dot(current_features, (v1 - v2)))
            total_dissimilarity += current_dissimilarity / len(state_pairs)
    else:
        vectors = [options[i, :] for i in range(args.num_vector)]
        pairs = list(itertools.combinations(vectors, 2))
        # parameters
        for x, y in zip(pos[0], pos[1]):
            current_dissimilarity = 0
            current_features = features[x, y, :]  # F dim feature is ready
            for v1, v2 in pairs:
                # sweep through every options for each feature
                current_dissimilarity += np.abs(np.dot(current_features, (v1 - v2)))
            total_dissimilarity += current_dissimilarity / len(pairs)

    return total_dissimilarity / feature_num


if __name__ == "__main__":
    args = init_args(num_vector=8)

    env = call_env(args)
    sf_network = call_sfNetwork(args)

    grid, pos = get_grid(args)
    feature_matrix = get_feature_matrix(sf_network.feaNet, grid, pos, args)

    n = 10
    diss_list = []
    for i in range(n):
        option_vals, options = get_vectors(args)
        diss = get_similarity_metric(feature_matrix, options, pos, args)
        diss_list.append(diss)
    print(f"Dissimilarity: {np.mean(diss_list)}")
