import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

from log.logger_util import colorize
from utils.get_all_states import get_grid_tensor, get_grid_tensor2
from utils.utils import estimate_psi
from utils.plotter import Plotter
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.policy.base_policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter
from log.wandb_logger import WandbLogger


def print_option_info(option_vals, options, algo_name, desired_num):
    if algo_name in ("SNAC", "SNAC+", "SNAC++"):
        vec_num = int(len(option_vals) / 2)

        msg = colorize(
            f"{algo_name} with R:{vec_num} S:{vec_num} / {desired_num} vectors with shape {options.shape}",
            "magenta",
            bold=True,
        )
        print(msg)

    elif algo_name in ("EigenOption", "EigenOption+", "EigenOption++"):
        vec_num = len(option_vals)
        msg = colorize(
            f"{algo_name} with {vec_num} / {desired_num} vectors with shape {options.shape}",
            "magenta",
            bold=True,
        )
        print(msg)


def discover_options(
    policy: BasePolicy,
    sampler: OnlineSampler,
    plotter: Plotter,
    grid_type: int = 0,
    algo_name: str = "SNAC",
    num: int = 10,
    gamma: int = 0.9,
    num_trj: int = 100,
    prev_batch: dict | None = None,
    idx: int | None = None,
    draw_map: bool = False,
    device=torch.device("cpu"),
):
    """
    policy: sf_network
    method: SVD
    classification:
        all: return all e-vecs
        top: top 10 eigenvectors,
        mid: every 10th mid vectors except top and bot,
        bot: bot 10 eigenvectors,
        mix: all of the above
    """
    # since we want to sample (+/-) pairs
    num = int(num / 2)

    ### Collect batch to compute phi for psi
    is_covering_option = True if idx is not None else False
    option_buffer = TrajectoryBuffer(
        min_num_trj=num_trj, max_num_trj=200, device=device
    )
    while option_buffer.num_trj < option_buffer.min_num_trj:
        batch, sample_time = sampler.collect_samples(
            policy, grid_type=grid_type, idx=idx, is_covering_option=is_covering_option
        )
        option_buffer.push(batch)
    batch = option_buffer.sample_all()
    option_buffer.wipe()

    ### Convert to the tensor
    features = torch.from_numpy(batch["features"]).to(torch.float32).to(device)
    terminals = torch.from_numpy(batch["terminals"]).to(torch.float32).to(device)

    #### Compute Psi from Phi
    with torch.no_grad():
        psi = estimate_psi(features, terminals, gamma)  # operate on cpu
        if prev_batch is not None:
            prev_features = (
                torch.from_numpy(prev_batch["features"]).to(torch.float32).to(device)
            )
            prev_terminals = (
                torch.from_numpy(prev_batch["terminals"]).to(torch.float32).to(device)
            )
            prev_psi = estimate_psi(
                prev_features, prev_terminals, gamma
            )  # operate on cpu

            psi = torch.cat((prev_psi, psi), axis=0)
        psi_r, psi_s = policy.split(psi)
        # to save VRAM
        del features, terminals
        torch.cuda.empty_cache()

    def vector(S, V, classification):
        divide_num = (option_dim - 2 * num) // num

        first_indices = list(range(0, num))
        middle_indices = list(range(num, option_dim - num, divide_num))
        final_indices = list(range(option_dim - num, option_dim))
        indices = first_indices + middle_indices + final_indices

        if classification == "top":
            S = S[first_indices]
            V = V[first_indices, :]
        elif classification == "mid":
            S = S[middle_indices]
            V = V[middle_indices, :]
        elif classification == "bot":
            S = S[final_indices]
            V = V[final_indices, :]
        elif classification == "mix":
            S = S[indices]
            V = V[indices, :]
        else:
            NotImplementedError(
                f"Given classification is not implemented {classification}"
            )
        return S, V

    ### Compute the vectors via SVD
    if algo_name in ("SNAC", "SNAC+", "SNAC++"):
        option_dim = psi_r.shape[-1]
        if option_dim < 3 * num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{3*num}"
            )

        _, S_r, V_r = torch.svd(psi_r)  # F/2, F/2 x F/2
        _, S_s, V_s = torch.svd(psi_s)  # F/2, F/2 x F/2

        if algo_name == "SNAC":
            S_r, V_r = vector(S_r, V_r, classification="top")
            S_s, V_s = vector(S_s, V_s, classification="top")

            S_r = torch.cat((S_r, -S_r), axis=0)
            V_r = torch.cat((V_r, -V_r), axis=0)
            S_s = torch.cat((S_s, -S_s), axis=0)
            V_s = torch.cat((V_s, -V_s), axis=0)

            option_vals = torch.cat((S_r, S_s), dim=0)
            options = torch.cat((V_r, V_s), dim=0)
        elif algo_name == "SNAC+":
            V_list = [V_r, V_s]

            # replacing original V with cluster centroids
            _, _, metaData = cluster_vecvtors([S_r, S_s], [V_r, V_s], k=num)

            # Pull out the cluster result
            S_r = metaData["evals_list"][0]
            S_s = metaData["evals_list"][1]
            V_r = metaData["centroids_list"][0]
            V_s = metaData["centroids_list"][1]

            # Obtain (+/-) of vectors
            S_r = torch.cat((S_r, -S_r), axis=0)
            V_r = torch.cat((V_r, -V_r), axis=0)
            S_s = torch.cat((S_s, -S_s), axis=0)
            V_s = torch.cat((V_s, -V_s), axis=0)

            option_vals = torch.cat((S_r, S_s), dim=0)
            options = torch.cat((V_r, V_s), dim=0)

            vec_list = [V_r, V_s]
        elif algo_name == "SNAC++":
            # V_list = [torch.cat((V_r, -V_r), axis=0), torch.cat((V_s, -V_s), axis=0)]
            S_list = [S_r, S_s]
            V_list = [V_r, V_s]

            r_rewards = V_r @ psi_r.T  # F x T (Num options (row) and rewards (column))
            s_rewards = V_s @ psi_s.T  # F x T

            # S_r, S_s are the dummy input since we just want clustered indicies
            _, _, metaData = cluster_vecvtors([S_r, S_s], [r_rewards, s_rewards], k=num)

            val_list = []
            vec_list = []
            for i, label in enumerate(metaData["labels_list"]):
                vals = torch.empty(num)
                vecs = torch.empty(num, option_dim)
                for k in range(num):
                    idx = label == k
                    vals[k] = torch.mean(S_list[i][idx])
                    vecs[k, :] = torch.mean(V_list[i][idx, :], axis=0)

                sorted_vals, indices = torch.sort(vals, descending=True)
                sorted_vecs = vecs[indices]

                sorted_vals = torch.cat((sorted_vals, -sorted_vals), axis=0)
                sorted_vecs = torch.cat((sorted_vecs, -sorted_vecs), axis=0)

                val_list.append(sorted_vals)
                vec_list.append(sorted_vecs)

            option_vals = torch.cat(val_list, dim=0)
            options = torch.cat(vec_list, dim=0)

        if algo_name in ("SNAC+", "SNAC++") and draw_map:
            plotter.plotClusteredVectors(
                V_list=V_list,
                centroids=vec_list,
                labels=metaData["labels_list"],
                names=["R-feature", "S-feature"],
                dir=plotter.sf_path,
            )

    elif algo_name in ("EigenOption", "EigenOption+", "EigenOption++"):
        option_dim = psi.shape[-1]
        if option_dim < 3 * num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{3*num}"
            )

        _, S, V = torch.svd(psi)  # S: max - min

        if algo_name == "EigenOption":
            ##### top n vectors #####
            S, V = vector(S, V, classification="top")

            option_vals = torch.cat((S, -S), axis=0)
            options = torch.cat((V, -V), axis=0)
        elif algo_name == "EigenOption+":
            V_list = [V]

            ##### cluster in eigen space #####
            S, V, metaData = cluster_vecvtors([S], [V], k=num)

            # Obtain (+/-) of vectors
            option_vals = torch.cat((S, -S), axis=0)
            options = torch.cat((V, -V), axis=0)
        elif algo_name == "EigenOption++":
            V_list = [V]

            ##### cluster in action-value space #####
            rewards = V @ psi.T  # (num options) x T

            _, _, metaData = cluster_vecvtors([S], [rewards], k=num)  # S is dummy

            vals = torch.empty(num)
            vecs = torch.empty(num, option_dim)

            for k in range(num):
                idx = metaData["labels_list"][0] == k

                vals[k] = torch.mean(S[idx])
                vecs[k, :] = torch.mean(V[idx, :], axis=0)

            sorted_vals, indices = torch.sort(vals, descending=True)
            sorted_vecs = vecs[indices]

            option_vals = torch.cat((sorted_vals, -sorted_vals), axis=0)
            options = torch.cat((sorted_vecs, -sorted_vecs), axis=0)

        if algo_name in ("EigenOption+", "EigenOption++") and draw_map:
            plotter.plotClusteredVectors(
                V_list=V_list,
                centroids=[options],
                labels=metaData["labels_list"],
                names=["S-feature"],
                dir=plotter.sf_path,
            )
    else:
        pass

    return option_vals, options, batch


def cluster_vecvtors(S_list, V_list, k=10):
    """V must be matrix of linear functionals: V_h"""
    centroids_list = []
    labels_list = []
    eigVals_list = []

    for S, V in zip(S_list, V_list):
        S = S.cpu().numpy()
        V = V.cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=0)

        kmeans.fit(V)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        eigVals = []
        for i in range(k):
            eigVal = S[labels == i]
            eigVal = np.mean(eigVal)
            eigVals.append(eigVal)
        sorted_indices = sorted(
            range(len(eigVals)), key=lambda i: eigVals[i], reverse=True
        )
        eigVals = torch.tensor(sorted(eigVals, reverse=True))
        centroids = torch.tensor(centroids[sorted_indices, :])

        centroids_list.append(centroids)
        labels_list.append(labels)
        eigVals_list.append(eigVals)

    eigVals = torch.cat(eigVals_list)
    centroids = torch.cat(centroids_list, axis=0)

    return (
        eigVals,
        centroids,
        {
            "evals_list": eigVals_list,
            "centroids_list": centroids_list,
            "labels_list": labels_list,
        },
    )


def get_eigenvectors(
    env,
    network,
    sampler,
    plotter,
    args,
    # idx: int | None = None,
    app_trj_num: int = 100,
    # prev_batch: dict | None = None,
    draw_map: bool = False,
):
    if args.algo_name in ("SNAC", "SNAC+", "SNAC++"):
        option_vals, options, batch = discover_options(
            policy=network,
            sampler=sampler,
            plotter=plotter,
            grid_type=args.grid_type,
            algo_name=args.algo_name,
            num=int(args.num_vector / 2),
            num_trj=app_trj_num,
            draw_map=draw_map,
            device=args.device,
        )
        print_option_info(option_vals, options, args.algo_name, args.num_vector)
    elif args.algo_name in ("EigenOption", "EigenOption+", "EigenOption++"):
        option_vals, options, batch = discover_options(
            policy=network,
            sampler=sampler,
            plotter=plotter,
            grid_type=args.grid_type,
            algo_name=args.algo_name,
            num=args.num_vector,
            num_trj=app_trj_num,
            draw_map=draw_map,
            device=args.device,
        )
        print_option_info(option_vals, options, args.algo_name, args.num_vector)
    elif args.algo_name == "CoveringOption":
        """It is separately implemented in CoveringOption class"""
        pass

    if draw_map:
        if args.env_name == "FourRooms":
            grid_tensor, coords, loc = get_grid_tensor(env, args.grid_type)
            plotter.plotRewardMap(
                feaNet=network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                loc=loc,
                dir=plotter.log_dir,
                device=args.device,
            )
        elif args.env_name == "LavaRooms":
            grid_tensor, coords, loc = get_grid_tensor(env, args.grid_type)
            plotter.plotRewardMap(
                feaNet=network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                loc=loc,
                dir=plotter.log_dir,
                device=args.device,
            )
        elif args.env_name == "CtF1v1" or args.env_name == "CtF1v2":
            grid_tensor, coords, loc = get_grid_tensor2(env, args.grid_type)
            plotter.plotRewardMap2(
                feaNet=network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                loc=loc,
                dir=plotter.log_dir,
                device=args.device,
            )
        ### action value map is shown via network
        ### therefore it is noisy so not being used
        # plotter.plotActionValueMap(
        #     feaNet=sf_network.feaNet,
        #     psiNet=sf_network.psiNet,
        #     S=option_vals,
        #     V=options,
        #     z=None,
        #     grid_tensor=grid_tensor,
        #     coords=coords,
        #     loc=loc,
        #     specific_path="randomPsi",
        # )

    return option_vals, options, batch
