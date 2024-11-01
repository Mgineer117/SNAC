import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

from utils.get_all_states import get_grid_tensor, get_grid_tensor2
from utils.utils import estimate_psi
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.policy.base_policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter
from log.wandb_logger import WandbLogger


def discover_options(
    policy: BasePolicy,
    sampler: OnlineSampler,
    grid_type: int = 0,
    algo_name: str = "SNAC-combined",
    method: str = "SVD",
    classification: str = "top",
    num: int = 10,
    gamma: int = 0.9,
    num_trj: int = 100,
    prev_batch: dict | None = None,
    idx: int | None = None,
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
    # Collect batch
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

    features = torch.from_numpy(batch["features"]).to(torch.float32).to(device)
    terminals = torch.from_numpy(batch["terminals"]).to(torch.float32).to(device)

    if algo_name == "SNAC":
        # Compute Psi
        with torch.no_grad():
            psi = estimate_psi(features, terminals, gamma)  # operate on cpu
            psi_r, psi_s = policy.split(psi)

            # to save VRAM
            del features, terminals, psi
            torch.cuda.empty_cache()

        option_dim = psi_r.shape[-1]
        if option_dim < num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{num}"
            )

        divide_num = (option_dim - 2 * num) // num
        first_indices = list(range(0, num))
        middle_indices = list(range(num, option_dim - num, divide_num))
        final_indices = list(range(option_dim - num, option_dim))

        indices = first_indices + middle_indices + final_indices

        if method == "SVD":
            # V_r, V_s ~ [F/2, F/2]
            _, S_r, V_r = torch.svd(psi_r)  # S: max - min
            _, S_s, V_s = torch.svd(psi_s)  # S: max - min

            if classification == "all":
                pass
            elif classification == "top":
                S_r = S_r[first_indices]
                S_s = S_s[first_indices]
                V_r = V_r[first_indices, :]
                V_s = V_s[first_indices, :]
            elif classification == "mid":
                S_r = S_r[middle_indices]
                S_s = S_s[middle_indices]
                V_r = V_r[middle_indices, :]
                V_s = V_s[middle_indices, :]
            elif classification == "bot":
                S_r = S_r[final_indices]
                S_s = S_s[final_indices]
                V_r = V_r[final_indices, :]
                V_s = V_s[final_indices, :]
            elif classification == "mix":
                if option_dim < 3 * num:
                    raise ValueError(
                        f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{2*num}"
                    )
                S_r = S_r[indices]
                S_s = S_s[indices]
                V_r = V_r[indices, :]
                V_s = V_s[indices, :]
            else:
                NotImplementedError(
                    f"Given classification is not implemented {classification}"
                )

            S = torch.cat((S_r, S_s), axis=0)
            V = torch.cat((V_r, V_s), axis=0)

            return S, V, [S_r, S_s], [V_r, V_s], ["R-feature", "S-feature"], batch
        else:
            NotImplementedError(f"Not implemented arg {method}")

    elif algo_name == "EigenOption2":
        with torch.no_grad():
            psi = estimate_psi(features, terminals, gamma)  # operate on cpu

            if prev_batch is not None:
                prev_features = (
                    torch.from_numpy(prev_batch["features"])
                    .to(torch.float32)
                    .to(device)
                )
                prev_terminals = (
                    torch.from_numpy(prev_batch["terminals"])
                    .to(torch.float32)
                    .to(device)
                )
                prev_psi = estimate_psi(
                    prev_features, prev_terminals, gamma
                )  # operate on cpu

                psi = torch.cat((prev_psi, psi), axis=0)

            # to save VRAM
            del features, terminals
            torch.cuda.empty_cache()

        option_dim = psi.shape[-1]
        if option_dim < 3 * num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{2*num}"
            )

        divide_num = (option_dim - 2 * num) // num
        first_indices = list(range(0, num))
        middle_indices = list(range(num, option_dim - num, divide_num))
        final_indices = list(range(option_dim - num, option_dim))

        indices = first_indices + middle_indices + final_indices

        if method == "SVD":
            # V_r, V_s ~ [F/2, F/2]
            _, S, V = torch.svd(psi)  # S: max - min

            if classification == "all":
                pass
            elif classification == "top":
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

            return S, V, [S], [V], ["S-feature"], batch

    elif algo_name == "EigenOption" or algo_name == "CoveringOption":
        with torch.no_grad():
            psi = estimate_psi(features, terminals, gamma)  # operate on cpu

            if prev_batch is not None:
                prev_features = (
                    torch.from_numpy(prev_batch["features"])
                    .to(torch.float32)
                    .to(device)
                )
                prev_terminals = (
                    torch.from_numpy(prev_batch["terminals"])
                    .to(torch.float32)
                    .to(device)
                )
                prev_psi = estimate_psi(
                    prev_features, prev_terminals, gamma
                )  # operate on cpu

                psi = torch.cat((prev_psi, psi), axis=0)

            # to save VRAM
            del features, terminals
            torch.cuda.empty_cache()

        option_dim = psi.shape[-1]
        if option_dim < 3 * num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{2*num}"
            )

        divide_num = (option_dim - 2 * num) // num
        first_indices = list(range(0, num))
        middle_indices = list(range(num, option_dim - num, divide_num))
        final_indices = list(range(option_dim - num, option_dim))

        indices = first_indices + middle_indices + final_indices

        if method == "SVD":
            # V_r, V_s ~ [F/2, F/2]
            _, S, V = torch.svd(psi)  # S: max - min

            if classification == "all":
                pass
            elif classification == "top":
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

            S = torch.cat((S, -S), axis=0)
            V = torch.cat((V, -V), axis=0)

            return S, V, [S], [V], ["S-feature"], batch
        else:
            NotImplementedError(f"Not implemented arg {method}")
    else:
        raise ValueError(f"Unknown algo-name: {algo_name}")


def cluster_vecvtors(S_list, V_list, k=10):
    """V must be matrix of linear functionals: V_h"""
    k = int(k / 2)

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
        eigVals = np.array(sorted(eigVals, reverse=True))
        centroids = centroids[sorted_indices, :]

        eigVals = np.concatenate((eigVals, -eigVals), axis=0)
        centroids = np.concatenate((centroids, -centroids), axis=0)

        centroids_list.append(centroids)
        labels_list.append(labels)
        eigVals_list.append(eigVals)

    eigVals = np.concatenate(eigVals_list)
    eigVals = torch.from_numpy(eigVals)

    centroids = np.concatenate(centroids_list, axis=0)
    centroids = torch.from_numpy(centroids)

    return (
        eigVals,
        centroids,
        {"centroids_list": centroids_list, "labels_list": labels_list},
    )


def get_eigenvectors(
    env,
    network,
    sampler,
    plotter,
    args,
    idx: int | None = None,
    app_trj_num: int = 100,
    prev_batch: dict | None = None,
    draw_map: bool = False,
):
    if args.algo_name == "SNAC":
        option_vals, options, S_list, V_list, names, batch = discover_options(
            policy=network,
            sampler=sampler,
            grid_type=args.grid_type,
            algo_name=args.algo_name,
            method="SVD",
            classification="all",
            num_trj=app_trj_num,
            device=args.device,
        )

        option_vals, options, metaData = cluster_vecvtors(
            S_list, V_list, k=int(args.num_vector / 2)
        )  # replacing original V with cluster centroids

        print(
            f"Selecting clustered R:{int(len(option_vals)/2)}  S:{int(len(option_vals)/2)} vector !!! | Given total options: {args.num_vector}"
        )
        if draw_map:
            plotter.plotClusteredVectors(
                V_list=V_list,
                centroids=metaData["centroids_list"],
                labels=metaData["labels_list"],
                names=names,
                dir=plotter.sf_path,
            )
    elif args.algo_name == "EigenOption":
        option_vals, options, S_list, V_list, names, batch = discover_options(
            policy=network,
            sampler=sampler,
            grid_type=args.grid_type,
            algo_name=args.algo_name,
            method="SVD",
            classification="top",
            num=int(args.num_vector / 2),
            num_trj=app_trj_num,
            device=args.device,
        )
        print(
            f"Selecting top {len(option_vals)} vector!!! | Given total options: {args.num_vector}"
        )
    elif args.algo_name == "EigenOption2":
        option_vals, options, S_list, V_list, names, batch = discover_options(
            policy=network,
            sampler=sampler,
            grid_type=args.grid_type,
            algo_name=args.algo_name,
            method="SVD",
            classification="all",
            num_trj=app_trj_num,
            device=args.device,
        )
        option_vals, options, metaData = cluster_vecvtors(
            S_list, V_list, k=args.num_vector
        )  # replacing original V with cluster centroids
        print(
            f"Selecting clustered {len(option_vals)} vector!!! | Given total options: {args.num_vector}"
        )
        if draw_map:
            plotter.plotClusteredVectors(
                V_list=V_list,
                centroids=metaData["centroids_list"],
                labels=metaData["labels_list"],
                names=names,
                dir=plotter.sf_path,
            )
    elif args.algo_name == "CoveringOption":
        """It is separately implemented in CoveringOption class"""
        pass
    else:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

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
