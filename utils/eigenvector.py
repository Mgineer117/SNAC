import torch
import numpy as np
from sklearn.cluster import KMeans

from log.logger_util import colorize
from utils.get_all_states import get_grid_tensor
from utils.utils import estimate_psi
from utils.plotter import Plotter
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.policy.base_policy import BasePolicy


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


def vector(evals, evecs, option_dim: int, num: int, classification: str):
    divide_num = (option_dim - 2 * num) // num

    first_indices = list(range(0, num))
    middle_indices = list(range(num, option_dim - num, divide_num))
    final_indices = list(range(option_dim - num, option_dim))
    indices = first_indices + middle_indices + final_indices

    if classification == "top":
        evals = evals[first_indices]
        evecs = evecs[first_indices, :]
    elif classification == "mid":
        evals = evals[middle_indices]
        evecs = evecs[middle_indices, :]
    elif classification == "bot":
        evals = evals[final_indices]
        evecs = evecs[final_indices, :]
    elif classification == "mix":
        evals = evals[indices]
        evecs = evecs[indices, :]
    else:
        NotImplementedError(f"Given classification is not implemented {classification}")
    return evals, evecs


def discover_options(
    policy: BasePolicy,
    sampler: OnlineSampler,
    plotter: Plotter,
    grid_type: int,
    env_name: str,
    algo_name: str,
    num: int = 10,
    gamma: int = 0.99,
    num_trj: int = 100,
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
        min_num_trj=num_trj, max_num_trj=500, device=device
    )
    while option_buffer.num_trj < option_buffer.min_num_trj:
        batch, sample_time = sampler.collect_samples(
            policy, grid_type=grid_type, idx=idx, is_covering_option=is_covering_option
        )
        option_buffer.push(batch)
    batch = option_buffer.sample_all()
    option_buffer.wipe()

    obs = {"observation": batch["states"], "agent_pos": batch["agent_pos"]}
    features, _ = policy.get_features(obs, to_numpy=True)

    ### Convert to the tensor
    features = torch.from_numpy(features).to(torch.float32)
    terminals = torch.from_numpy(batch["terminals"]).to(torch.float32)

    decomp_psi = True
    if decomp_psi:
        #### Compute Psi from Phi
        with torch.no_grad():
            psi = estimate_psi(features, terminals, gamma)  # operate on cpu

        # to save VRAM
        del features, terminals
    else:
        psi = features.clone()

    ### Compute the vectors via SVD
    if algo_name in ("SNAC", "SNAC+", "SNAC++"):
        psi_r, psi_s = policy.split(psi)

        option_dim = psi_r.shape[-1]

        if option_dim < 3 * num:
            raise ValueError(
                f"The number of eigenvectors smaller than what you are to sample!!{option_dim}<{3*num}"
            )

        _, evals_r, evecs_r = torch.svd(psi_r)  # F/2, F/2 x F/2
        _, evals_s, evecs_s = torch.svd(psi_s)  # F/2, F/2 x F/2

        if algo_name == "SNAC":
            S_r, V_r = vector(
                evals_r, evecs_r, option_dim=option_dim, num=num, classification="top"
            )
            S_s, V_s = vector(
                evals_s, evecs_s, option_dim=option_dim, num=num, classification="top"
            )

            S_r = torch.cat((S_r, -S_r), axis=0)
            V_r = torch.cat((V_r, -V_r), axis=0)
            S_s = torch.cat((S_s, -S_s), axis=0)
            V_s = torch.cat((V_s, -V_s), axis=0)

            option_vals = torch.cat((S_r, S_s), dim=0)
            options = torch.cat((V_r, V_s), dim=0)
        elif algo_name == "SNAC+":
            # replacing original V with cluster centroids
            _, _, metaData = cluster_vecvtors(
                [evals_r, evals_s], [evecs_r, evecs_s], k=num
            )

            # Pull out the cluster result
            S_r, S_s = metaData["evals_list"]
            V_r, V_s = metaData["centroids_list"]

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
            S_list = [evals_r, evals_s]
            V_list = [evecs_r, evecs_s]

            r_rewards = (
                evecs_r @ psi_r.T
            )  # F x T (Num options (row) and rewards (column))
            s_rewards = evecs_s @ psi_s.T  # F x T

            # S_r, S_s are the dummy input since we just want clustered indicies
            _, _, metaData = cluster_vecvtors(
                [evals_r, evals_s], [r_rewards, s_rewards], k=num
            )

            val_list = []
            vec_list = []
            for i, label in enumerate(metaData["labels_list"]):
                values = torch.empty(num)
                vectors = torch.empty(num, option_dim)
                for k in range(num):
                    idx = label == k
                    values[k] = torch.mean(S_list[i][idx])
                    vectors[k, :] = torch.mean(V_list[i][idx, :], axis=0)

                S, indices = torch.sort(values, descending=True)
                V = vectors[indices]

                S = torch.cat((S, -S), axis=0)
                V = torch.cat((V, -V), axis=0)

                val_list.append(S)
                vec_list.append(V)

            option_vals = torch.cat(val_list, dim=0)
            options = torch.cat(vec_list, dim=0)

        if algo_name in ("SNAC+", "SNAC++") and draw_map:
            plotter.plotClusteredVectors(
                V_list=[evecs_r, evecs_s],
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

        _, evals, evecs = torch.svd(psi)  # S: max - min

        if algo_name == "EigenOption":
            ##### top n vectors #####
            S, V = vector(
                evals, evecs, option_dim=option_dim, num=num, classification="top"
            )

            option_vals = torch.cat((S, -S), axis=0)
            options = torch.cat((V, -V), axis=0)
        elif algo_name == "EigenOption+":
            ##### cluster in eigen space #####
            S, V, metaData = cluster_vecvtors([evals], [evecs], k=num)

            # Obtain (+/-) of vectors
            option_vals = torch.cat((S, -S), axis=0)
            options = torch.cat((V, -V), axis=0)
        elif algo_name == "EigenOption++":
            ##### cluster in action-value space #####
            rewards = evecs @ psi.T  # (num options) x T

            _, _, metaData = cluster_vecvtors([evals], [rewards], k=num)  # S is dummy

            values = torch.empty(num)
            vectors = torch.empty(num, option_dim)
            for k in range(num):
                idx = metaData["labels_list"][0] == k

                values[k] = torch.mean(evals[idx])
                vectors[k, :] = torch.mean(evecs[idx, :], axis=0)

            S, indices = torch.sort(values, descending=True)
            V = vectors[indices]

            option_vals = torch.cat((S, -S), axis=0)
            options = torch.cat((V, -V), axis=0)

        if algo_name in ("EigenOption+", "EigenOption++") and draw_map:
            plotter.plotClusteredVectors(
                V_list=[evecs],
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
    sf_network,
    sampler,
    plotter,
    args,
    draw_map: bool = False,
):
    if args.algo_name in ("SNAC", "SNAC+", "SNAC++"):
        option_vals, options, batch = discover_options(
            policy=sf_network,
            sampler=sampler,
            plotter=plotter,
            grid_type=args.grid_type,
            env_name=args.env_name,
            algo_name=args.algo_name,
            num=int(args.num_vector / 2),
            num_trj=args.num_traj_decomp,
            draw_map=draw_map,
            gamma=args.gamma,
            device=args.device,
        )
        print_option_info(option_vals, options, args.algo_name, args.num_vector)
    elif args.algo_name in ("EigenOption", "EigenOption+", "EigenOption++"):
        option_vals, options, batch = discover_options(
            policy=sf_network,
            sampler=sampler,
            plotter=plotter,
            grid_type=args.grid_type,
            env_name=args.env_name,
            algo_name=args.algo_name,
            num=args.num_vector,
            num_trj=args.num_traj_decomp,
            draw_map=draw_map,
            gamma=args.gamma,
            device=args.device,
        )
        print_option_info(option_vals, options, args.algo_name, args.num_vector)
    elif args.algo_name == "CoveringOption":
        """It is separately implemented in CoveringOption class"""
        pass

    if draw_map:
        if args.env_name == "FourRooms":
            grid_tensor, coords, agent_pos = get_grid_tensor(env, args.grid_type)
            plotter.plotRewardMap(
                feaNet=sf_network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                agent_pos=agent_pos,
                dir=plotter.log_dir,
                device=args.device,
            )
        elif args.env_name == "LavaRooms":
            grid_tensor, coords, agent_pos = get_grid_tensor(env, args.grid_type)
            plotter.plotRewardMap(
                feaNet=sf_network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                agent_pos=agent_pos,
                dir=plotter.log_dir,
                device=args.device,
            )
        elif args.env_name == "CtF1v1" or args.env_name == "CtF1v2":
            grid_tensor, coords, agent_pos = get_grid_tensor(env, args.grid_type)
            plotter.plotRewardMap2(
                feaNet=sf_network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,  # since V = [V, -V]
                algo_name=args.algo_name,
                grid_tensor=grid_tensor,
                coords=coords,
                agent_pos=agent_pos,
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
