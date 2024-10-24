import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE
from typing import Optional, Dict, List


class Plotter:
    def __init__(
        self,
        grid_size=19,
        img_tile_size=32,
        sf_path: str | None = None,
        ppo_path: str | None = None,
        op_path: str | None = None,
        ug_path: str | None = None,
        hc_path: str | None = None,
        log_dir: str | None = None,
        device=torch.device("cpu"),
    ):
        """
        This is plotter function where every methods below receives all information from the Evaluator class
        """
        self.grid_size = grid_size
        self.img_tile_size = img_tile_size

        self.sf_path = sf_path
        self.ppo_path = ppo_path
        self.op_path = op_path
        self.ug_path = ug_path
        self.hc_path = hc_path
        self.log_dir = log_dir

        self._dtype = torch.float32
        self.device = device

    def plotEigenFunction1(self, eigenvectors: np.ndarray, dir: str, epoch: int):
        """3D plot of the basis functions. Each coordinate of the eigenvector corresponds
        to the value to be plotted for the corresponding state."""

        # Ensure the vector length is a perfect square
        vector_length = eigenvectors.shape[0]
        grid_size = int(np.sqrt(vector_length))

        if grid_size**2 != vector_length:
            raise ValueError(
                "The length of the eigenvector must be a perfect square to reshape into a grid."
            )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Reshape the eigenvector to a 2D grid
        grid = eigenvectors[:, 0].reshape((grid_size, grid_size))

        # Create grid coordinates
        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        Z = grid

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("jet"))

        # Customize the view angle
        ax.view_init(elev=30, azim=30)

        # Save the plot
        plt.savefig(f"{dir}/{epoch}_eig.png")
        plt.close()

    def plotEigenFunctionAll(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        dir: str,
        epoch: int = None,
    ):
        """3D plot of the basis functions. Each coordinate of the eigenvector corresponds
        to the value to be plotted for the corresponding state."""

        # Ensure the vector length is a perfect square
        vector_length = eigenvectors.shape[0]
        grid_size = int(np.sqrt(vector_length))

        if grid_size**2 != vector_length:
            raise ValueError(
                "The length of the eigenvector must be a perfect square to reshape into a grid."
            )

        for i in range(len(eigenvalues)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Reshape the eigenvector to a 2D grid
            grid = eigenvectors[:, i].reshape((grid_size, grid_size))

            # Create grid coordinates
            X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            Z = grid

            # Plot the surface
            surf = ax.plot_surface(
                X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("jet")
            )

            # Customize the view angle
            ax.view_init(elev=30, azim=30)

            # Save the plot
            plt.savefig(f"{dir}/{i}_eig.png")
            plt.close()

        # Plot the eigenvalues
        plt.figure()
        plt.plot(eigenvalues, "o")
        plt.savefig(f"{dir}/eigenvalues.png")
        plt.close()

    def plotFeature(
        self,
        grid: np.ndarray,
        grid_r: np.ndarray,
        grid_s: np.ndarray,
        grid_v: np.ndarray,
        grid_q: np.ndarray,
        dir: str,
        epoch: int,
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        r_img = grid + grid_r
        r_img = r_img / np.max(r_img)
        axes[0, 0].imshow(r_img)
        axes[0, 0].axis("off")  # Turn off axis labels
        axes[0, 0].set_title("phi_r")

        # Plot the second image in the second subplot
        s_img = grid + grid_s
        s_img = s_img / np.max(s_img)
        axes[1, 0].imshow(s_img)
        axes[1, 0].axis("off")  # Turn off axis labels
        axes[1, 0].set_title("phi_s")

        # Plot the second image in the second subplot
        grid_v = grid_v / np.max(grid_v)
        v_img = grid + grid_v
        v_img = np.clip(v_img, 0, 1.0)
        axes[0, 1].imshow(v_img)
        axes[0, 1].axis("off")  # Turn off axis labels
        axes[0, 1].set_title("Visitation")

        q_img = grid + grid_q
        q_img = q_img / np.max(q_img)
        axes[1, 1].imshow(q_img)
        axes[1, 1].axis("off")  # Turn off axis labels
        axes[1, 1].set_title("q")

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(
            f"{dir}/{epoch}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    def plotPath(
        self,
        grid: np.ndarray,
        path: List,
        dir: str,
        epoch: int,
    ):
        img_size = grid.shape[0]

        if not os.path.exists(dir):
            os.mkdir(dir)
        grid = np.rot90(grid, k=1)
        plt.imshow(grid, origin="upper")
        plt.axis("off")

        path_length = len(path[:-1]) - 1
        idx = 0
        for i_point, f_point in zip(path[:-1], path[1:]):
            y = (
                i_point[0] * self.img_tile_size + self.img_tile_size / 2,
                f_point[0] * self.img_tile_size + self.img_tile_size / 2,
            )
            x = (
                i_point[1] * self.img_tile_size + self.img_tile_size / 2,
                f_point[1] * self.img_tile_size + self.img_tile_size / 2,
            )

            y = [img_size - i for i in y]

            if idx == 0:
                plt.scatter(x[0], y[0], color="red", s=30)
            if idx == path_length:
                plt.scatter(x[1], y[1], color="blue", s=30)

            # list bool comparison is error
            if not isinstance(i_point, tuple):
                i_point = tuple(i_point)
            if not isinstance(f_point, tuple):
                f_point = tuple(f_point)

            if i_point != f_point:
                plt.plot(x, y, color="green", linewidth=2)
            idx += 1

        plt.tight_layout()
        plt.savefig(
            f"{dir}/{epoch}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    def plotPath2(
        self,
        grid: np.ndarray,
        paths: List,
        dir: str,
        epoch: int,
    ):
        if not os.path.exists(dir):
            os.mkdir(dir)

        img_size = grid.shape[0]
        grid = np.rot90(grid, k=1)
        plt.imshow(grid, origin="upper")
        plt.axis("off")

        positional_colors = ["blue", "red"]
        colors = ["green", "yellow"]
        for index, path in enumerate(paths):
            path_length = len(path[:-1]) - 1
            idx = 0
            for i_point, f_point in zip(path[:-1], path[1:]):
                y = (
                    i_point[0] * self.img_tile_size
                    + self.img_tile_size / 2
                    + 3 * index,
                    f_point[0] * self.img_tile_size
                    + self.img_tile_size / 2
                    + 3 * index,
                )
                x = (
                    i_point[1] * self.img_tile_size
                    + self.img_tile_size / 2
                    + 3 * index,
                    f_point[1] * self.img_tile_size
                    + self.img_tile_size / 2
                    + 3 * index,
                )
                y = [img_size - i for i in y]
                if idx == 0:
                    plt.scatter(x[0], y[0], color=positional_colors[index], s=30)
                if idx == path_length:
                    plt.scatter(x[1], y[1], color=positional_colors[index], s=30)

                # list bool comparison is error
                if not isinstance(i_point, tuple):
                    i_point = tuple(i_point)
                if not isinstance(f_point, tuple):
                    f_point = tuple(f_point)

                if i_point != f_point:
                    plt.plot(x, y, color=colors[index], linewidth=2, alpha=0.7)
                idx += 1

        # Apply rotation to the whole plot after tight_layout
        rotation = Affine2D().rotate_deg(90)
        plt.gca().set_transform(rotation + plt.gca().transAxes)  # Rotate entire axis

        plt.tight_layout()
        plt.savefig(
            f"{dir}/{epoch}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    def plotRendering(
        self,
        frames: List,
        dir: str,
        epoch: int,
        fps: int = 10,
        width: int = 608,
        height: int = 608,
    ):
        # high fps -> faster
        file_name = str(epoch) + ".avi"
        output_file = os.path.join(dir, file_name)
        if not os.path.exists(dir):
            os.mkdir(dir)
        fourcc = cv2.VideoWriter_fourcc(
            *"MJPG"
        )  # Try different codecs like MJPG or MP4V

        out = cv2.VideoWriter(output_file, fourcc, fps, (height, width))
        for frame in frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

    def plotClusteredVectors(self, V_list, centroids, labels, names: List, dir: str):
        for vector, centroid, label, name in zip(V_list, centroids, labels, names):
            vector = vector.cpu().numpy()

            k = centroid.shape[0]

            tsne = TSNE(n_components=2, random_state=0)

            data = np.concatenate((vector, centroid), axis=0)
            data_2d = tsne.fit_transform(data)

            centroid_2d = data_2d[-k:, :]
            data_2d = data_2d[:-k, :]
            plt.figure(figsize=(8, 6))

            plt.scatter(
                data_2d[:, 0],
                data_2d[:, 1],
                c=label,
                cmap="viridis",
                label="Data Points",
            )
            plt.scatter(
                centroid_2d[:, 0],
                centroid_2d[:, 1],
                c="red",
                marker="x",
                s=100,
                label="Centroids",
            )  # Plot centroids

            plt.colorbar()
            plt.title(f"{name} Clustering with {k} Clusters")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend()

            plt.savefig(dir + f"/{name}Vectors.png")
            plt.close()

    def plotRewardMap(
        self,
        feaNet: nn.Module,
        S: torch.Tensor,
        V: torch.Tensor,
        feature_dim: int,
        algo_name: str,
        grid_tensor: np.ndarray,
        coords: tuple,
        loc: np.ndarray,
        dir: str,
        device=torch.device("cpu"),
    ):
        """
        The input V is a eigenvectors which are row-vector.
        Given feature dim: f, V ~ [f, f/2]
        """
        x_grid_dim, y_grid_dim, _ = grid_tensor.shape

        ### Load states
        # the path is likely to be: args.path_allStates and direction of agent
        # should come afterwards
        num_vec, _ = V.shape

        # DO NOT CARE AGENT DIR
        agent_dirs = [0, 1, 2, 3]

        grid_tensor = torch.from_numpy(grid_tensor)

        features = torch.zeros(x_grid_dim, y_grid_dim, feature_dim)
        deltaPhi = torch.zeros(len(agent_dirs), x_grid_dim, y_grid_dim, feature_dim)

        # will avg across agent_dirs
        rewards = torch.zeros(num_vec, x_grid_dim, y_grid_dim)

        for x, y in zip(coords[0], coords[1]):
            # # Load the image as a NumPy array
            img = grid_tensor.clone()
            img[x, y, :] = 10  # 10 is an agent

            with torch.no_grad():
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).to(self._dtype).to(self.device)
                if len(img.shape) == 3:
                    img = img[None, :, :, :].to(self._dtype).to(self.device)

                agent_pos = torch.tensor([[x, y]]).to(self._dtype).to(self.device)

                phi, _ = feaNet(img, agent_pos)
            features[x, y, :] = phi

        ### COMPUTE DELTA-PHI
        coordinates = np.stack((coords[0], coords[1]), axis=-1)
        for agent_dir in agent_dirs:
            """
            agent_dir 0: left
            agent_dir 1: up
            agent_dir 2: right
            agent_dir 3: down
            """
            for x, y in zip(coords[0], coords[1]):
                if agent_dir == 0:
                    temp_x, temp_y = x, y - 1
                    if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
                        deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
                    else:
                        deltaPhi[agent_dir, x, y, :] += features[x, y, :]
                elif agent_dir == 1:
                    temp_x, temp_y = x - 1, y
                    if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
                        deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
                    else:
                        deltaPhi[agent_dir, x, y, :] += features[x, y, :]
                elif agent_dir == 2:
                    temp_x, temp_y = x, y + 1
                    if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
                        deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
                    else:
                        deltaPhi[agent_dir, x, y, :] += features[x, y, :]
                elif agent_dir == 3:
                    temp_x, temp_y = x + 1, y
                    if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
                        deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
                    else:
                        deltaPhi[agent_dir, x, y, :] += features[x, y, :]

        # sum all connected next_phi - current phi
        deltaPhi -= features
        deltaPhi = torch.mean(deltaPhi, axis=0)  # [x, y, f]

        if algo_name == "SNAC":
            r_deltaPhi, s_deltaPhi = torch.split(
                deltaPhi, deltaPhi.size(-1) // 2, dim=-1
            )

            for vec_idx in range(num_vec):
                # deltaPhi ~ [n_possible_states, f/2]
                # V ~ [1, f/2]
                if vec_idx < int(num_vec / 2):
                    reward = torch.sum(
                        torch.mul(r_deltaPhi[:, :, :], V[vec_idx, :]),
                        axis=-1,
                    )
                else:
                    reward = torch.sum(
                        torch.mul(s_deltaPhi[:, :, :], V[vec_idx, :]),
                        axis=-1,
                    )

                # rewards[vec_idx, :, :] = reward
                # center the rew min to 0
                # rew_min = torch.min(reward.reshape(-1))
                # reward -= rew_min

                for x, y in zip(coords[0], coords[1]):
                    rewards[vec_idx, x, y] += reward[x, y]
        else:
            for vec_idx in range(num_vec):
                # deltaPhi ~ [n_possible_states, f]
                # V ~ [1, f]
                reward = torch.sum(
                    torch.mul(deltaPhi[:, :, :], V[vec_idx, :]),
                    axis=-1,
                )

                for x, y in zip(coords[0], coords[1]):
                    rewards[vec_idx, x, y] += reward[x, y]

        for k in range(num_vec):
            pos_rewards = rewards[k, :, :] > 0
            neg_rewards = rewards[k, :, :] < 0

            r_pos_max = torch.max(rewards[k, pos_rewards], axis=-1)[0]
            r_neg_min = torch.min(rewards[k, neg_rewards], axis=-1)[0]

            rewards[k, pos_rewards] = (rewards[k, pos_rewards] - 1e-10) / (
                r_pos_max - 1e-10
            )
            rewards[k, neg_rewards] = (rewards[k, neg_rewards] + 1e-10) / (
                1e-10 - r_neg_min
            )

        # Define a custom colormap with black at the center
        colors = [
            (0.2, 0.2, 1),
            (0.0, 0.2, 0.1),
            (1, 0.2, 0.2),
        ]  # Blue -> Black -> Red
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pale_blue_dark_brown_pale_red", colors
        )

        vec_dir_path = os.path.join(dir, "rewardMap")
        os.mkdir(vec_dir_path)
        for vec_idx in range(num_vec):
            grid = torch.zeros(self.grid_size, self.grid_size)
            grid += rewards[vec_idx, :, :]

            x = np.linspace(0, self.grid_size - 1, self.grid_size)
            y = np.linspace(0, self.grid_size - 1, self.grid_size)
            x, y = np.meshgrid(x, y)

            # Create the figure and two subplots: one for the 3D plot and one for the 2D heatmap
            fig = plt.figure(figsize=(18, 6))  # Adjust figsize as needed

            # reassign the agent
            img = grid_tensor.clone()
            # img[loc[0], loc[1]] = 10

            ax0 = fig.add_subplot(131)
            ax0.imshow(img * 20)
            ax0.axis("off")  # Turn off the axis for the image
            ax0.invert_yaxis()  # Invert the y-axis

            # Second subplot: 3D surface plot in the middle
            ax1 = fig.add_subplot(132, projection="3d")
            ax1.plot_surface(x, y, grid.numpy(), cmap=cmap)

            # Third subplot: 2D heatmap on the right
            ax2 = fig.add_subplot(133)
            ax2.axis("off")  # Turn off the axis for the image
            heatmap = ax2.imshow(
                grid.numpy(), cmap=cmap, extent=[-9, 9, -9, 9], origin="lower"
            )
            fig.colorbar(heatmap, ax=ax2)  # Add color bar for the heatmap

            # Save the plot with both the 3D surface and the 2D heatmap
            plt.savefig(f"{vec_dir_path}/{vec_idx}_{S[vec_idx]:3f}.png")
            plt.close()

    def plotActionValueMap(
        self,
        feaNet: nn.Module,
        psiNet: nn.Module,
        S: torch.Tensor,
        V: torch.Tensor,
        z: int,
        grid_tensor: np.ndarray,
        coords: tuple,
        loc: np.ndarray,
        specific_path: str,
    ):
        """
        The input V is a eigenvectors which are row-vector.
        Given feature dim: f, V ~ [f, f/2]
        """
        x_grid_dim, y_grid_dim, _ = grid_tensor.shape

        ### Load states
        # the path is likely to be: args.path_allStates and direction of agent
        # should come afterwards
        num_vec, feature_dim = V.shape
        feature_dim = int(2 * feature_dim)

        # DO NOT CARE AGENT DIR
        action_dim = 4
        grid_tensor = torch.from_numpy(grid_tensor)
        psiArray = torch.zeros(x_grid_dim, y_grid_dim, action_dim, feature_dim)

        # will avg across agent_dirs
        rewards = torch.zeros(num_vec, x_grid_dim, y_grid_dim)
        actions = torch.zeros(num_vec, x_grid_dim, y_grid_dim)

        for x, y in zip(coords[0], coords[1]):
            img = grid_tensor.clone()
            img[x, y, :] = 10  # 10 is an agent
            with torch.no_grad():
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).to(self._dtype).to(self.device)
                if len(img.shape) == 3:
                    img = img[None, :, :, :].to(self._dtype).to(self.device)
                phi, _ = feaNet(img)
                psi, _ = psiNet(phi, z)  # dim ~ [N, A, F]
            psiArray[x, y, :, :] = psi

        psi_r, psi_s = torch.split(psiArray, psiArray.size(-1) // 2, dim=-1)

        for vec_idx in range(num_vec):
            # deltaPhi ~ [n_possible_states, f/2]
            # V ~ [1, f/2]
            if vec_idx < int(num_vec / 2):
                reward = torch.sum(
                    torch.mul(psi_r[:, :, :, :], V[vec_idx, :]),
                    axis=-1,
                )
            else:
                reward = torch.sum(
                    torch.mul(psi_s[:, :, :, :], V[vec_idx, :]),
                    axis=-1,
                )

            reward, idx = torch.max(reward, axis=-1)
            reward[reward <= 0] = torch.tensor(0.0)

            for x, y in zip(coords[0], coords[1]):
                rewards[vec_idx, x, y] += reward[x, y]
                actions[vec_idx, x, y] += idx[x, y]

        ### post-processing
        r_min = torch.min(rewards.reshape(rewards.shape[0], -1), axis=-1)[0]
        r_max = torch.max(rewards.reshape(rewards.shape[0], -1), axis=-1)[0]

        r_min = r_min[:, None, None]
        r_max = r_max[:, None, None]

        rewards = (rewards - r_min) / (r_max - r_min + 1e-10)

        action_x = torch.zeros(actions.shape)
        action_y = torch.zeros(actions.shape)

        for vec_idx in range(num_vec):
            for x, y in zip(coords[0], coords[1]):
                action_idx = actions[vec_idx, x, y]
                if action_idx == 0:
                    action_y[vec_idx, x, y] -= 1
                elif action_idx == 1:
                    action_x[vec_idx, x, y] -= 1
                elif action_idx == 2:
                    action_y[vec_idx, x, y] += 1
                elif action_idx == 3:
                    action_x[vec_idx, x, y] += 1

        # create a path
        vec_dir_path = os.path.join(self.log_dir, "actionValueMap")
        if not os.path.exists(vec_dir_path):
            os.mkdir(vec_dir_path)
        vec_dir_path = os.path.join(vec_dir_path, specific_path)
        os.mkdir(vec_dir_path)

        for vec_idx in range(num_vec):
            grid = torch.zeros(self.grid_size, self.grid_size)
            grid += rewards[vec_idx, :, :]

            x = np.linspace(0, self.grid_size - 1, self.grid_size)
            y = np.linspace(0, self.grid_size - 1, self.grid_size)
            x, y = np.meshgrid(x, y)

            # Create the figure and two subplots: one for the 3D plot and one for the 2D heatmap
            fig = plt.figure(figsize=(18, 6))  # Adjust figsize as needed

            # reassign the agent
            grid_tensor[loc[0], loc[1]] = 10

            ax0 = fig.add_subplot(131)
            ax0.imshow(grid_tensor * 20)
            ax0.axis("off")  # Turn off the axis for the image
            ax0.invert_yaxis()  # Invert the y-axis

            # Second subplot: 3D surface plot in the middle
            ax1 = fig.add_subplot(132, projection="3d")
            ax1.plot_surface(x, y, grid.numpy(), cmap="viridis")

            # Third subplot: 2D heatmap on the right
            ax2 = fig.add_subplot(133)
            ax2.quiver(
                x,
                y,
                action_x[vec_idx, :, :],
                action_y[vec_idx, :, :],
                color="white",
                pivot="middle",
            )
            # ax2.axis("off")  # Turn off the axis for the image
            heatmap = ax2.imshow(
                grid.numpy(),
                cmap="viridis",
                extent=[0, self.grid_size - 1, 0, self.grid_size - 1],
                origin="lower",
            )
            fig.colorbar(heatmap, ax=ax2)  # Add color bar for the heatmap

            # Save the plot with both the 3D surface and the 2D heatmap
            plt.savefig(f"{vec_dir_path}/{vec_idx}_{S[vec_idx]:3f}.png")
            plt.close()

    def plotActionValueMap2(
        self,
        feaNet: nn.Module,
        psiNet: nn.Module,
        S: torch.Tensor,
        V: torch.Tensor,
        z: int,
        grid_tensor: np.ndarray,
        coords: tuple,
        loc: np.ndarray,
        specific_path: str,
    ):
        """
        The input V is a eigenvectors which are row-vector.
        Given feature dim: f, V ~ [f, f/2]
        """
        x_grid_dim, y_grid_dim, _ = grid_tensor.shape

        ### Load states
        # the path is likely to be: args.path_allStates and direction of agent
        # should come afterwards
        num_vec, feature_dim = V.shape
        feature_dim = int(2 * feature_dim)

        # DO NOT CARE AGENT DIR
        action_dim = 4
        grid_tensor = torch.from_numpy(grid_tensor)
        psiArray = torch.zeros(x_grid_dim, y_grid_dim, action_dim, feature_dim)

        # will avg across agent_dirs
        rewards = torch.zeros(num_vec, x_grid_dim, y_grid_dim)
        actions = torch.zeros(num_vec, x_grid_dim, y_grid_dim)

        for x, y in zip(coords[0], coords[1]):
            img = grid_tensor.clone()
            img[x, y, :] = 10  # 10 is an agent
            with torch.no_grad():
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).to(self._dtype).to(self.device)
                if len(img.shape) == 3:
                    img = img[None, :, :, :].to(self._dtype).to(self.device)
                phi, _ = feaNet(img)
                psi, _ = psiNet(phi, z)  # dim ~ [N, A, F]
            psiArray[x, y, :, :] = psi

        psi_r, psi_s = torch.split(psiArray, psiArray.size(-1) // 2, dim=-1)

        for vec_idx in range(num_vec):
            # deltaPhi ~ [n_possible_states, f/2]
            # V ~ [1, f/2]
            if vec_idx < int(num_vec / 2):
                reward = torch.sum(
                    torch.mul(psi_r[:, :, :, :], V[vec_idx, :]),
                    axis=-1,
                )
            else:
                reward = torch.sum(
                    torch.mul(psi_s[:, :, :, :], V[vec_idx, :]),
                    axis=-1,
                )

            reward, idx = torch.max(reward, axis=-1)
            reward[reward <= 0] = torch.tensor(0.0)

            for x, y in zip(coords[0], coords[1]):
                rewards[vec_idx, x, y] += reward[x, y]
                actions[vec_idx, x, y] += idx[x, y]

        ### post-processing
        r_min = torch.min(rewards.reshape(rewards.shape[0], -1), axis=-1)[0]
        r_max = torch.max(rewards.reshape(rewards.shape[0], -1), axis=-1)[0]

        r_min = r_min[:, None, None]
        r_max = r_max[:, None, None]

        rewards = (rewards - r_min) / (r_max - r_min + 1e-10)

        action_x = torch.zeros(actions.shape)
        action_y = torch.zeros(actions.shape)

        for vec_idx in range(num_vec):
            for x, y in zip(coords[0], coords[1]):
                action_idx = actions[vec_idx, x, y]
                if action_idx == 0:
                    action_y[vec_idx, x, y] -= 1
                elif action_idx == 1:
                    action_x[vec_idx, x, y] -= 1
                elif action_idx == 2:
                    action_y[vec_idx, x, y] += 1
                elif action_idx == 3:
                    action_x[vec_idx, x, y] += 1

        # create a path
        vec_dir_path = os.path.join(self.log_dir, "actionValueMap")
        if not os.path.exists(vec_dir_path):
            os.mkdir(vec_dir_path)

        for vec_idx in range(num_vec):
            grid = torch.zeros(self.grid_size, self.grid_size)
            grid += rewards[vec_idx, :, :]

            x = np.linspace(0, self.grid_size - 1, self.grid_size)
            y = np.linspace(0, self.grid_size - 1, self.grid_size)
            x, y = np.meshgrid(x, y)

            # Create the figure and two subplots: one for the 3D plot and one for the 2D heatmap
            fig = plt.figure(figsize=(18, 6))  # Adjust figsize as needed

            # reassign the agent
            grid_tensor[loc[0], loc[1]] = 10

            ax0 = fig.add_subplot(131)
            ax0.imshow(grid_tensor * 20)
            ax0.axis("off")  # Turn off the axis for the image
            ax0.invert_yaxis()  # Invert the y-axis

            # Second subplot: 3D surface plot in the middle
            ax1 = fig.add_subplot(132, projection="3d")
            ax1.plot_surface(x, y, grid.numpy(), cmap="viridis")

            # Third subplot: 2D heatmap on the right
            ax2 = fig.add_subplot(133)
            ax2.quiver(
                x,
                y,
                action_x[vec_idx, :, :],
                action_y[vec_idx, :, :],
                color="white",
                pivot="middle",
            )
            # ax2.axis("off")  # Turn off the axis for the image
            heatmap = ax2.imshow(
                grid.numpy(),
                cmap="viridis",
                extent=[0, self.grid_size - 1, 0, self.grid_size - 1],
                origin="lower",
            )
            fig.colorbar(heatmap, ax=ax2)  # Add color bar for the heatmap

            # Save the plot with both the 3D surface and the 2D heatmap
            plt.savefig(f"{vec_dir_path}/{specific_path}_{S[vec_idx]:3f}.png")
            plt.close()
