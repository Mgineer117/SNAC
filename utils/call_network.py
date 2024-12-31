import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Union

from utils.normalizer import ObservationNormalizer
from models.layers import (
    VAE,
    ConvNetwork,
    PsiCritic,
    OptionPolicy,
    OP_Critic,
    PsiCritic2,
    HC_Policy,
    HC_PPO,
    HC_RW,
    HC_Critic,
    SAC_Policy,
    SAC_Critic,
    SAC_CriticTwin,
    OP_CriticTwin,
    PPO_Policy,
    PPO_Critic,
    OC_Policy,
    OC_Critic,
)

from log.logger_util import colorize


def get_conv_layer(args):
    _, _, in_channels = args.s_dim

    if args.env_name == "Maze":
        encoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.Tanh(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 128,
                "out_filters": 128,
            },  # Halve spatial dimensions again
        ]

        decoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.Tanh(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 128,
                "out_filters": 128,
            },  # Halve spatial dimensions again
        ]

    else:
        encoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.Tanh(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
        ]

        decoder_conv_layers = [
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": nn.Tanh(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
        ]

    return encoder_conv_layers, decoder_conv_layers


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def call_sacNetwork(args):
    from models.policy import SAC_Learner

    if args.import_sac_model:
        print("Loading previous SAC parameters....")
        policy, critic_twin, alpha, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/sac_model.p", "rb")
        )
    else:
        # Define the Actor (Policy) network
        actor = SAC_Policy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.ReLU(),
            is_discrete=args.is_discrete,
        )

        # Define the Critic networks
        critic_twin = SAC_CriticTwin(
            input_dim=args.s_flat_dim + args.a_dim,
            fc_dim=args.fc_dim,
            activation=nn.ReLU(),
        )

        alpha = args.sac_init_alpha

        if args.obs_norm != "none":
            normalizer = ObservationNormalizer(state_dim=args.s_dim)
        else:
            normalizer = None
        
    # Create the SAC Learner
    policy = SAC_Learner(
        policy=actor,
        critic_twin=critic_twin,
        alpha=alpha,
        normalizer=normalizer,
        policy_lr=args.sac_policy_lr,
        critic_lr=args.sac_critic_lr,
        alpha_lr=args.sac_alpha_lr,
        tau=args.sac_soft_update_rate,
        gamma=args.gamma,
        trj_per_iter=args.sac_trj_per_iter,
        target_update_interval=args.target_update_interval,
        tune_alpha=args.tune_alpha,
        device=args.device,
    )

    return policy


def call_ppoNetwork(args):
    from models.policy import PPO_Learner

    if args.import_ppo_model:
        print("Loading previous PPO parameters....")
        policy, critic, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/ppo_model.p", "rb")
        )
    else:
        actor = PPO_Policy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.Tanh(),
            is_discrete=args.is_discrete,
        )
        critic = PPO_Critic(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            activation=nn.Tanh(),
        )

        if args.obs_norm != "none":
            normalizer = ObservationNormalizer(state_dim=args.s_dim)
        else:
            normalizer = None

    policy = PPO_Learner(
        policy=actor,
        critic=critic,
        normalizer=normalizer,
        policy_lr=args.ppo_policy_lr,
        critic_lr=args.ppo_critic_lr,
        entropy_scaler=args.ppo_entropy_scaler,
        eps=args.eps_clip,
        tau=args.tau,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_ocNetwork(args):
    from models.policy import OC_Learner

    if args.import_oc_model:
        print("Loading previous OC parameters....")
        policy, critic, normalizer = pickle.load(
            open(f"log/eval_log/model_for_eval/{args.env_name}/oc_model.p", "rb")
        )
    else:
        encoder_conv_layers, _ = get_conv_layer(args)
        policy = OC_Policy(
            state_dim=args.s_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            num_options=args.num_vector,
            encoder_conv_layers=encoder_conv_layers,
            activation=nn.Tanh(),
            is_discrete=args.is_discrete,
        )
        critic = OC_Critic(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            num_options=args.num_vector,
            activation=nn.Tanh(),
        )

        if args.obs_norm != "none":
            normalizer = ObservationNormalizer(state_dim=args.s_dim)
        else:
            normalizer = None

    policy = OC_Learner(
        policy=policy,
        critic=critic,
        normalizer=normalizer,
        policy_lr=args.ppo_policy_lr,
        critic_lr=args.ppo_critic_lr,
        entropy_scaler=args.ppo_entropy_scaler,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_sfNetwork(args, sf_path: str | None = None):
    from models.policy import SF_Combined, SF_Split

    if args.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
        if args.import_sf_model:
            print("Loading previous SF parameters....")
            feaNet, psiNet, options = pickle.load(
                open(f"log/eval_log/model_for_eval/{args.env_name}/sf_SNAC.p", "rb")
            )
        else:
            if args.env_name in ("PointNavigation"):
                msg = colorize(
                    "\nVAE Feature Extractor is selected!!!",
                    "yellow",
                    bold=True,
                )
                print(msg)
                feaNet = VAE(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    fc_dim=args.feature_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=int(args.sf_dim / 2),
                    is_snac=True,
                    activation=nn.Tanh(),
                )
            else:
                msg = colorize(
                    "\nCNN Feature Extractor is selected!!!",
                    "yellow",
                    bold=True,
                )
                print(msg)

                encoder_conv_layers, decoder_conv_layers = get_conv_layer(args)
                feaNet = ConvNetwork(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    agent_num=args.agent_num,
                    grid_size=args.grid_size,
                    encoder_conv_layers=encoder_conv_layers,
                    decoder_conv_layers=decoder_conv_layers,
                    fc_dim=args.feature_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=int(args.sf_dim / 2),
                    activation=nn.Tanh(),
                )

            psiNet = PsiCritic(
                fc_dim=args.fc_dim,
                sf_dim=args.sf_dim,
                a_dim=args.a_dim,
                activation=nn.Tanh(),
            )

            options = None

        policy = SF_Split(
            feaNet=feaNet,
            psiNet=psiNet,
            options=options,
            feature_lr=args.feature_lr,
            option_lr=args.option_lr,
            psi_lr=args.psi_lr,
            phi_loss_r_scaler=args.phi_loss_r_scaler,
            phi_loss_s_scaler=args.phi_loss_s_scaler,
            phi_loss_kl_scaler=args.phi_loss_kl_scaler,
            phi_loss_l2_scaler=args.phi_loss_l2_scaler,
            trj_per_iter=args.trj_per_iter,
            a_dim=args.a_dim,
            is_discrete=args.is_discrete,
            sf_path=sf_path,
            device=args.device,
        )
    else:
        if args.import_sf_model:
            print("Loading previous SF parameters....")
            feaNet, psiNet, options = pickle.load(
                open(f"log/eval_log/model_for_eval/{args.env_name}/sf_Spatial.p", "rb")
            )
        else:
            if args.env_name in ("PointNavigation"):
                msg = colorize(
                    "\nVAE Feature Extractor is selected!!!",
                    "yellow",
                    bold=True,
                )
                print(msg)
                feaNet = VAE(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    fc_dim=args.feature_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=args.sf_dim,
                    is_snac=False,
                    activation=nn.Tanh(),
                )
            else:
                msg = colorize(
                    "\nCNN Feature Extractor is selected!!!",
                    "yellow",
                    bold=True,
                )
                print(msg)
                encoder_conv_layers, decoder_conv_layers = get_conv_layer(args)
                feaNet = ConvNetwork(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    agent_num=args.agent_num,
                    grid_size=args.grid_size,
                    encoder_conv_layers=encoder_conv_layers,
                    decoder_conv_layers=decoder_conv_layers,
                    fc_dim=args.feature_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=args.sf_dim,
                    activation=nn.Tanh(),
                )

            psiNet = PsiCritic(
                fc_dim=args.fc_dim,
                sf_dim=args.sf_dim,
                a_dim=args.a_dim,
                activation=nn.Tanh(),
            )

            options = None

        policy = SF_Combined(
            feaNet=feaNet,
            psiNet=psiNet,
            options=options,
            feature_lr=args.feature_lr,
            option_lr=args.option_lr,
            psi_lr=args.psi_lr,
            phi_loss_r_scaler=args.phi_loss_r_scaler,
            phi_loss_s_scaler=args.phi_loss_s_scaler,
            phi_loss_kl_scaler=args.phi_loss_kl_scaler,
            phi_loss_l2_scaler=args.phi_loss_l2_scaler,
            trj_per_iter=args.trj_per_iter,
            a_dim=args.a_dim,
            is_discrete=args.is_discrete,
            device=args.device,
        )

    return policy


def call_opNetwork(
    sf_network: nn.Module,
    args,
    option_vals: Union[torch.Tensor, None] = None,
    options: Union[torch.Tensor, None] = None,
):
    from models.policy import OP_Controller

    if args.import_op_model:
        print("Loading previous OP parameters....")
        if args.op_mode == "sac":
            if args.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
                optionPolicy, optionCritic, option_vals, options, alpha, normalizer = pickle.load(
                    open(f"log/eval_log/model_for_eval/{args.env_name}/op_SNAC.p", "rb")
                )
            else:
                optionPolicy, optionCritic, option_vals, options, alpha, normalizer = pickle.load(
                    open(
                        f"log/eval_log/model_for_eval/{args.env_name}/op_Spatial.p",
                        "rb",
                    )
                )
        elif args.op_mode == "ppo":
            alpha = None
            if args.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
                optionPolicy, optionCritic, option_vals, options, normalizer = pickle.load(
                    open(f"log/eval_log/model_for_eval/{args.env_name}/op_SNAC.p", "rb")
                )
            else:
                optionPolicy, optionCritic, option_vals, options, normalizer = pickle.load(
                    open(
                        f"log/eval_log/model_for_eval/{args.env_name}/op_Spatial.p",
                        "rb",
                    )
                )
    else:
        if args.op_mode == "sac":
            optionPolicy = OptionPolicy(
                input_dim=args.s_flat_dim,
                fc_dim=args.option_fc_dim,
                a_dim=args.a_dim,
                num_options=options.shape[0],
                activation=nn.ReLU(),
                is_discrete=args.is_discrete,
            )
            optionCritic = OP_CriticTwin(
                input_dim=args.s_flat_dim + args.a_dim,
                fc_dim=args.fc_dim,
                num_options=options.shape[0],
                activation=nn.ReLU(),
            )
        else:
            optionPolicy = OptionPolicy(
                input_dim=args.s_flat_dim,
                fc_dim=args.option_fc_dim,
                a_dim=args.a_dim,
                num_options=options.shape[0],
                activation=nn.Tanh(),
                is_discrete=args.is_discrete,
            )
            optionCritic = OP_Critic(
                input_dim=args.s_flat_dim,
                fc_dim=args.fc_dim,
                num_options=options.shape[0],
                activation=nn.Tanh(),
            )
        alpha = None
        if args.obs_norm != "none":
            normalizer = ObservationNormalizer(state_dim=args.s_dim)
        else:
            normalizer = None

    optimizers = {}
    if args.op_mode == "sac":
        is_bfgs = False
        optimizers["policy"] = torch.optim.AdamW(
            optionPolicy.parameters(), lr=args.sac_policy_lr
        )
        optimizers["critic"] = torch.optim.AdamW(
            optionCritic.parameters(), lr=args.sac_critic_lr
        )
    elif args.op_mode == "ppo":
        if args.op_critic_lr is None:
            optimizers["ppo"] = torch.optim.AdamW(
                optionPolicy.parameters(), lr=args.op_policy_lr
            )
            is_bfgs = True
        else:
            optimizers["ppo"] = torch.optim.AdamW(
                [
                    {"params": optionPolicy.parameters(), "lr": args.op_policy_lr},
                    {"params": optionCritic.parameters(), "lr": args.op_critic_lr},
                ]
            )
            is_bfgs = False

    use_psi_action = True if args.Psi_epoch > 0 else False

    policy = OP_Controller(
        sf_network=sf_network,
        optionPolicy=optionPolicy,
        optionCritic=optionCritic,
        alpha=alpha,
        normalizer=normalizer,
        optimizers=optimizers,
        options=options,
        option_vals=option_vals,
        is_bfgs=is_bfgs,
        use_psi_action=use_psi_action,
        args=args,
    )

    return policy


def call_hcNetwork(sf_network, op_network, args):
    from models.policy import HC_Controller

    if args.import_hc_model:
        print("Loading previous HC parameters....")
        policy, primitivePolicy, critic, normalizer = pickle.load(
            open("log/eval_log/model_for_eval/hc_model.p", "rb")
        )
    else:
        policy = HC_Policy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            num_options=args.num_vector,
            activation=nn.Tanh(),
        )
        if args.PM_policy == "PPO":
            primitivePolicy = HC_PPO(
                input_dim=args.s_flat_dim,
                fc_dim=args.fc_dim,
                a_dim=args.a_dim,
                is_discrete=args.is_discrete,
                activation=nn.Tanh(),
            )
        elif args.PM_policy == "RW":
            primitivePolicy = HC_RW(
                a_dim=args.a_dim,
                is_discrete=args.is_discrete,
            )
        else:
            NotImplementedError(f"{args.PM_policy} is not implemented")
        critic = HC_Critic(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            activation=nn.Tanh(),
        )

        if args.obs_norm != "none":
            normalizer = ObservationNormalizer(state_dim=args.s_dim)
        else:
            normalizer = None

    policy = HC_Controller(
        sf_network=sf_network,
        op_network=op_network,
        policy=policy,
        primitivePolicy=primitivePolicy,
        critic=critic,
        normalizer=normalizer,
        a_dim=args.a_dim,
        policy_lr=args.hc_policy_lr,
        critic_lr=args.hc_critic_lr,
        entropy_scaler=args.hc_entropy_scaler,
        eps=args.eps_clip,
        tau=args.tau,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_rpNetwork(convNet, qNet, options, args):
    from models.policy import RandomWalk

    policy = RandomWalk(
        convNet=convNet,
        qNet=qNet,
        options=options,
        a_dim=args.a_dim,
        device=args.device,
    )

    return policy
