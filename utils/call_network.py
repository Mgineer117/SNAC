import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Union

from models.layers import (
    VAE,
    ConvNetwork,
    PsiCritic,
    OptionPolicy,
    OptionCritic,
    PsiCritic2,
    HC_Policy,
    HC_PrimitivePolicy,
    HC_Critic,
    PPO_Policy,
    PPO_Critic,
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
                "activation": nn.ELU(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.ELU(),
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
                "activation": nn.ELU(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.ELU(),
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
                "activation": nn.ELU(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.ELU(),
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
                "activation": nn.ELU(),
                "in_filters": in_channels,
                "out_filters": 32,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            {
                "type": "conv",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "activation": nn.ELU(),
                "in_filters": 64,
                "out_filters": 128,
            },  # Halve spatial dimensions again
        ]

    return encoder_conv_layers, decoder_conv_layers


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def call_ppoNetwork(args):
    from models.policy import PPO_Learner

    if args.import_ppo_model:
        print("Loading previous PPO parameters....")
        policy, critic = pickle.load(
            open("log/eval_log/model_for_eval/ppo_model.p", "rb")
        )
    else:
        policy = PPO_Policy(
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

    policy = PPO_Learner(
        policy=policy,
        critic=critic,
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

    pass


def call_sfNetwork(args):
    from models.policy import SF_Combined, SF_Split

    if args.algo_name in ("SNAC", "SNAC+", "SNAC++"):
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
        if args.algo_name in ("SNAC", "SNAC+", "SNAC++"):
            optionPolicy, optionCritic, option_vals, options = pickle.load(
                open(f"log/eval_log/model_for_eval/{args.env_name}/op_SNAC.p", "rb")
            )
        else:
            optionPolicy, optionCritic, option_vals, options = pickle.load(
                open(f"log/eval_log/model_for_eval/{args.env_name}/op_Spatial.p", "rb")
            )
    else:
        optionPolicy = OptionPolicy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            num_options=options.shape[0],
            activation=nn.Tanh(),
            is_discrete=args.is_discrete,
        )
        optionCritic = OptionCritic(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            num_options=options.shape[0],
            activation=nn.Tanh(),
        )

    use_psi_action = True if args.Psi_epoch > 0 else False

    policy = OP_Controller(
        sf_network=sf_network,
        optionPolicy=optionPolicy,
        optionCritic=optionCritic,
        algo_name=args.algo_name,
        options=options,
        option_vals=option_vals,
        use_psi_action=use_psi_action,
        a_dim=args.a_dim,
        policy_lr=args.op_policy_lr,
        critic_lr=args.op_critic_lr,
        entropy_scaler=args.op_entropy_scaler,
        eps=args.eps_clip,
        tau=args.tau,
        gamma=args.gamma,
        K=args.OP_K_epochs,
        is_discrete=args.is_discrete,
        device=args.device,
    )

    return policy


def call_hcNetwork(sf_network, op_network, args):
    from models.policy import HC_Controller

    if args.import_hc_model:
        print("Loading previous HC parameters....")
        policy, primitivePolicy, critic = pickle.load(
            open("log/eval_log/model_for_eval/hc_model.p", "rb")
        )
    else:
        policy = HC_Policy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            num_options=args.num_vector,
            activation=nn.Tanh(),
        )
        primitivePolicy = HC_PrimitivePolicy(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.Tanh(),
        )
        critic = HC_Critic(
            input_dim=args.s_flat_dim,
            fc_dim=args.fc_dim,
            activation=nn.Tanh(),
        )

    policy = HC_Controller(
        sf_network=sf_network,
        op_network=op_network,
        policy=policy,
        primitivePolicy=primitivePolicy,
        critic=critic,
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
