import numpy as np
import torch
import torch.nn as nn
import pickle

import gymnasium as gym

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


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def call_sfNetwork(args):
    from models.policy import SF_Combined, SF_Split

    if args.algo_name == "SNAC":
        if args.import_sf_model:
            print("Loading previous SF parameters....")
            convNet, psiNet, options = pickle.load(
                open("log/eval_log/model_for_eval/sf_model.p", "rb")
            )
            feaNet = convNet
        else:

            psiNet = PsiCritic(
                fc_dim=args.fc_dim,
                sf_dim=args.sf_dim,
                a_dim=args.a_dim,
                activation=nn.Tanh(),
            )

            options = None

            if hasattr(args, "feat_net_type") and args.feat_net_type == "VAE":
                feaNet = VAE(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    fc_dim=args.conv_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=int(args.sf_dim / 2),
                    activation=nn.Tanh(),
                )
            else:
                feaNet = ConvNetwork(
                    state_dim=args.s_dim,
                    action_dim=args.a_dim,
                    fc_dim=args.conv_fc_dim,
                    sf_dim=args.sf_dim,
                    decoder_inpuit_dim=int(args.sf_dim / 2),
                    activation=nn.Tanh(),
                )

        policy = SF_Split(
            feaNet=feaNet,
            psiNet=psiNet,
            options=options,
            feature_lr=args.feature_lr,
            option_lr=args.option_lr,
            psi_lr=args.psi_lr,
            update_iter=args.update_iter,
            trj_per_iter=args.trj_per_iter,
            a_dim=args.a_dim,
            device=args.device,
        )
    else:
        if args.import_sf_model:
            print("Loading previous SF parameters....")
            convNet, psiNet, options = pickle.load(
                open("log/eval_log/model_for_eval/sf_model.p", "rb")
            )
        else:
            convNet = ConvNetwork(
                state_dim=args.s_dim,
                action_dim=args.a_dim,
                fc_dim=args.conv_fc_dim,
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
            feaNet=convNet,
            psiNet=psiNet,
            options=options,
            feature_lr=args.feature_lr,
            option_lr=args.option_lr,
            psi_lr=args.psi_lr,
            update_iter=args.update_iter,
            trj_per_iter=args.trj_per_iter,
            a_dim=args.a_dim,
            device=args.device,
        )

    return policy


def call_ppoNetwork(sf_network: nn.Module, args):
    from models.policy import PPO_Learner

    if args.import_ppo_model:
        print("Loading previous PPO parameters....")
        optionPolicy, optionCritic = pickle.load(
            open("log/eval_log/model_for_eval/ppo_model.p", "rb")
        )
    else:
        optionPolicy = PPO_Policy(
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.Tanh(),
        )
        optionCritic = PPO_Critic(
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            activation=nn.Tanh(),
        )

    policy = PPO_Learner(
        policy=optionPolicy,
        critic=optionCritic,
        convNet=sf_network.feaNet,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        entropy_scaler=args.entropy_scaler,
        eps=args.eps_clip,
        tau=args.tau,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy


def call_opNetwork(
    sf_network: nn.Module,
    args,
    option_vals: torch.Tensor | None = None,
    options: torch.Tensor | None = None,
):
    from models.policy import OP_Controller

    if args.import_op_model:
        print("Loading previous OP parameters....")
        optionPolicy, optionCritic, psiNet, option_vals, options = pickle.load(
            open("log/eval_log/model_for_eval/op_model.p", "rb")
        )
    else:
        optionPolicy = OptionPolicy(
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            num_options=options.shape[0],
            activation=nn.Tanh(),
        )
        optionCritic = OptionCritic(
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            num_options=options.shape[0],
            activation=nn.Tanh(),
        )
        psiNet = PsiCritic2(
            fc_dim=args.fc_dim,
            sf_dim=args.sf_dim,
            a_dim=args.a_dim,
            num_options=options.shape[0],
            activation=nn.Tanh(),
        )

    policy = OP_Controller(
        optionPolicy=optionPolicy,
        optionCritic=optionCritic,
        convNet=sf_network.feaNet,
        psiNet=psiNet,
        algo_name=args.algo_name,
        options=options,
        option_vals=option_vals,
        a_dim=args.a_dim,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        psi_lr=args.psi_lr,
        entropy_scaler=args.entropy_scaler,
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


def call_hcNetwork(convNet, optionPolicy, args):
    from models.policy import HC_Controller

    if args.import_hc_model:
        print("Loading previous HC parameters....")
        policy, primitivePolicy, critic = pickle.load(
            open("log/eval_log/model_for_eval/hc_model.p", "rb")
        )
    else:
        policy = HC_Policy(
            # input_dim=np.prod(args.s_dim),
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            num_options=optionPolicy._num_options,
            activation=nn.Tanh(),
        )
        primitivePolicy = HC_PrimitivePolicy(
            # input_dim=np.prod(args.s_dim),
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            a_dim=args.a_dim,
            activation=nn.Tanh(),
        )
        critic = HC_Critic(
            # input_dim=np.prod(args.s_dim),
            input_dim=args.sf_dim,
            fc_dim=args.fc_dim,
            activation=nn.Tanh(),
        )

    policy = HC_Controller(
        policy=policy,
        primitivePolicy=primitivePolicy,
        critic=critic,
        convNet=convNet,
        optionPolicy=optionPolicy,
        a_dim=args.a_dim,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        entropy_scaler=args.entropy_scaler,
        eps=args.eps_clip,
        tau=args.tau,
        gamma=args.gamma,
        K=args.K_epochs,
        device=args.device,
    )

    return policy
