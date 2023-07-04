import os
from os import path as os_path

import torch
from torch import cuda
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import AmazonFullDataset as AFDataset
from datasets import AmazonReviewDataset as ARDataset
from datasets import AmazonSentenceDataset as ASDataset
from Q import (
    D,
    Epsilon,
    F_train_one_batch,
    R,
    S,
    Step,
    Storage,
    summarize_input,
    train_one_batch,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-j", "--json", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("-t", "--timesteps", type=int, default=120)
    parser.add_argument("-hi", "--hidden", type=int, default=600)
    parser.add_argument("-th", "--threshold", type=int, default=500)
    parser.add_argument("-g", "--gru", type=int, default=3)
    parser.add_argument("-lr", "--lr", type=float, default=1e-3)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-r", "--reward", type=float, default=0.005)
    parser.add_argument("-S", "--sentence", action="store_true")
    parser.add_argument("-F", "--full", action="store_true")
    parser.add_argument("-md", "--Mdecay", type=float, default=0.25)
    parser.add_argument("-ed", "--Edecay", type=float, default=0.0)
    parser.add_argument("-ep", "--epsilon", type=float, default=0.1)
    parser.add_argument("-c", "--categorical", action="store_true")
    parser.add_argument("-E", "--step", type=float, default=0.5)
    parser.add_argument("-sy", "--sync", type=int, default=100)
    parser.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("-W", "--weight_dir", type=str, default="weight_dir")
    parser.add_argument("-sl", "--selflearn", action="store_true")
    parser.add_argument("-ss", "--singlestep", action="store_true")
    parser.add_argument("-tdl", "--tdlambda", type=float, default=-1)
    parser.add_argument("-ro", "--rouge", action="store_true")
    parser.add_argument("--off_gpu", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs
    json_file = args.json
    batch_size = args.batch
    hidden_size = args.hidden
    timesteps = args.timesteps
    threshold = args.threshold
    full = args.full
    lr = args.lr
    sReward = args.reward
    step_E = args.step
    num_layers = args.gru
    device = args.device if cuda.is_available() else "cpu"
    on_gpu = not args.off_gpu
    memory_decay = args.Mdecay
    epsilon_decay = np.exp2(-args.Edecay)
    epsilon_0 = args.epsilon
    categorical = args.categorical
    syncOn = args.sync
    multistep = not args.singlestep
    tdlambda = args.tdlambda
    weight_dir = args.weight_dir
    RTeacher = not args.selflearn
    rouge = args.rouge
    predict = args.predict

    if full:
        """
        if full:
            data is not transposed: shape -> (batch, timesteps)
        else:
            data is transposed: shape -> (timesteps, batch)
        """

        dataset = AFDataset(
            filename=json_file,
            threshold=threshold,
            timesteps=timesteps,
            device=device,
            on_gpu=on_gpu,
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    else:
        if args.sentence:
            dataset = ASDataset(
                filename=json_file,
                threshold=threshold,
                batch_size=batch_size,
                timesteps=timesteps,
                device=device,
                on_gpu=on_gpu,
            )
        else:
            dataset = ARDataset(
                filename=json_file,
                threshold=threshold,
                batch_size=batch_size,
                device=device,
                on_gpu=on_gpu,
            )
        data_loader = DataLoader(
            dataset, batch_size=timesteps, shuffle=True, drop_last=True
        )

    SOS = dataset.to_index("__SOS__")
    PAD = dataset.to_index("__PAD__")

    torch.save(obj=dataset.pair._to_index, f=os_path.join(weight_dir, "to_index.pt"))
    torch.save(obj=dataset.pair._to_word, f=os_path.join(weight_dir, "to_word.pt"))

    voc_size = dataset.size

    dis = D(
        voc_size=voc_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        on=device,
    ).to(device)

    gen_update = G(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        sos_pad=(SOS, PAD),
        on=device,
        num_layers=num_layers,
    ).to(device)

    gen_target = G(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        sos_pad=(SOS, PAD),
        on=device,
        num_layers=num_layers,
    ).to(device)

    rec = R(
        voc_size=voc_size,
        hidden_size=hidden_size,
        timesteps=timesteps,
        on=device,
        sos=SOS,
        num_layers=num_layers,
    ).to(device)

    dis_optim = Adam(params=dis.parameters(), lr=lr)
    gen_optim = Adam(params=gen_update.parameters(), lr=lr)
    rec_optim = Adam(params=rec.parameters(), lr=lr)

    dis_loss = BCELoss()
    gen_loss = SmoothL1Loss()
    rec_loss = CrossEntropyLoss()

    if full:
        scr = S(
            voc_size=voc_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            on=device,
        ).to(device)
        scr_optim = Adam(params=scr.parameters(), lr=lr)
        scr_loss = CrossEntropyLoss()

    epsilon = Epsilon(epsilon_0, epsilon_decay)
    step = Step(E=step_E)
    storage = Storage(memory_decay, multistep=multistep, tdlambda=tdlambda)

    os.makedirs(weight_dir, exist_ok=True)

    for epoch in range(1, 1 + epochs):
        print("Epoch: {}/{}".format(epoch, epochs))
        i = 0
        for batch in data_loader:
            i %= syncOn
            sync = i == 0
            i += 1
            if full:
                F_train_one_batch(
                    batched_data=batch,
                    D=dis,
                    GUpdate=gen_update,
                    GTarget=gen_target,
                    R=rec,
                    S=scr,
                    Doptim=dis_optim,
                    Goptim=gen_optim,
                    Roptim=rec_optim,
                    Soptim=scr_optim,
                    DLossFunc=dis_loss,
                    GLossFunc=gen_loss,
                    RLossFunc=rec_loss,
                    SLossFunc=scr_loss,
                    epsilon=epsilon,
                    stepFunc=step,
                    storage=storage,
                    sReward=sReward,
                    sync=sync,
                    RTeacher=RTeacher,
                    categorical=categorical,
                    rouge=rouge,
                )
            else:
                train_one_batch(
                    batched_data=batch,
                    D=dis,
                    GUpdate=gen_update,
                    GTarget=gen_target,
                    R=rec,
                    Doptim=dis_optim,
                    Goptim=gen_optim,
                    Roptim=rec_optim,
                    DLossFunc=dis_loss,
                    GLossFunc=gen_loss,
                    RLossFunc=rec_loss,
                    epsilon=epsilon,
                    stepFunc=step,
                    storage=storage,
                    sReward=sReward,
                    sync=sync,
                    RTeacher=RTeacher,
                    categorical=categorical,
                )

        torch.save(
            obj=dis.state_dict(),
            f=os_path.join(weight_dir, "Discriminator_{:03d}.pt".format(epoch)),
        )
        torch.save(
            obj=gen_update.state_dict(),
            f=os_path.join(weight_dir, "Generator_{:03d}.pt".format(epoch)),
        )
        torch.save(
            obj=rec.state_dict(),
            f=os_path.join(weight_dir, "Reconstructor_{:03d}.pt".format(epoch)),
        )
        # for prediction use
        torch.save(obj=dis.state_dict(), f=os_path.join(weight_dir, "Discriminator.pt"))
        torch.save(
            obj=gen_update.state_dict(), f=os_path.join(weight_dir, "Generator.pt")
        )
        torch.save(obj=rec.state_dict(), f=os_path.join(weight_dir, "Reconstructor.pt"))

        if full:
            torch.save(
                obj=scr.state_dict(),
                f=os_path.join(weight_dir, "ScorePredict_{:03d}.pt".format(epoch)),
            )
            torch.save(
                obj=scr.state_dict(), f=os_path.join(weight_dir, "ScorePredict.pt")
            )

    if predict:
        summarize_input(
            G=gen_target,
            weight_dir=weight_dir,
            on="cpu",
            wi_iw=[dataset.pair._to_index, dataset.pair._to_word],
        )
