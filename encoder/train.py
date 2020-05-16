from pathlib import Path

import torch

from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.hyperparameters import *
from encoder.model import SpeakerEncoder

def sync(device: torch.device):
    #FIXME
    return
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        SPEAKERS_PER_BATCH,
        UTTERANCES_PER_SPEAKER,
        num_workers=8
    )

    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_INIT)
    init_step = 1

    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = LEARNING_RATE_INIT
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting training from scratch.")
    model.train()

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    # Training loop
    for step, speaker_batch in enumerate(loader, init_step):
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        embed = model(inputs)
        sync(device)
        embeds_loss = embeds.view((SPEAKERS_PER_BATCH, UTTERANCES_PER_SPEAKER, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)

        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()

        # Update visualizations
        vis.update(loss.item(), eer, step)

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, UTTERANCES_PER_SPEAKER, step, projection_fpath)
            vis.save()
        
        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
        
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
