import numpy as np
import torch

from src import dataset


def train_iter(batch, model, optimizer, loss_func, device) -> float:
    optimizer.zero_grad()

    images1, images2, labels = dataset.flatten(batch)
    feats1, feats2 = model(images1.cuda(device), images2.cuda(device))
    error = loss_func(feats1, feats2, labels.cuda(device))

    error.backward()
    optimizer.step()

    return error.data.cpu().numpy()


def evaluate(dataloader_val, model, loss, device, n_iters=2):
    model.eval()

    error = 0
    cm_total = np.zeros((2, 2))

    with torch.no_grad():
        for ix, batch in enumerate(dataloader_val):
            if ix >= n_iters:
                break

            images1, images2, labels = dataset.flatten(batch)
            feats1, feats2 = model(images1.cuda(device), images2.cuda(device))
            error += loss(feats1, feats2, labels.cuda(device))

        error /= ix

    model.train(True)
    return error, cm_total


def run_training_loop(
        model,
        dataloader_train, dataloader_val,
        loss_func, optimizer,
        writer, device, weights_folder,
        iter_start=0, max_epochs=100, val_every=50, save_every=500,
) -> None:
    iter_cntr = iter_start
    epoch = 0
    model.train(True)

    try:
        while epoch < max_epochs:
            for batch in dataloader_train:
                iter_cntr += 1

                error_train = train_iter(
                    batch=batch,
                    model=model,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    device=device
                )

                writer.add_scalar('train.loss', error_train, iter_cntr)

                if iter_cntr % val_every == 0:
                    error_val, cm = evaluate(
                        dataloader_val, model=model, loss=loss_func, device=device, n_iters=2
                    )
                    writer.add_scalar('val.loss', error_val, iter_cntr)

                if iter_cntr % save_every == 0:
                    torch.save(model.state_dict(), "%s/feat-%d.pt" % (weights_folder, iter_cntr))

            epoch += 1

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "%s/feat-%d.pt" % (weights_folder, iter_cntr))
