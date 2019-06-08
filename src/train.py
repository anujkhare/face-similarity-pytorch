
def train_iter(batch, model, optimizer, loss_func, device) -> float:
    optimizer.zero_grad()

    images1, images2, labels = flatten(batch)
    probs = model(images1.cuda(device_id), images2.cuda(device_id))

    error = loss_func(probs, labels.cuda(device))
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

            images1, images2, labels = flatten(batch)
            probs = model(images1.cuda(device), images2.cuda(device))
            error += loss(probs, labels.cuda(device_id)).data.cpu().numpy()
            
            _, labels_pred = torch.max(probs, dim=1)
            cm_total += sklearn.metrics.confusion_matrix(labels.data.cpu().numpy(), labels_pred.data.cpu().numpy())

            # Plot the image
            visualize(batch, maxn=10)
            print('Prob', np.exp(probs[:, 1].data.cpu().numpy()))
            print('Pred', labels_pred)

    model.train(True)
    return error, cm_total