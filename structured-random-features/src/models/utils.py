import torch.nn.functional as F
import torch

def train(log_interval, device, model, train_loader, optimizer, epoch, loss_fn=F.cross_entropy, verbose=True):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_fn(output, target.to(device))
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        if verbose == True and batch_idx % log_interval == 0:
            print('Train_epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    return loss_list
        
            
def test(model, device, test_loader, loss_fn=F.cross_entropy, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data.to(device))
            test_loss += loss_fn(output, target.to(device), reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
    
    if verbose == True:
        print('\nTest set: Average loss: {:.6f}. Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, 
                                                                                     len(test_loader.dataset), test_accuracy))
    return test_accuracy