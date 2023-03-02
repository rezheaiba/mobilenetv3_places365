import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import dataloader_places365
from model_v3 import MobileNetV3
from utils import log


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = r'../'  # get data root path
    image_path = os.path.join(data_root, "places365_standard")  # flower data set path
    batch_size = 8
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = dataloader_places365.Loader(img_path=r'../places365_standard/train.txt',
                                                label_path=r'../places365_standard/IO_places365.txt',
                                                root=image_path,
                                                transforms_=data_transform["train"],
                                                )
    train_num = len(train_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    writer = SummaryWriter('./final')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = dataloader_places365.Loader(img_path=r'../places365_standard/val.txt',
                                                   label_path=r'../places365_standard/IO_places365.txt',
                                                   root=image_path,
                                                   transforms_=data_transform["val"],
                                                   )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = MobileNetV3(dilated=True, num_classes_1=2, num_classes_2=7,
                      arch='mobilenet_v3_small', reduced_tail=True,
                      width_mult=0.5)
	
	# freeze classifier2
	for k, v in net.named_parameters():
		if 'classifier2' in k:
			v.requires_grad = False
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.1)

    epochi = 0
    epochs = 101
    best_acc = 0.0
    save_path_best = './weight-final-v3/model_v3_best.pth'
    save_path_newest = './weight-final-v3/model_v3_newest.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    prev_time = time.time()

    # creat logger 用来保存训练以及验证过程中信息
    logger_file = "./weight-final-v3/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = log.create_logger(logger_file)

    if epochi != 0:
        net.load_state_dict(torch.load(save_path_newest, map_location=device))
    for epoch in range(epochi, epochs):
        # train
        net.train()
        running_loss = 0.0
        running_val_loss = 0.0
        train_bar = tqdm(train_loader)
        if epoch == 20:
            optimizer.param_groups[0]['lr'] = 1e-4
        for step, data in enumerate(train_bar):
            # forward
            optimizer.zero_grad()
            logits = net(data['image'].to(device))
            loss1 = loss_function(logits['classifier1'], data['label1'].to(device))
            loss2 = loss_function(logits['classifier2'], data['label2'].to(device))
            loss = (loss1 + loss2) / 2

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # left time
            batches_done = epoch * train_steps + step  # n_epochs所遍历的总batches
            batches_left = epochs * train_steps - batches_done  # 总共剩余的batches
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()  # 某epoch中某batch的时刻

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} left-time:{}".format(epoch + 1,
                                                                                  epochs,
                                                                                  loss,
                                                                                  time_left)
            writer.add_scalar('train_loss', loss, epoch * train_steps + step)

        # validate
        net.eval()
        acc1 = 0.0
        acc2 = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for step, val_data in enumerate(val_bar):
                outputs = net(val_data['image'].to(device))

                # torch.max(input,dim)是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值,return:values(值),indices(索引)
                predict_y_1 = torch.max(outputs['classifier1'], dim=1)[1]
                predict_y_2 = torch.max(outputs['classifier2'], dim=1)[1]

                # torch.ep：逐元素比较是否相等返回TorF/1or0
                acc1 += torch.eq(predict_y_1, val_data['label1'].to(device)).sum().item()
                acc2 += torch.eq(predict_y_2, val_data['label2'].to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                loss1 = loss_function(outputs['classifier1'], val_data['label1'].to(device))
                loss2 = loss_function(outputs['classifier2'], val_data['label2'].to(device))
                val_loss = (loss1 + loss2) / 2
                writer.add_scalar('val_loss', val_loss, epoch * val_steps + step)
                running_val_loss += val_loss.item()

        val_accurate_1 = acc1 / val_num
        val_accurate_2 = acc2 / val_num
        print(
            '[epoch %d] train_loss: %.3f val_loss: %.3f val_accurate_1: %.3f val_accurate_2: %.3f lr: %f' %
            (epoch + 1, running_loss / train_steps, running_val_loss / val_steps, val_accurate_1, val_accurate_2,
             optimizer.param_groups[0]['lr']))

        # save logging info
        logger.info(
            '[epoch %d] train_loss: %.3f val_loss: %.3f val_accurate_1: %.3f val_accurate_2: %.3f lr: %f' %
            (epoch + 1, running_loss / train_steps, running_val_loss / val_steps, val_accurate_1, val_accurate_2,
             optimizer.param_groups[0]['lr']))

        # 保存最新一次的权重
        torch.save(net.state_dict(), save_path_newest)

        # 保存准确率最高的权重
        val_accurate = (val_accurate_1 + val_accurate_2) / 2
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path_best)

    print('Finished Training')


if __name__ == '__main__':
    main()
