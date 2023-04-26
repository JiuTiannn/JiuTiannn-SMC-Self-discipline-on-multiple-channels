# Install

- Install Pytorch

# Usage

- In the first step, you need to create multi-channel objects (we'll use SMC-2 as an example).

  ```python
  # Your seeds
  manualSeed = 9  
  
  # Channel A
  train_A_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
  # ↓ Set seeds so that the data in each channel is sorted consistently ↓
  torch.manual_seed(manualSeed)
  g_A = torch.Generator()
  # ↑ Set seeds so that the data in each channel is sorted consistently ↑
  train_A_loader = data.DataLoader(train_A_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, generator=g_A)
  
  # Channel B
  train_B_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
  # ↓ Set seeds so that the data in each channel is sorted consistently ↓
  torch.manual_seed(manualSeed)
  g_B = torch.Generator()
  # ↑ Set seeds so that the data in each channel is sorted consistently ↑
  train_B_loader = data.DataLoader(train_B_set, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, generator=g_B)
  ```

- In the second step, you need to create the loader object and store the data queue.

  ```python
  from queue import Queue
  
  B_loader, pre_outs = None, Queue()
  ```

- In the third step, you need to add some judgments to the train() function. enable_running_stats() and disable_running_stats() are to prevent the data from passing through the BN layer twice (Ref. https://github.com/davda54/sam).

  ```python
  # now_step: Count how many steps you have taken so far
  # total_step: The total number of steps taken throughout the training process
  # model: Your model
  # criterion: Your basis loss function (we use a cross-entropy loss function)
  # optimizer: Your Optimizer
  # args.alpha: Upper limit of weighting ratio (hyperparameter of SMC)
  # args.T: Distillation temperature (hyperparameter of SMC)
  def train(train_A_loader, train_B_loader, model, criterion, optimizer, B_loader, pre_outs):
      if B_loader is None:
  		B_loader = enumerate(train_B_loader)
          index, (B_imgs, B_label) = next(B_loader)
      for batch_idx, (inputs, targets) in enumerate(train_A_loader):
          # Weighting ratio (hyperparameter of SMC-2)
          a = args.alpha * (1 - 0.5 * (1 + math.cos(math.pi * now_step / total_step)))
          
          # Obtain B-channel data
          try:
              index, (B_imgs, B_label) = next(B_loader)
          except StopIteration:
              B_loader = enumerate(train_B_loader)
              index, (B_imgs, B_label) = next(B_loader)
              
          # Obtaining soft labels for B-channel data
          enable_running_stats(model)
          with torch.no_grad():
              B_out = model(B_imgs).detach_()
              pre_outs.put(B_out)
              
          # Obtaining soft labels for A-channel data    
         	disable_running_stats(model)
          outputs = model(inputs)
          
          # Calculating the loss function
  	    if pre_outs.qsize() >= 2:
              pre_out = pre_outs.get()
              ce_loss = criterion(outputs, targets)
              dml_loss = (
                  F.kl_div(
                      F.log_softmax(outputs / args.T, dim=1),
                      F.softmax(pre_out / args.T, dim=1),
                      reduction="batchmean",
                  )
              )
              loss = (1 - a) * ce_loss + a * dml_loss
  		else:
              loss = criterion(outputs, targets)
  		
          optimizer.zero_grad()
          loss.mean().backward()
          optimizer.step()
          
          # Count how many steps you have taken so far
  		now_step += 1
          
  	return B_loader, pre_outs
  ```

  

# Training templates

- SMC-2.py is the sample we offer for overall training use.

- How to run

  ```lua
  python SMC-2.py -a vgg19_bn --dataset cifar100 --epochs 200 --wd 5e-4 --manualSeed 9 --checkpoint VGG19
  ```

# Cite

We have borrowed or used code from the following authors:

https://github.com/davda54/sam

https://github.com/bearpaw/pytorch-classification