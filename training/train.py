from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device, epochs):

    for epoch in range(epochs):

        running_loss = 0.0

        for gray_imgs, color_imgs in tqdm(dataloader):

            color_imgs = color_imgs.to(device)
            gray_imgs = gray_imgs.to(device)

            predictions = model(gray_imgs)

            optimizer.zero_grad()

            loss = criterion(predictions, color_imgs)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss:.6f}")